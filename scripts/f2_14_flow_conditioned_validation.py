from pathlib import Path
import json
import os
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
FEATURE_DIR = ROOT / "reports" / "f2" / "features"
SPLIT_DIR = ROOT / "reports" / "f2" / "splits"
MODEL_DIR = ROOT / "reports" / "f2" / "models"

CLASSIFICATION = RAW / "classification.csv.gz"
BIO_PATH = FEATURE_DIR / "node_biological_features.csv.gz"
TOPO_PATH = FEATURE_DIR / "node_topological_features.csv.gz"
TARGET_PATH = FEATURE_DIR / "node_targets.csv.gz"
BIO_SETS_PATH = FEATURE_DIR / "biological_feature_sets.json"
TOPO_SETS_PATH = FEATURE_DIR / "topological_feature_sets.json"
TOPO_MANIFEST_PATH = FEATURE_DIR / "topological_feature_manifest.csv"
MAIN_SPLIT_PATH = SPLIT_DIR / "main_split.csv.gz"

LOG_SELECTED_PATH = (
    MODEL_DIR / "logistic_validation" / "selected_logistic_configs.csv"
)
RF_SELECTED_PATH = (
    MODEL_DIR / "random_forest_validation" / "selected_random_forest_configs.csv"
)

OUT_DIR = MODEL_DIR / "flow_conditioned_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUT_DIR / "flow_conditioned_results.csv"
PER_CLASS_PATH = OUT_DIR / "flow_conditioned_per_class_metrics.csv"
CLASS_COUNTS_PATH = OUT_DIR / "flow_conditioned_class_counts.csv"
EXCLUSIONS_PATH = OUT_DIR / "flow_conditioned_exclusions.csv"
SUMMARY_TXT_PATH = OUT_DIR / "flow_conditioned_summary.txt"
SUMMARY_JSON_PATH = OUT_DIR / "flow_conditioned_summary.json"

RANDOM_STATE = 42
NT_IMPUTATION_SEED = 314159
MIN_TRAIN_SUPPORT_PER_CLASS = 20

MORPHOLOGY_COLS = ["length_nm", "area_nm", "size_nm"]
NT_PROB_COLS = [
    "da_avg",
    "ser_avg",
    "gaba_avg",
    "glut_avg",
    "ach_avg",
    "oct_avg",
]
SIDE_COL = "side"
NT_AVAILABLE_COL = "nt_available"

PRIMARY_EXPERIMENTS = [
    "bio_strict",
    "network_full",
    "bio_strict_plus_network_full",
]

SENSITIVITY_EXPERIMENTS = [
    "bio_plus_availability",
    "bio_plus_availability_plus_network_full",
]

ALL_EXPERIMENTS = PRIMARY_EXPERIMENTS + SENSITIVITY_EXPERIMENTS


def section(title):
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def log1p_array(x):
    x = np.asarray(x, dtype=np.float64)
    if np.nanmin(x) < 0:
        raise ValueError("log1p recebeu valor negativo.")
    return np.log1p(x)


class JointNTVectorImputer(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=NT_IMPUTATION_SEED):
        self.random_state = random_state

    def fit(self, X, y=None):
        x = np.asarray(X, dtype=np.float64)
        complete = np.isfinite(x).all(axis=1)
        partial = np.isfinite(x).any(axis=1) & ~complete
        if partial.any():
            raise ValueError("Vetores NT parcialmente ausentes encontrados.")
        self.valid_vectors_ = x[complete].copy()
        if len(self.valid_vectors_) == 0:
            raise ValueError("Nenhum vetor NT válido no treino.")
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, X):
        x = np.asarray(X, dtype=np.float64).copy()
        complete = np.isfinite(x).all(axis=1)
        partial = np.isfinite(x).any(axis=1) & ~complete
        if partial.any():
            raise ValueError("Vetores NT parcialmente ausentes encontrados.")

        missing_idx = np.flatnonzero(~complete)
        if len(missing_idx):
            seed = (
                int(self.random_state)
                + 104729 * len(x)
            ) % (2**32 - 1)
            rng = np.random.default_rng(seed)
            sampled = rng.integers(
                0,
                len(self.valid_vectors_),
                size=len(missing_idx),
            )
            x[missing_idx] = self.valid_vectors_[sampled]

        if not np.isfinite(x).all():
            raise RuntimeError("Imputação NT deixou NaN/infinito.")
        return x

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray(
                [f"x{i}" for i in range(self.n_features_in_)],
                dtype=object,
            )
        return np.asarray(input_features, dtype=object)


def build_feature_spec(experiment, bio_sets, topo_sets):
    if experiment == "bio_strict":
        return list(bio_sets["bio_strict"])
    if experiment == "bio_plus_availability":
        return list(bio_sets["bio_plus_availability"])
    if experiment == "network_full":
        return list(topo_sets["network_full"])
    if experiment == "bio_strict_plus_network_full":
        return (
            list(bio_sets["bio_strict"])
            + list(topo_sets["network_full"])
        )
    if experiment == "bio_plus_availability_plus_network_full":
        return (
            list(bio_sets["bio_plus_availability"])
            + list(topo_sets["network_full"])
        )
    raise ValueError(experiment)


def infer_topology_preprocessing(topo_manifest, cols):
    relevant = topo_manifest[
        topo_manifest["feature"].isin(cols)
    ]
    log_cols = relevant.loc[
        relevant["recommended_preprocessing"] == "log1p_then_scale",
        "feature",
    ].tolist()
    scale_cols = relevant.loc[
        relevant["recommended_preprocessing"] == "scale",
        "feature",
    ].tolist()
    return log_cols, scale_cols


def make_logistic_preprocessor(feature_cols, topo_manifest):
    transformers = []

    if SIDE_COL in feature_cols:
        transformers.append(
            (
                "side",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="unknown",
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                [SIDE_COL],
            )
        )

    morph = [c for c in MORPHOLOGY_COLS if c in feature_cols]
    if morph:
        transformers.append(
            (
                "morphology",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        (
                            "log1p",
                            FunctionTransformer(
                                log1p_array,
                                validate=False,
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                morph,
            )
        )

    nt = [c for c in NT_PROB_COLS if c in feature_cols]
    if nt:
        transformers.append(
            (
                "nt",
                Pipeline(
                    steps=[
                        ("imputer", JointNTVectorImputer()),
                        ("scale", StandardScaler()),
                    ]
                ),
                nt,
            )
        )

    if NT_AVAILABLE_COL in feature_cols:
        transformers.append(
            ("nt_available", "passthrough", [NT_AVAILABLE_COL])
        )

    topo_features = set(topo_manifest["feature"])
    topo_cols = [c for c in feature_cols if c in topo_features]
    if topo_cols:
        log_cols, scale_cols = infer_topology_preprocessing(
            topo_manifest,
            topo_cols,
        )
        if log_cols:
            transformers.append(
                (
                    "topology_log",
                    Pipeline(
                        steps=[
                            (
                                "log1p",
                                FunctionTransformer(
                                    log1p_array,
                                    validate=False,
                                ),
                            ),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    log_cols,
                )
            )
        if scale_cols:
            transformers.append(
                ("topology_scale", StandardScaler(), scale_cols)
            )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_rf_preprocessor(feature_cols, topo_feature_names):
    transformers = []

    if SIDE_COL in feature_cols:
        transformers.append(
            (
                "side",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="unknown",
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                [SIDE_COL],
            )
        )

    morph = [c for c in MORPHOLOGY_COLS if c in feature_cols]
    if morph:
        transformers.append(
            ("morphology", SimpleImputer(strategy="median"), morph)
        )

    nt = [c for c in NT_PROB_COLS if c in feature_cols]
    if nt:
        transformers.append(
            ("nt", JointNTVectorImputer(), nt)
        )

    if NT_AVAILABLE_COL in feature_cols:
        transformers.append(
            ("nt_available", "passthrough", [NT_AVAILABLE_COL])
        )

    topo_cols = [
        c for c in feature_cols
        if c in topo_feature_names
    ]
    if topo_cols:
        transformers.append(
            ("topology", "passthrough", topo_cols)
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def evaluate(y_true, y_pred, labels):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def per_class_frame(y_true, y_pred, labels, flow, family, experiment):
    p, r, f1, s = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "flow": flow,
            "model_family": family,
            "experiment": experiment,
            "super_class": labels,
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": s.astype(np.int64),
        }
    )


def main():
    t0 = time.time()

    section("F2.14 — super_class condicionada a flow")
    print("Escopo: train/validation do split principal.")
    print(
        f"Classes com < {MIN_TRAIN_SUPPORT_PER_CLASS} exemplos no treino "
        "dentro do flow são excluídas deste controle."
    )

    required = [
        CLASSIFICATION,
        BIO_PATH,
        TOPO_PATH,
        TARGET_PATH,
        BIO_SETS_PATH,
        TOPO_SETS_PATH,
        TOPO_MANIFEST_PATH,
        MAIN_SPLIT_PATH,
        LOG_SELECTED_PATH,
        RF_SELECTED_PATH,
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)

    bio_sets = load_json(BIO_SETS_PATH)
    topo_sets = load_json(TOPO_SETS_PATH)
    topo_manifest = pd.read_csv(TOPO_MANIFEST_PATH)

    cls = pd.read_csv(
        CLASSIFICATION,
        usecols=["root_id", "flow", "super_class"],
        low_memory=False,
    )
    target = pd.read_csv(TARGET_PATH)
    split = pd.read_csv(MAIN_SPLIT_PATH)
    bio = pd.read_csv(BIO_PATH, low_memory=False)
    topo = pd.read_csv(TOPO_PATH, low_memory=False)

    # frozen target must match classification exactly
    target_check = target.merge(
        cls[["root_id", "super_class"]],
        on="root_id",
        suffixes=("_frozen", "_classification"),
        validate="one_to_one",
    )
    if not (
        target_check["super_class_frozen"]
        == target_check["super_class_classification"]
    ).all():
        raise RuntimeError(
            "Frozen target diverge de classification.csv.gz."
        )

    data = (
        cls
        .merge(
            split[["root_id", "split"]],
            on="root_id",
            validate="one_to_one",
        )
        .merge(bio, on="root_id", validate="one_to_one")
        .merge(topo, on="root_id", validate="one_to_one")
    )

    if (data["split"] == "test").sum() == 0:
        raise RuntimeError("Split principal inesperado.")

    log_selected = pd.read_csv(
        LOG_SELECTED_PATH
    ).set_index("experiment")

    rf_selected = pd.read_csv(
        RF_SELECTED_PATH
    ).set_index("experiment")

    topo_feature_names = set(topo.columns) - {"root_id"}

    class_counts = (
        data[data["split"].isin(["train", "val"])]
        .groupby(
            ["flow", "super_class", "split"],
            observed=True,
        )
        .size()
        .rename("count")
        .reset_index()
    )
    class_counts.to_csv(CLASS_COUNTS_PATH, index=False)

    results = []
    per_class_parts = []
    exclusion_rows = []

    for flow in sorted(data["flow"].dropna().unique()):
        section(f"Flow = {flow}")

        train_flow = data[
            (data["flow"] == flow)
            & (data["split"] == "train")
        ].copy()

        val_flow = data[
            (data["flow"] == flow)
            & (data["split"] == "val")
        ].copy()

        train_counts = (
            train_flow["super_class"]
            .value_counts()
        )

        eligible = sorted(
            train_counts[
                train_counts >= MIN_TRAIN_SUPPORT_PER_CLASS
            ].index.tolist()
        )

        excluded = sorted(
            set(train_counts.index) - set(eligible)
        )

        for c in excluded:
            exclusion_rows.append(
                {
                    "flow": flow,
                    "super_class": c,
                    "train_support": int(train_counts[c]),
                    "reason": (
                        f"train_support < {MIN_TRAIN_SUPPORT_PER_CLASS}"
                    ),
                }
            )

        train_flow = train_flow[
            train_flow["super_class"].isin(eligible)
        ].copy()

        val_flow = val_flow[
            val_flow["super_class"].isin(eligible)
        ].copy()

        # ensure each eligible class is present in validation
        eligible = [
            c for c in eligible
            if c in set(val_flow["super_class"])
        ]
        train_flow = train_flow[
            train_flow["super_class"].isin(eligible)
        ].copy()
        val_flow = val_flow[
            val_flow["super_class"].isin(eligible)
        ].copy()

        if len(eligible) < 2:
            print("Menos de 2 classes elegíveis; pulando.")
            continue

        print(
            f"Classes elegíveis ({len(eligible)}): "
            + ", ".join(eligible)
        )
        print(
            f"Train={len(train_flow):,} | Val={len(val_flow):,}"
        )

        # majority baseline within flow
        majority = (
            train_flow["super_class"]
            .value_counts()
            .idxmax()
        )
        majority_pred = np.full(
            len(val_flow),
            majority,
            dtype=object,
        )
        m = evaluate(
            val_flow["super_class"].to_numpy(),
            majority_pred,
            eligible,
        )
        results.append(
            {
                "flow": flow,
                "model_family": "majority",
                "experiment": "majority_within_flow",
                "n_classes": len(eligible),
                "classes": ";".join(eligible),
                "train_rows": len(train_flow),
                "val_rows": len(val_flow),
                "hyperparameter": "",
                **m,
            }
        )

        for experiment in ALL_EXPERIMENTS:
            feature_cols = build_feature_spec(
                experiment,
                bio_sets,
                topo_sets,
            )

            X_train = train_flow[feature_cols]
            y_train = train_flow["super_class"].to_numpy()

            X_val = val_flow[feature_cols]
            y_val = val_flow["super_class"].to_numpy()

            # Logistic: frozen C from main validation
            if experiment not in log_selected.index:
                raise RuntimeError(
                    f"Sem config logistic congelada: {experiment}"
                )

            C = float(
                log_selected.loc[experiment, "C"]
            )

            log_pipe = Pipeline(
                steps=[
                    (
                        "preprocess",
                        make_logistic_preprocessor(
                            feature_cols,
                            topo_manifest,
                        ),
                    ),
                    (
                        "model",
                        LogisticRegression(
                            C=C,
                            solver="lbfgs",
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                log_pipe.fit(X_train, y_train)

            log_pred = log_pipe.predict(X_val)
            lm = evaluate(y_val, log_pred, eligible)

            results.append(
                {
                    "flow": flow,
                    "model_family": "logistic",
                    "experiment": experiment,
                    "n_classes": len(eligible),
                    "classes": ";".join(eligible),
                    "train_rows": len(train_flow),
                    "val_rows": len(val_flow),
                    "hyperparameter": f"C={C}",
                    **lm,
                }
            )

            per_class_parts.append(
                per_class_frame(
                    y_val,
                    log_pred,
                    eligible,
                    flow,
                    "logistic",
                    experiment,
                )
            )

            print(
                f"[LOG] {experiment:<42} "
                f"macro-F1={lm['macro_f1']:.4f}"
            )

            # Random Forest: frozen config from main validation
            if experiment not in rf_selected.index:
                raise RuntimeError(
                    f"Sem config RF congelada: {experiment}"
                )

            rf_row = rf_selected.loc[experiment]

            raw_max_features = str(rf_row["max_features"])
            max_features = (
                "sqrt"
                if raw_max_features == "sqrt"
                else float(raw_max_features)
            )

            min_leaf = int(rf_row["min_samples_leaf"])
            n_estimators = int(rf_row["n_estimators"])

            rf_pipe = Pipeline(
                steps=[
                    (
                        "preprocess",
                        make_rf_preprocessor(
                            feature_cols,
                            topo_feature_names,
                        ),
                    ),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=n_estimators,
                            criterion="gini",
                            max_depth=None,
                            min_samples_split=2,
                            min_samples_leaf=min_leaf,
                            max_features=max_features,
                            bootstrap=True,
                            class_weight="balanced_subsample",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )

            rf_pipe.fit(X_train, y_train)
            rf_pred = rf_pipe.predict(X_val)
            rm = evaluate(y_val, rf_pred, eligible)

            results.append(
                {
                    "flow": flow,
                    "model_family": "random_forest",
                    "experiment": experiment,
                    "n_classes": len(eligible),
                    "classes": ";".join(eligible),
                    "train_rows": len(train_flow),
                    "val_rows": len(val_flow),
                    "hyperparameter": (
                        f"max_features={max_features};"
                        f"min_samples_leaf={min_leaf};"
                        f"trees={n_estimators}"
                    ),
                    **rm,
                }
            )

            per_class_parts.append(
                per_class_frame(
                    y_val,
                    rf_pred,
                    eligible,
                    flow,
                    "random_forest",
                    experiment,
                )
            )

            print(
                f"[RF ] {experiment:<42} "
                f"macro-F1={rm['macro_f1']:.4f}"
            )

    results_df = pd.DataFrame(results)
    per_class_df = pd.concat(
        per_class_parts,
        ignore_index=True,
    )
    exclusions_df = pd.DataFrame(exclusion_rows)

    results_df.to_csv(RESULTS_PATH, index=False)
    per_class_df.to_csv(PER_CLASS_PATH, index=False)
    exclusions_df.to_csv(EXCLUSIONS_PATH, index=False)

    # Primary summary: how much network adds over BIO within each flow.
    primary = results_df[
        (
            results_df["model_family"]
            == "random_forest"
        )
        & (
            results_df["experiment"].isin(
                [
                    "bio_strict",
                    "network_full",
                    "bio_strict_plus_network_full",
                ]
            )
        )
    ].pivot(
        index="flow",
        columns="experiment",
        values="macro_f1",
    )

    if {
        "bio_strict",
        "network_full",
        "bio_strict_plus_network_full",
    }.issubset(primary.columns):
        primary["delta_network_over_bio"] = (
            primary["bio_strict_plus_network_full"]
            - primary["bio_strict"]
        )
        primary["delta_combined_over_network"] = (
            primary["bio_strict_plus_network_full"]
            - primary["network_full"]
        )

    summary = {
        "stage": "flow_conditioned_validation",
        "test_set_accessed": False,
        "minimum_train_support_per_class": MIN_TRAIN_SUPPORT_PER_CLASS,
        "primary_question": (
            "Does network topology still add predictive information for "
            "super_class after conditioning on coarse flow?"
        ),
        "results": results_df.to_dict(orient="records"),
        "excluded_low_support_flow_superclass_pairs": (
            exclusions_df.to_dict(orient="records")
        ),
        "primary_random_forest_summary": (
            primary.reset_index().to_dict(orient="records")
        ),
        "runtime_seconds": time.time() - t0,
    }

    with SUMMARY_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    lines = [
        "FlyWire FAFB v783 — F2.14 super_class condicionada a flow",
        "",
        "Escopo: train/validation do split principal.",
        "",
        "OBJETIVO",
        (
            "Avaliar se a topologia ainda distingue superclasses depois de "
            "fixar a organização grosseira afferent/intrinsic/efferent."
        ),
        "",
        "CRITÉRIO DE INCLUSÃO",
        (
            f">= {MIN_TRAIN_SUPPORT_PER_CLASS} exemplos no TRAIN dentro do flow."
        ),
        "",
        "RF — RESUMO PRINCIPAL",
    ]

    if len(primary):
        for flow, row in primary.iterrows():
            lines.append(
                f"{flow}: "
                f"BIO={row.get('bio_strict', float('nan')):.6f}, "
                f"REDE={row.get('network_full', float('nan')):.6f}, "
                f"BIO+REDE={row.get('bio_strict_plus_network_full', float('nan')):.6f}, "
                f"delta={row.get('delta_network_over_bio', float('nan')):+.6f}"
            )

    lines.extend(
        [
            "",
            f"Resultados: {RESULTS_PATH.relative_to(ROOT)}",
            f"Por classe: {PER_CLASS_PATH.relative_to(ROOT)}",
            f"Exclusões: {EXCLUSIONS_PATH.relative_to(ROOT)}",
            "",
            f"Tempo total: {(time.time() - t0) / 60:.2f} min",
        ]
    )

    SUMMARY_TXT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    section("Resumo principal — Random Forest")
    print(primary.to_string())

    print()
    print(f"Resultados: {RESULTS_PATH}")
    print(f"Por classe: {PER_CLASS_PATH}")
    print(f"Exclusões: {EXCLUSIONS_PATH}")
    print(f"Resumo: {SUMMARY_TXT_PATH}")


if __name__ == "__main__":
    main()
