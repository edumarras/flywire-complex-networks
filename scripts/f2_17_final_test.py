"""
F2.17 — Avaliação final no conjunto de teste

Avaliação confirmatória no split principal.
- TRAIN + VAL são usados para o ajuste final.
- O conjunto de teste é usado somente para avaliação.
- O preprocessing é ajustado em TRAIN+VAL.
- Cinco experimentos pré-declarados:
  1) bio_strict
  2) bio_plus_availability
  3) network_full
  4) bio_strict_plus_network_full              [PRINCIPAL]
  5) bio_plus_availability_plus_network_full   [SENSIBILIDADE]
"""

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
    confusion_matrix,
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

FEATURE_DIR = ROOT / "reports" / "f2" / "features"
SPLIT_DIR = ROOT / "reports" / "f2" / "splits"
OUT_DIR = ROOT / "reports" / "f2" / "final_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BIO_PATH = FEATURE_DIR / "node_biological_features.csv.gz"
TOPO_PATH = FEATURE_DIR / "node_topological_features.csv.gz"
TARGET_PATH = FEATURE_DIR / "node_targets.csv.gz"
BIO_SETS_PATH = FEATURE_DIR / "biological_feature_sets.json"
TOPO_SETS_PATH = FEATURE_DIR / "topological_feature_sets.json"
TOPO_MANIFEST_PATH = FEATURE_DIR / "topological_feature_manifest.csv"
MAIN_SPLIT_PATH = SPLIT_DIR / "main_split.csv.gz"

RESULTS_PATH = OUT_DIR / "final_test_results.csv"
PER_CLASS_PATH = OUT_DIR / "final_test_per_class_metrics.csv"
PREDICTIONS_PATH = OUT_DIR / "final_test_predictions.csv.gz"
CONFUSION_PATH = OUT_DIR / "final_test_confusion_matrices.csv"
SUMMARY_TXT_PATH = OUT_DIR / "final_test_summary.txt"
SUMMARY_JSON_PATH = OUT_DIR / "final_test_summary.json"

RANDOM_STATE = 42
NT_IMPUTATION_SEED = 314159

MORPHOLOGY_COLS = ["length_nm", "area_nm", "size_nm"]
NT_PROB_COLS = ["da_avg", "ser_avg", "gaba_avg", "glut_avg", "ach_avg", "oct_avg"]
SIDE_COL = "side"
NT_AVAILABLE_COL = "nt_available"

EXPERIMENTS = [
    "bio_strict",
    "bio_plus_availability",
    "network_full",
    "bio_strict_plus_network_full",
    "bio_plus_availability_plus_network_full",
]

PRIMARY_EXPERIMENT = "bio_strict_plus_network_full"
SENSITIVITY_EXPERIMENT = "bio_plus_availability_plus_network_full"

# Congelados antes do TEST.
LOGISTIC_CONFIG = {
    "bio_strict": 10.0,
    "bio_plus_availability": 10.0,
    "network_full": 0.1,
    "bio_strict_plus_network_full": 10.0,
    "bio_plus_availability_plus_network_full": 10.0,
}

RF_CONFIG = {
    "bio_strict": {"max_features": 0.5, "min_samples_leaf": 5, "n_estimators": 300},
    "bio_plus_availability": {"max_features": 0.5, "min_samples_leaf": 5, "n_estimators": 300},
    "network_full": {"max_features": 0.5, "min_samples_leaf": 1, "n_estimators": 300},
    "bio_strict_plus_network_full": {"max_features": "sqrt", "min_samples_leaf": 5, "n_estimators": 300},
    "bio_plus_availability_plus_network_full": {"max_features": 0.5, "min_samples_leaf": 5, "n_estimators": 300},
}


def section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def require(path):
    if not path.exists():
        raise FileNotFoundError(path)


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def log1p_array(x):
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if len(finite) and finite.min() < 0:
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
            raise ValueError("Nenhum vetor NT válido no conjunto de ajuste.")
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
            seed = (int(self.random_state) + 104729 * len(x)) % (2**32 - 1)
            rng = np.random.default_rng(seed)
            sampled = rng.integers(0, len(self.valid_vectors_), size=len(missing_idx))
            x[missing_idx] = self.valid_vectors_[sampled]

        if not np.isfinite(x).all():
            raise RuntimeError("Imputação NT deixou NaN/infinito.")
        return x

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return np.asarray(input_features, dtype=object)


def build_feature_spec(experiment, bio_sets, topo_sets):
    if experiment == "bio_strict":
        return list(bio_sets["bio_strict"])
    if experiment == "bio_plus_availability":
        return list(bio_sets["bio_plus_availability"])
    if experiment == "network_full":
        return list(topo_sets["network_full"])
    if experiment == "bio_strict_plus_network_full":
        return list(bio_sets["bio_strict"]) + list(topo_sets["network_full"])
    if experiment == "bio_plus_availability_plus_network_full":
        return list(bio_sets["bio_plus_availability"]) + list(topo_sets["network_full"])
    raise ValueError(experiment)


def infer_topology_preprocessing(topo_manifest, cols):
    relevant = topo_manifest[topo_manifest["feature"].isin(cols)]

    log_cols = relevant.loc[
        relevant["recommended_preprocessing"] == "log1p_then_scale",
        "feature",
    ].tolist()

    scale_cols = relevant.loc[
        relevant["recommended_preprocessing"] == "scale",
        "feature",
    ].tolist()

    known = set(log_cols) | set(scale_cols)
    unknown = sorted(set(cols) - known)
    if unknown:
        raise RuntimeError(
            "Features topológicas sem preprocessing definido: " + ", ".join(unknown)
        )

    return log_cols, scale_cols


def make_logistic_preprocessor(feature_cols, topo_manifest):
    transformers = []

    if SIDE_COL in feature_cols:
        transformers.append(
            (
                "side",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
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
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("log1p", FunctionTransformer(log1p_array, validate=False)),
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
                    [
                        ("imputer", JointNTVectorImputer()),
                        ("scale", StandardScaler()),
                    ]
                ),
                nt,
            )
        )

    if NT_AVAILABLE_COL in feature_cols:
        transformers.append(("nt_available", "passthrough", [NT_AVAILABLE_COL]))

    topo_names = set(topo_manifest["feature"])
    topo_cols = [c for c in feature_cols if c in topo_names]

    if topo_cols:
        log_cols, scale_cols = infer_topology_preprocessing(topo_manifest, topo_cols)

        if log_cols:
            transformers.append(
                (
                    "topology_log",
                    Pipeline(
                        [
                            ("log1p", FunctionTransformer(log1p_array, validate=False)),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    log_cols,
                )
            )

        if scale_cols:
            transformers.append(("topology_scale", StandardScaler(), scale_cols))

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
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                [SIDE_COL],
            )
        )

    morph = [c for c in MORPHOLOGY_COLS if c in feature_cols]
    if morph:
        transformers.append(("morphology", SimpleImputer(strategy="median"), morph))

    nt = [c for c in NT_PROB_COLS if c in feature_cols]
    if nt:
        transformers.append(("nt", JointNTVectorImputer(), nt))

    if NT_AVAILABLE_COL in feature_cols:
        transformers.append(("nt_available", "passthrough", [NT_AVAILABLE_COL]))

    topo_cols = [c for c in feature_cols if c in topo_feature_names]
    if topo_cols:
        transformers.append(("topology", "passthrough", topo_cols))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def evaluate(y_true, y_pred, labels):
    return {
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
    }


def per_class_frame(y_true, y_pred, labels, family, experiment):
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return pd.DataFrame(
        {
            "model_family": family,
            "experiment": experiment,
            "super_class": labels,
            "precision": p,
            "recall": r,
            "f1": f1,
            "support": s.astype(np.int64),
        }
    )


def confusion_long(y_true, y_pred, labels, family, experiment):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            rows.append(
                {
                    "model_family": family,
                    "experiment": experiment,
                    "true_class": true_label,
                    "predicted_class": pred_label,
                    "count": int(cm[i, j]),
                }
            )
    return pd.DataFrame(rows)


def main():
    t0 = time.time()

    section("F2.17 — Avaliação final no conjunto de teste")

    for p in [
        BIO_PATH,
        TOPO_PATH,
        TARGET_PATH,
        BIO_SETS_PATH,
        TOPO_SETS_PATH,
        TOPO_MANIFEST_PATH,
        MAIN_SPLIT_PATH,
    ]:
        require(p)

    bio_sets = load_json(BIO_SETS_PATH)
    topo_sets = load_json(TOPO_SETS_PATH)
    topo_manifest = pd.read_csv(TOPO_MANIFEST_PATH)

    target = pd.read_csv(TARGET_PATH, low_memory=False)
    split = pd.read_csv(MAIN_SPLIT_PATH, low_memory=False)
    bio = pd.read_csv(BIO_PATH, low_memory=False)
    topo = pd.read_csv(TOPO_PATH, low_memory=False)

    data = (
        target
        .merge(split[["root_id", "split"]], on="root_id", validate="one_to_one")
        .merge(bio, on="root_id", validate="one_to_one")
        .merge(topo, on="root_id", validate="one_to_one")
    )

    counts = data["split"].value_counts().to_dict()
    expected = {"train": 97478, "val": 20890, "test": 20887}

    print(f"Split encontrado: {counts}")

    for name, n in expected.items():
        if counts.get(name, 0) != n:
            raise RuntimeError(
                f"Split inesperado para {name}: {counts.get(name, 0)} != {n}"
            )

    fit_df = data[data["split"].isin(["train", "val"])].copy()
    test_df = data[data["split"] == "test"].copy()

    if set(fit_df["root_id"]) & set(test_df["root_id"]):
        raise RuntimeError("Sobreposição de root_id entre TRAIN+VAL e TEST.")

    print(f"TRAIN+VAL final: {len(fit_df):,}")
    print(f"TEST final:      {len(test_df):,}")

    labels = sorted(fit_df["super_class"].unique().tolist())
    test_labels = sorted(test_df["super_class"].unique().tolist())

    if labels != test_labels:
        raise RuntimeError("Classes de TRAIN+VAL e TEST não coincidem.")

    print(f"Classes ({len(labels)}): " + ", ".join(labels))

    topo_feature_names = set(topo.columns) - {"root_id"}

    result_rows = []
    per_class_parts = []
    prediction_parts = []
    confusion_parts = []

    for experiment in EXPERIMENTS:
        section(f"EXPERIMENTO: {experiment}")

        feature_cols = build_feature_spec(experiment, bio_sets, topo_sets)

        missing = [c for c in feature_cols if c not in data.columns]
        if missing:
            raise RuntimeError(
                f"{experiment}: features ausentes: " + ", ".join(missing)
            )

        print(f"Features brutas: {len(feature_cols)}")

        X_fit = fit_df[feature_cols]
        y_fit = fit_df["super_class"].to_numpy()

        X_test = test_df[feature_cols]
        y_test = test_df["super_class"].to_numpy()

        # LOGISTIC
        C = LOGISTIC_CONFIG[experiment]

        log_pipe = Pipeline(
            [
                ("preprocess", make_logistic_preprocessor(feature_cols, topo_manifest)),
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

        t_fit = time.time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            log_pipe.fit(X_fit, y_fit)

        log_fit_seconds = time.time() - t_fit
        log_pred = log_pipe.predict(X_test)
        log_metrics = evaluate(y_test, log_pred, labels)

        conv = [
            str(w.message)
            for w in caught
            if issubclass(w.category, ConvergenceWarning)
        ]

        try:
            n_iter = int(np.max(log_pipe.named_steps["model"].n_iter_))
        except Exception:
            n_iter = None

        result_rows.append(
            {
                "model_family": "logistic",
                "experiment": experiment,
                "is_primary": experiment == PRIMARY_EXPERIMENT,
                "is_sensitivity": experiment == SENSITIVITY_EXPERIMENT,
                "train_val_rows": len(fit_df),
                "test_rows": len(test_df),
                "n_features_raw": len(feature_cols),
                "hyperparameters": json.dumps({"C": C}),
                "fit_seconds": log_fit_seconds,
                "n_iter": n_iter,
                "convergence_warning": bool(conv),
                **log_metrics,
            }
        )

        per_class_parts.append(
            per_class_frame(y_test, log_pred, labels, "logistic", experiment)
        )
        confusion_parts.append(
            confusion_long(y_test, log_pred, labels, "logistic", experiment)
        )
        prediction_parts.append(
            pd.DataFrame(
                {
                    "root_id": test_df["root_id"].to_numpy(),
                    "true_super_class": y_test,
                    "predicted_super_class": log_pred,
                    "model_family": "logistic",
                    "experiment": experiment,
                }
            )
        )

        print(
            f"[LOG] macro-F1={log_metrics['macro_f1']:.6f} | "
            f"balacc={log_metrics['balanced_accuracy']:.6f} | "
            f"acc={log_metrics['accuracy']:.6f}"
        )

        # RANDOM FOREST
        cfg = RF_CONFIG[experiment]

        rf_pipe = Pipeline(
            [
                ("preprocess", make_rf_preprocessor(feature_cols, topo_feature_names)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=cfg["n_estimators"],
                        criterion="gini",
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=cfg["min_samples_leaf"],
                        max_features=cfg["max_features"],
                        bootstrap=True,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

        t_fit = time.time()
        rf_pipe.fit(X_fit, y_fit)
        rf_fit_seconds = time.time() - t_fit

        rf_pred = rf_pipe.predict(X_test)
        rf_metrics = evaluate(y_test, rf_pred, labels)

        result_rows.append(
            {
                "model_family": "random_forest",
                "experiment": experiment,
                "is_primary": experiment == PRIMARY_EXPERIMENT,
                "is_sensitivity": experiment == SENSITIVITY_EXPERIMENT,
                "train_val_rows": len(fit_df),
                "test_rows": len(test_df),
                "n_features_raw": len(feature_cols),
                "hyperparameters": json.dumps(cfg),
                "fit_seconds": rf_fit_seconds,
                "n_iter": np.nan,
                "convergence_warning": False,
                **rf_metrics,
            }
        )

        per_class_parts.append(
            per_class_frame(y_test, rf_pred, labels, "random_forest", experiment)
        )
        confusion_parts.append(
            confusion_long(y_test, rf_pred, labels, "random_forest", experiment)
        )
        prediction_parts.append(
            pd.DataFrame(
                {
                    "root_id": test_df["root_id"].to_numpy(),
                    "true_super_class": y_test,
                    "predicted_super_class": rf_pred,
                    "model_family": "random_forest",
                    "experiment": experiment,
                }
            )
        )

        print(
            f"[RF ] macro-F1={rf_metrics['macro_f1']:.6f} | "
            f"balacc={rf_metrics['balanced_accuracy']:.6f} | "
            f"acc={rf_metrics['accuracy']:.6f}"
        )

    results = pd.DataFrame(result_rows)
    per_class = pd.concat(per_class_parts, ignore_index=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    confusion = pd.concat(confusion_parts, ignore_index=True)

    results.to_csv(RESULTS_PATH, index=False)
    per_class.to_csv(PER_CLASS_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False, compression="gzip")
    confusion.to_csv(CONFUSION_PATH, index=False)

    section("RESULTADO FINAL")

    pivot = results.pivot(
        index="model_family",
        columns="experiment",
        values="macro_f1",
    )

    print(pivot.to_string(float_format=lambda x: f"{x:.6f}"))

    derived = {}

    for family in ["logistic", "random_forest"]:
        row = pivot.loc[family]
        derived[family] = {
            "bio_strict_macro_f1": float(row["bio_strict"]),
            "network_full_macro_f1": float(row["network_full"]),
            "combined_primary_macro_f1": float(
                row["bio_strict_plus_network_full"]
            ),
            "combined_sensitivity_macro_f1": float(
                row["bio_plus_availability_plus_network_full"]
            ),
            "network_gain_over_bio": float(
                row["bio_strict_plus_network_full"] - row["bio_strict"]
            ),
            "availability_gain_in_combined": float(
                row["bio_plus_availability_plus_network_full"]
                - row["bio_strict_plus_network_full"]
            ),
        }

    print()
    for family, d in derived.items():
        print(
            f"{family}: "
            f"BIO={d['bio_strict_macro_f1']:.6f} | "
            f"REDE={d['network_full_macro_f1']:.6f} | "
            f"BIO+REDE={d['combined_primary_macro_f1']:.6f} | "
            f"ganho_REDE={d['network_gain_over_bio']:+.6f}"
        )

    runtime = time.time() - t0

    summary = {
        "stage": "final_test_confirmatory",
        "test_opened": True,
        "test_used_for_tuning": False,
        "train_val_rows": int(len(fit_df)),
        "test_rows": int(len(test_df)),
        "classes": labels,
        "primary_experiment": PRIMARY_EXPERIMENT,
        "sensitivity_experiment": SENSITIVITY_EXPERIMENT,
        "logistic_config": LOGISTIC_CONFIG,
        "random_forest_config": RF_CONFIG,
        "results": results.to_dict(orient="records"),
        "derived_comparisons": derived,
        "runtime_seconds": runtime,
        "methodological_note": (
            "All preprocessing was refit using TRAIN+VAL only; "
            "the TEST split was used for confirmatory evaluation only."
        ),
    }

    with SUMMARY_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    lines = [
        "F2.17 — Avaliação final no conjunto de teste",
        "",
        "Avaliação confirmatória no split principal.",
        "",
        f"TRAIN+VAL: {len(fit_df)}",
        f"TEST: {len(test_df)}",
        "",
        "MACRO-F1 FINAL",
    ]

    for family in ["logistic", "random_forest"]:
        d = derived[family]
        lines.extend(
            [
                "",
                family.upper(),
                f"BIO-strict: {d['bio_strict_macro_f1']:.6f}",
                f"REDE completa: {d['network_full_macro_f1']:.6f}",
                f"BIO-strict + REDE: {d['combined_primary_macro_f1']:.6f}",
                f"Ganho REDE sobre BIO: {d['network_gain_over_bio']:+.6f}",
                (
                    "BIO+availability+REDE: "
                    f"{d['combined_sensitivity_macro_f1']:.6f}"
                ),
                (
                    "Ganho availability no combinado: "
                    f"{d['availability_gain_in_combined']:+.6f}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "ARQUIVOS",
            f"Resultados: {RESULTS_PATH.relative_to(ROOT)}",
            f"Por classe: {PER_CLASS_PATH.relative_to(ROOT)}",
            f"Predições: {PREDICTIONS_PATH.relative_to(ROOT)}",
            f"Confusion matrices: {CONFUSION_PATH.relative_to(ROOT)}",
            f"JSON: {SUMMARY_JSON_PATH.relative_to(ROOT)}",
            "",
            f"Tempo total: {runtime/60:.2f} min",
        ]
    )

    SUMMARY_TXT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print()
    print(f"Resultados: {RESULTS_PATH}")
    print(f"Por classe: {PER_CLASS_PATH}")
    print(f"Predições:  {PREDICTIONS_PATH}")
    print(f"Confusão:   {CONFUSION_PATH}")
    print(f"Resumo:     {SUMMARY_TXT_PATH}")
    print()
    print("Avaliação final concluída.")


if __name__ == "__main__":
    main()
