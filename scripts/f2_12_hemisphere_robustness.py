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
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


# ============================================================
# Caminhos
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = ROOT / "reports" / "f2" / "features"
SPLIT_DIR = ROOT / "reports" / "f2" / "splits"
MODEL_DIR = ROOT / "reports" / "f2" / "models"

BIO_PATH = FEATURE_DIR / "node_biological_features.csv.gz"
TOPO_PATH = FEATURE_DIR / "node_topological_features.csv.gz"
TARGET_PATH = FEATURE_DIR / "node_targets.csv.gz"

BIO_SETS_PATH = FEATURE_DIR / "biological_feature_sets.json"
TOPO_SETS_PATH = FEATURE_DIR / "topological_feature_sets.json"
TOPO_MANIFEST_PATH = FEATURE_DIR / "topological_feature_manifest.csv"

LEFT_TO_RIGHT_PATH = SPLIT_DIR / "hemisphere_left_to_right.csv.gz"
RIGHT_TO_LEFT_PATH = SPLIT_DIR / "hemisphere_right_to_left.csv.gz"

LOG_SELECTED_PATH = (
    MODEL_DIR
    / "logistic_validation"
    / "selected_logistic_configs.csv"
)

RF_SELECTED_PATH = (
    MODEL_DIR
    / "random_forest_validation"
    / "selected_random_forest_configs.csv"
)

OUT_DIR = MODEL_DIR / "hemisphere_robustness"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUT_DIR / "hemisphere_results.csv"
PER_CLASS_PATH = OUT_DIR / "hemisphere_per_class_metrics.csv"
CONFUSION_PATH = OUT_DIR / "hemisphere_confusion_matrices.csv"
PREDICTIONS_PATH = OUT_DIR / "hemisphere_predictions.csv.gz"
SUMMARY_TXT_PATH = OUT_DIR / "hemisphere_robustness_summary.txt"
SUMMARY_JSON_PATH = OUT_DIR / "hemisphere_robustness_summary.json"


# ============================================================
# Configuração
# ============================================================

RANDOM_STATE = 42
NT_IMPUTATION_SEED = 314159

MORPHOLOGY_COLS = [
    "length_nm",
    "area_nm",
    "size_nm",
]

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

EXPERIMENTS = [
    "bio_strict",
    "bio_plus_availability",
    "network_full",
    "bio_strict_plus_network_full",
    "bio_plus_availability_plus_network_full",
]

DIRECTIONS = {
    "left_to_right": LEFT_TO_RIGHT_PATH,
    "right_to_left": RIGHT_TO_LEFT_PATH,
}


# ============================================================
# Transformadores
# ============================================================

def log1p_array(x):
    x = np.asarray(x, dtype=np.float64)

    if np.nanmin(x) < 0:
        raise ValueError(
            "log1p_array recebeu valor negativo."
        )

    return np.log1p(x)


class JointNTVectorImputer(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=NT_IMPUTATION_SEED):
        self.random_state = random_state

    def fit(self, X, y=None):
        x = np.asarray(X, dtype=np.float64)

        if x.ndim != 2 or x.shape[1] != 6:
            raise ValueError(
                "JointNTVectorImputer espera 6 colunas."
            )

        complete = np.isfinite(x).all(axis=1)
        partial = np.isfinite(x).any(axis=1) & ~complete

        if partial.any():
            raise ValueError(
                "Vetores NT parcialmente ausentes encontrados."
            )

        self.valid_vectors_ = x[complete].copy()

        if len(self.valid_vectors_) == 0:
            raise ValueError(
                "Nenhum vetor NT válido no treino."
            )

        self.n_features_in_ = 6
        return self

    def transform(self, X):
        x = np.asarray(X, dtype=np.float64).copy()

        complete = np.isfinite(x).all(axis=1)
        partial = np.isfinite(x).any(axis=1) & ~complete

        if partial.any():
            raise ValueError(
                "Vetores NT parcialmente ausentes encontrados."
            )

        missing_idx = np.flatnonzero(~complete)

        if len(missing_idx):
            seed = (
                int(self.random_state)
                + 104729 * len(x)
            ) % (2**32 - 1)

            rng = np.random.default_rng(seed)

            sampled_idx = rng.integers(
                0,
                len(self.valid_vectors_),
                size=len(missing_idx),
            )

            x[missing_idx] = self.valid_vectors_[
                sampled_idx
            ]

        if not np.isfinite(x).all():
            raise RuntimeError(
                "Imputação NT deixou NaN/infinito."
            )

        return x

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray(
                [f"x{i}" for i in range(self.n_features_in_)],
                dtype=object,
            )
        return np.asarray(input_features, dtype=object)


# ============================================================
# Auxiliares
# ============================================================

def section(title):
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_feature_spec(experiment, bio_sets, topo_sets):
    if experiment == "bio_strict":
        cols = list(bio_sets["bio_strict"])

    elif experiment == "bio_plus_availability":
        cols = list(
            bio_sets["bio_plus_availability"]
        )

    elif experiment == "network_full":
        cols = list(
            topo_sets["network_full"]
        )

    elif experiment == "bio_strict_plus_network_full":
        cols = (
            list(bio_sets["bio_strict"])
            + list(topo_sets["network_full"])
        )

    elif experiment == "bio_plus_availability_plus_network_full":
        cols = (
            list(
                bio_sets["bio_plus_availability"]
            )
            + list(
                topo_sets["network_full"]
            )
        )

    else:
        raise ValueError(
            f"Experimento desconhecido: {experiment}"
        )

    # Regra central da robustez hemisférica:
    # side define o domínio e NÃO pode entrar como preditor.
    cols = [
        c
        for c in cols
        if c != SIDE_COL
    ]

    return cols


def infer_topology_preprocessing(
    topo_manifest,
    feature_cols,
):
    relevant = topo_manifest[
        topo_manifest["feature"].isin(
            feature_cols
        )
    ]

    log_cols = relevant.loc[
        relevant[
            "recommended_preprocessing"
        ] == "log1p_then_scale",
        "feature",
    ].tolist()

    scale_cols = relevant.loc[
        relevant[
            "recommended_preprocessing"
        ] == "scale",
        "feature",
    ].tolist()

    return log_cols, scale_cols


def make_logistic_preprocessor(
    feature_cols,
    topo_manifest,
):
    transformers = []

    morph_cols = [
        c
        for c in MORPHOLOGY_COLS
        if c in feature_cols
    ]

    if morph_cols:
        transformers.append(
            (
                "morphology",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "log1p",
                            FunctionTransformer(
                                log1p_array,
                                validate=False,
                            ),
                        ),
                        (
                            "scale",
                            StandardScaler(),
                        ),
                    ]
                ),
                morph_cols,
            )
        )

    nt_cols = [
        c
        for c in NT_PROB_COLS
        if c in feature_cols
    ]

    if nt_cols:
        if nt_cols != NT_PROB_COLS:
            raise RuntimeError(
                "Bloco NT incompleto."
            )

        transformers.append(
            (
                "nt_probabilities",
                Pipeline(
                    steps=[
                        (
                            "joint_imputer",
                            JointNTVectorImputer(),
                        ),
                        (
                            "scale",
                            StandardScaler(),
                        ),
                    ]
                ),
                nt_cols,
            )
        )

    if NT_AVAILABLE_COL in feature_cols:
        transformers.append(
            (
                "nt_available",
                "passthrough",
                [NT_AVAILABLE_COL],
            )
        )

    topo_features = set(
        topo_manifest["feature"]
    )

    topo_cols = [
        c
        for c in feature_cols
        if c in topo_features
    ]

    if topo_cols:
        log_cols, scale_cols = (
            infer_topology_preprocessing(
                topo_manifest,
                topo_cols,
            )
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
                            (
                                "scale",
                                StandardScaler(),
                            ),
                        ]
                    ),
                    log_cols,
                )
            )

        if scale_cols:
            transformers.append(
                (
                    "topology_scale",
                    StandardScaler(),
                    scale_cols,
                )
            )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_rf_preprocessor(
    feature_cols,
    topo_feature_names,
):
    transformers = []

    morph_cols = [
        c
        for c in MORPHOLOGY_COLS
        if c in feature_cols
    ]

    if morph_cols:
        transformers.append(
            (
                "morphology",
                SimpleImputer(
                    strategy="median"
                ),
                morph_cols,
            )
        )

    nt_cols = [
        c
        for c in NT_PROB_COLS
        if c in feature_cols
    ]

    if nt_cols:
        if nt_cols != NT_PROB_COLS:
            raise RuntimeError(
                "Bloco NT incompleto."
            )

        transformers.append(
            (
                "nt_probabilities",
                JointNTVectorImputer(),
                nt_cols,
            )
        )

    if NT_AVAILABLE_COL in feature_cols:
        transformers.append(
            (
                "nt_available",
                "passthrough",
                [NT_AVAILABLE_COL],
            )
        )

    topo_cols = [
        c
        for c in feature_cols
        if c in topo_feature_names
    ]

    if topo_cols:
        transformers.append(
            (
                "topology",
                "passthrough",
                topo_cols,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def metrics(y_true, y_pred, labels):
    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
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


def per_class_frame(
    y_true,
    y_pred,
    labels,
    direction,
    family,
    experiment,
):
    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
        )
    )

    return pd.DataFrame(
        {
            "direction": direction,
            "model_family": family,
            "experiment": experiment,
            "super_class": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(
                np.int64
            ),
        }
    )


def confusion_frame(
    y_true,
    y_pred,
    labels,
    direction,
    family,
    experiment,
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    rows = []

    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            rows.append(
                {
                    "direction": direction,
                    "model_family": family,
                    "experiment": experiment,
                    "true_class": true_label,
                    "pred_class": pred_label,
                    "count": int(
                        cm[i, j]
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():
    start = time.time()

    section(
        "F2.12 — Robustez hemisférica"
    )

    print(
        "Configuração: hiperparâmetros selecionados na validação principal; "
        "side excluída das features."
    )

    required = [
        BIO_PATH,
        TOPO_PATH,
        TARGET_PATH,
        BIO_SETS_PATH,
        TOPO_SETS_PATH,
        TOPO_MANIFEST_PATH,
        LEFT_TO_RIGHT_PATH,
        RIGHT_TO_LEFT_PATH,
        LOG_SELECTED_PATH,
        RF_SELECTED_PATH,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo ausente: {path}"
            )

    # --------------------------------------------------------
    # 1. Dados e configs
    # --------------------------------------------------------

    section("1. Carregando dados e configurações congeladas")

    bio_sets = load_json(BIO_SETS_PATH)
    topo_sets = load_json(TOPO_SETS_PATH)

    topo_manifest = pd.read_csv(
        TOPO_MANIFEST_PATH
    )

    bio = pd.read_csv(
        BIO_PATH,
        low_memory=False,
    )

    topo = pd.read_csv(
        TOPO_PATH,
        low_memory=False,
    )

    target = pd.read_csv(
        TARGET_PATH,
        usecols=[
            "root_id",
            "super_class",
        ],
    )

    log_selected = pd.read_csv(
        LOG_SELECTED_PATH
    ).set_index(
        "experiment"
    )

    rf_selected = pd.read_csv(
        RF_SELECTED_PATH
    ).set_index(
        "experiment"
    )

    data = (
        target
        .merge(
            bio,
            on="root_id",
            validate="one_to_one",
        )
        .merge(
            topo,
            on="root_id",
            validate="one_to_one",
        )
    )

    labels = sorted(
        data["super_class"].unique()
    )

    if len(labels) != 10:
        raise RuntimeError(
            "Esperadas 10 classes."
        )

    topo_feature_names = set(
        topo.columns
    ) - {"root_id"}

    # --------------------------------------------------------
    # 2. Experimentos
    # --------------------------------------------------------

    results = []
    per_class_parts = []
    confusion_parts = []
    prediction_parts = []

    for direction, split_path in DIRECTIONS.items():
        section(
            f"2. Direção: {direction}"
        )

        split = pd.read_csv(
            split_path,
            usecols=[
                "root_id",
                "side",
                "split",
            ],
        )

        hemi = data.merge(
            split,
            on="root_id",
            validate="one_to_one",
        )

        source = hemi[
            hemi["split"].isin(
                ["train", "val"]
            )
        ].copy()

        target_hemi = hemi[
            hemi["split"] == "test"
        ].copy()

        excluded = hemi[
            hemi["split"] == "excluded"
        ]

        print(
            f"Source train+val: {len(source):,}"
        )
        print(
            f"Target hemisphere: {len(target_hemi):,}"
        )
        print(
            f"Excluded center/unknown: {len(excluded):,}"
        )

        if (
            source["super_class"].nunique()
            != 10
            or target_hemi[
                "super_class"
            ].nunique()
            != 10
        ):
            raise RuntimeError(
                f"{direction}: faltam classes."
            )

        for experiment in EXPERIMENTS:
            feature_cols = build_feature_spec(
                experiment,
                bio_sets,
                topo_sets,
            )

            if SIDE_COL in feature_cols:
                raise RuntimeError(
                    "side entrou indevidamente nas features."
                )

            X_source = source[
                feature_cols
            ]
            y_source = source[
                "super_class"
            ].to_numpy()

            X_target = target_hemi[
                feature_cols
            ]
            y_target = target_hemi[
                "super_class"
            ].to_numpy()

            # ---------------- Logistic ----------------
            if experiment not in log_selected.index:
                raise RuntimeError(
                    f"Sem config Logistic para {experiment}."
                )

            C = float(
                log_selected.loc[
                    experiment,
                    "C",
                ]
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

            fit_start = time.time()

            with warnings.catch_warnings(
                record=True
            ) as caught:
                warnings.simplefilter(
                    "always",
                    ConvergenceWarning,
                )

                log_pipe.fit(
                    X_source,
                    y_source,
                )

            log_fit = time.time() - fit_start

            log_pred = log_pipe.predict(
                X_target
            )

            log_metrics = metrics(
                y_target,
                log_pred,
                labels,
            )

            results.append(
                {
                    "direction": direction,
                    "model_family": "logistic",
                    "experiment": experiment,
                    "hyperparameter": f"C={C}",
                    "source_rows": len(source),
                    "target_rows": len(
                        target_hemi
                    ),
                    "fit_seconds": log_fit,
                    **log_metrics,
                }
            )

            per_class_parts.append(
                per_class_frame(
                    y_target,
                    log_pred,
                    labels,
                    direction,
                    "logistic",
                    experiment,
                )
            )

            confusion_parts.append(
                confusion_frame(
                    y_target,
                    log_pred,
                    labels,
                    direction,
                    "logistic",
                    experiment,
                )
            )

            prediction_parts.append(
                pd.DataFrame(
                    {
                        "root_id": target_hemi[
                            "root_id"
                        ].to_numpy(),
                        "direction": direction,
                        "model_family": "logistic",
                        "experiment": experiment,
                        "y_true": y_target,
                        "y_pred": log_pred,
                    }
                )
            )

            print(
                f"[LOG] {experiment:<42} "
                f"macro-F1={log_metrics['macro_f1']:.4f}"
            )

            # ---------------- RF ----------------
            if experiment not in rf_selected.index:
                raise RuntimeError(
                    f"Sem config RF para {experiment}."
                )

            rf_row = rf_selected.loc[
                experiment
            ]

            raw_max_features = str(
                rf_row["max_features"]
            )

            if raw_max_features == "sqrt":
                max_features = "sqrt"
            else:
                max_features = float(
                    raw_max_features
                )

            min_leaf = int(
                rf_row[
                    "min_samples_leaf"
                ]
            )

            n_estimators = int(
                rf_row[
                    "n_estimators"
                ]
            )

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

            fit_start = time.time()

            rf_pipe.fit(
                X_source,
                y_source,
            )

            rf_fit = time.time() - fit_start

            rf_pred = rf_pipe.predict(
                X_target
            )

            rf_metrics = metrics(
                y_target,
                rf_pred,
                labels,
            )

            results.append(
                {
                    "direction": direction,
                    "model_family": "random_forest",
                    "experiment": experiment,
                    "hyperparameter": (
                        f"max_features={max_features};"
                        f"min_samples_leaf={min_leaf};"
                        f"trees={n_estimators}"
                    ),
                    "source_rows": len(source),
                    "target_rows": len(
                        target_hemi
                    ),
                    "fit_seconds": rf_fit,
                    **rf_metrics,
                }
            )

            per_class_parts.append(
                per_class_frame(
                    y_target,
                    rf_pred,
                    labels,
                    direction,
                    "random_forest",
                    experiment,
                )
            )

            confusion_parts.append(
                confusion_frame(
                    y_target,
                    rf_pred,
                    labels,
                    direction,
                    "random_forest",
                    experiment,
                )
            )

            prediction_parts.append(
                pd.DataFrame(
                    {
                        "root_id": target_hemi[
                            "root_id"
                        ].to_numpy(),
                        "direction": direction,
                        "model_family": "random_forest",
                        "experiment": experiment,
                        "y_true": y_target,
                        "y_pred": rf_pred,
                    }
                )
            )

            print(
                f"[RF ] {experiment:<42} "
                f"macro-F1={rf_metrics['macro_f1']:.4f}"
            )

    # --------------------------------------------------------
    # 3. Consolida
    # --------------------------------------------------------

    section("3. Consolidando robustez")

    results_df = pd.DataFrame(results)

    per_class_df = pd.concat(
        per_class_parts,
        ignore_index=True,
    )

    confusion_df = pd.concat(
        confusion_parts,
        ignore_index=True,
    )

    predictions_df = pd.concat(
        prediction_parts,
        ignore_index=True,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    per_class_df.to_csv(
        PER_CLASS_PATH,
        index=False,
    )

    confusion_df.to_csv(
        CONFUSION_PATH,
        index=False,
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
        compression="gzip",
    )

    # Médias bidirecionais.
    bidirectional = (
        results_df.groupby(
            [
                "model_family",
                "experiment",
            ],
            observed=True,
        )[
            [
                "macro_f1",
                "balanced_accuracy",
                "accuracy",
                "weighted_f1",
            ]
        ]
        .mean()
        .reset_index()
    )

    # --------------------------------------------------------
    # 4. Resumo
    # --------------------------------------------------------

    summary = {
        "stage": "hemisphere_transfer_robustness",
        "main_random_test_accessed": False,
        "side_used_as_predictor": False,
        "training_rule": (
            "for each direction, concatenate source-side frozen train+val; "
            "fit with hyperparameters selected on the main validation experiment; "
            "evaluate once on the opposite hemisphere"
        ),
        "experiments": EXPERIMENTS,
        "directions": list(
            DIRECTIONS.keys()
        ),
        "results": results_df.to_dict(
            orient="records"
        ),
        "bidirectional_mean": (
            bidirectional.to_dict(
                orient="records"
            )
        ),
        "runtime_seconds": (
            time.time() - start
        ),
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
        "FlyWire FAFB v783 — F2.12 Robustez hemisférica",
        "",
        "DESENHO",
        "- hiperparâmetros selecionados na validação principal;",
        "- side removida de todas as features;",
        "- source = train + val do hemisfério de origem;",
        "- target = hemisfério oposto;",
        "- center/unknown excluídos desta robustez;",
        "- avaliação realizada exclusivamente no desenho hemisférico.",
        "",
        "RESULTADOS POR DIREÇÃO",
    ]

    for _, row in results_df.iterrows():
        lines.append(
            f"{row['direction']} | "
            f"{row['model_family']} | "
            f"{row['experiment']} | "
            f"macro-F1={row['macro_f1']:.6f} | "
            f"balanced_acc={row['balanced_accuracy']:.6f} | "
            f"accuracy={row['accuracy']:.6f}"
        )

    lines.extend(
        [
            "",
            "MÉDIA BIDIRECIONAL",
        ]
    )

    for _, row in bidirectional.iterrows():
        lines.append(
            f"{row['model_family']} | "
            f"{row['experiment']} | "
            f"macro-F1={row['macro_f1']:.6f} | "
            f"balanced_acc={row['balanced_accuracy']:.6f} | "
            f"accuracy={row['accuracy']:.6f}"
        )

    lines.append("")
    lines.append(
        f"Tempo total: {(time.time() - start) / 60:.2f} min"
    )

    SUMMARY_TXT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("Média bidirecional:")
    print(
        bidirectional[
            [
                "model_family",
                "experiment",
                "macro_f1",
                "balanced_accuracy",
                "accuracy",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(f"Resultados:   {RESULTS_PATH}")
    print(f"Por classe:   {PER_CLASS_PATH}")
    print(f"Confusões:    {CONFUSION_PATH}")
    print(f"Predições:    {PREDICTIONS_PATH}")
    print(f"Resumo:       {SUMMARY_TXT_PATH}")

    section("Concluído")
    print(
        f"Tempo total: {(time.time() - start) / 60:.2f} min"
    )


if __name__ == "__main__":
    main()
