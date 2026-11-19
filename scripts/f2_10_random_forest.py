from pathlib import Path
import argparse
import json
import os
import time

# Evita oversubscription da pilha numérica; o paralelismo principal do RF
# é controlado por n_jobs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

try:
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
except ImportError as exc:
    raise SystemExit(
        "Este script requer scikit-learn.\n"
        "Instale com:\n"
        "  conda install -c conda-forge scikit-learn"
    ) from exc


# ============================================================
# Caminhos
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = ROOT / "reports" / "f2" / "features"
SPLIT_DIR = ROOT / "reports" / "f2" / "splits"

BIO_PATH = FEATURE_DIR / "node_biological_features.csv.gz"
TOPO_PATH = FEATURE_DIR / "node_topological_features.csv.gz"
TARGET_PATH = FEATURE_DIR / "node_targets.csv.gz"

BIO_SETS_PATH = FEATURE_DIR / "biological_feature_sets.json"
TOPO_SETS_PATH = FEATURE_DIR / "topological_feature_sets.json"
MAIN_SPLIT_PATH = SPLIT_DIR / "main_split.csv.gz"

LOGISTIC_SELECTED_PATH = (
    ROOT
    / "reports"
    / "f2"
    / "models"
    / "logistic_validation"
    / "selected_logistic_configs.csv"
)

MODEL_DIR = (
    ROOT
    / "reports"
    / "f2"
    / "models"
    / "random_forest_validation"
)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_ALL_PATH = MODEL_DIR / "validation_results_all.csv"
SELECTED_PATH = MODEL_DIR / "selected_random_forest_configs.csv"
PER_CLASS_PATH = MODEL_DIR / "validation_per_class_metrics.csv"
CONFUSION_PATH = MODEL_DIR / "validation_confusion_matrices.csv"
PREDICTIONS_PATH = MODEL_DIR / "validation_predictions.csv.gz"
COMPARISON_PATH = MODEL_DIR / "validation_model_family_comparison.csv"
SUMMARY_TXT_PATH = MODEL_DIR / "random_forest_validation_summary.txt"
SUMMARY_JSON_PATH = MODEL_DIR / "random_forest_validation_summary.json"


# ============================================================
# Configuração
# ============================================================

RANDOM_STATE = 42
NT_IMPUTATION_SEED = 314159

DEFAULT_N_ESTIMATORS = 300

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
    "network_degree_strength",
    "network_plus_centrality",
    "network_plus_leiden",
    "network_full",
    "bio_strict_plus_network_full",
    "bio_plus_availability_plus_network_full",
]

# Grade deliberadamente pequena.
# max_depth fica sem limite; a regularização é feita principalmente por
# min_samples_leaf e max_features.
RF_GRID = [
    {
        "config_id": "sqrt_leaf5",
        "max_features": "sqrt",
        "min_samples_leaf": 5,
        "complexity_rank": 0,
    },
    {
        "config_id": "half_leaf5",
        "max_features": 0.5,
        "min_samples_leaf": 5,
        "complexity_rank": 1,
    },
    {
        "config_id": "sqrt_leaf1",
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "complexity_rank": 2,
    },
    {
        "config_id": "half_leaf1",
        "max_features": 0.5,
        "min_samples_leaf": 1,
        "complexity_rank": 3,
    },
]


# ============================================================
# Transformador NT
# ============================================================

class JointNTVectorImputer(BaseEstimator, TransformerMixin):
    """
    Imputa vetores NT 6D inteiramente ausentes reamostrando vetores válidos
    observados SOMENTE no treino.

    Evita substituir todo missing por um vetor constante, o que poderia
    recriar um indicador determinístico de disponibilidade no BIO-strict.
    """

    def __init__(self, random_state=NT_IMPUTATION_SEED):
        self.random_state = random_state

    def fit(self, X, y=None):
        x = np.asarray(X, dtype=np.float64)

        if x.ndim != 2 or x.shape[1] != 6:
            raise ValueError(
                "JointNTVectorImputer espera exatamente 6 colunas."
            )

        complete = np.isfinite(x).all(axis=1)
        partial = np.isfinite(x).any(axis=1) & ~complete

        if partial.any():
            raise ValueError(
                "Foram encontrados vetores NT parcialmente ausentes."
            )

        self.valid_vectors_ = x[complete].copy()

        if len(self.valid_vectors_) == 0:
            raise ValueError(
                "Nenhum vetor NT válido disponível no treino."
            )

        self.n_features_in_ = 6
        return self

    def transform(self, X):
        x = np.asarray(X, dtype=np.float64).copy()

        complete = np.isfinite(x).all(axis=1)
        partial = np.isfinite(x).any(axis=1) & ~complete

        if partial.any():
            raise ValueError(
                "Foram encontrados vetores NT parcialmente ausentes."
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

            x[missing_idx] = self.valid_vectors_[sampled_idx]

        if not np.isfinite(x).all():
            raise RuntimeError(
                "Imputação NT deixou NaN ou infinito."
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

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "F2.10 — Random Forest em validação. "
            "O conjunto de teste não é avaliado nesta etapa."
        )
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=EXPERIMENTS,
        default=EXPERIMENTS,
        help="Subconjunto de experimentos a executar.",
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
        help=f"Número de árvores. Default: {DEFAULT_N_ESTIMATORS}.",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Paralelismo do Random Forest. Default: -1.",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Execução reduzida: amostra train/val, usa 50 árvores e somente "
            "sqrt_leaf5 para verificação de execução."
        ),
    )

    args = parser.parse_args()

    if args.n_estimators < 1:
        parser.error("--n-estimators deve ser >= 1.")

    return args


def section(title):
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def fmt_int(x):
    return f"{int(x):,}".replace(",", ".")


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(rows):
    if rows:
        pd.DataFrame(rows).to_csv(
            RESULTS_ALL_PATH,
            index=False,
        )


def stratified_sample(df, label_col, max_rows, seed):
    if len(df) <= max_rows:
        return df.copy()

    rng = np.random.default_rng(seed)
    parts = []

    counts = (
        df[label_col]
        .value_counts()
        .sort_index()
    )

    for label, count in counts.items():
        n_take = max(
            1,
            round(
                max_rows
                * count
                / len(df)
            ),
        )

        group = df[df[label_col] == label]
        n_take = min(n_take, len(group))

        idx = rng.choice(
            group.index.to_numpy(),
            size=n_take,
            replace=False,
        )

        parts.append(df.loc[idx])

    out = pd.concat(parts, ignore_index=True)

    if len(out) > max_rows:
        idx = rng.choice(
            out.index.to_numpy(),
            size=max_rows,
            replace=False,
        )
        out = out.loc[idx].copy()

    return out.reset_index(drop=True)


def build_feature_spec(experiment, bio_sets, topo_sets):
    if experiment == "bio_strict":
        return list(bio_sets["bio_strict"])

    if experiment == "bio_plus_availability":
        return list(bio_sets["bio_plus_availability"])

    if experiment in topo_sets:
        return list(topo_sets[experiment])

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

    raise ValueError(
        f"Experimento desconhecido: {experiment}"
    )


def make_preprocessor(feature_cols, topo_feature_names):
    """
    Árvores não precisam de scaling/log-transform.

    Ajustes necessários:
      side -> missing='unknown' -> one-hot;
      morphology -> mediana do treino;
      NT -> imputação conjunta aprendida no treino;
      nt_available -> passthrough;
      topologia -> passthrough.
    """
    feature_cols = list(feature_cols)
    transformers = []

    if SIDE_COL in feature_cols:
        side_pipe = Pipeline(
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
        )

        transformers.append(
            ("side", side_pipe, [SIDE_COL])
        )

    morph_cols = [
        c
        for c in MORPHOLOGY_COLS
        if c in feature_cols
    ]

    if morph_cols:
        morph_pipe = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
            ]
        )

        transformers.append(
            ("morphology", morph_pipe, morph_cols)
        )

    nt_cols = [
        c
        for c in NT_PROB_COLS
        if c in feature_cols
    ]

    if nt_cols:
        if nt_cols != NT_PROB_COLS:
            raise RuntimeError(
                "O bloco NT deve conter exatamente as 6 probabilidades."
            )

        nt_pipe = Pipeline(
            steps=[
                (
                    "joint_imputer",
                    JointNTVectorImputer(
                        random_state=NT_IMPUTATION_SEED,
                    ),
                ),
            ]
        )

        transformers.append(
            ("nt_probabilities", nt_pipe, nt_cols)
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
            ("topology", "passthrough", topo_cols)
        )

    if not transformers:
        raise RuntimeError(
            "Nenhum transformador criado."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def evaluate_predictions(y_true, y_pred, labels):
    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
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


def per_class_metrics(
    y_true,
    y_pred,
    labels,
    experiment,
    config_id,
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
            "experiment": experiment,
            "model": "random_forest_balanced_subsample",
            "config_id": config_id,
            "super_class": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(np.int64),
        }
    )


def confusion_long(
    y_true,
    y_pred,
    labels,
    experiment,
    config_id,
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
                    "experiment": experiment,
                    "model": "random_forest_balanced_subsample",
                    "config_id": config_id,
                    "true_class": true_label,
                    "pred_class": pred_label,
                    "count": int(cm[i, j]),
                }
            )

    return pd.DataFrame(rows)


def fit_random_forest(
    X_train,
    y_train,
    X_val,
    y_val,
    labels,
    feature_cols,
    topo_feature_names,
    config,
    n_estimators,
    n_jobs,
):
    preprocessor = make_preprocessor(
        feature_cols=feature_cols,
        topo_feature_names=topo_feature_names,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"],
        bootstrap=True,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    fit_start = time.time()

    pipeline.fit(
        X_train[feature_cols],
        y_train,
    )

    fit_seconds = time.time() - fit_start

    pred_start = time.time()

    y_pred = pipeline.predict(
        X_val[feature_cols]
    )

    predict_seconds = time.time() - pred_start

    metrics = evaluate_predictions(
        y_true=y_val,
        y_pred=y_pred,
        labels=labels,
    )

    metrics.update(
        {
            "fit_seconds": float(fit_seconds),
            "predict_seconds": float(predict_seconds),
        }
    )

    return pipeline, y_pred, metrics


def selected_comparison_with_logistic(rf_selected):
    if not LOGISTIC_SELECTED_PATH.exists():
        return None

    logistic = pd.read_csv(LOGISTIC_SELECTED_PATH)

    logistic = logistic[
        logistic["experiment"].isin(EXPERIMENTS)
    ].copy()

    logistic_keep = logistic[
        [
            "experiment",
            "macro_f1",
            "balanced_accuracy",
            "accuracy",
            "weighted_f1",
        ]
    ].rename(
        columns={
            "macro_f1": "logistic_macro_f1",
            "balanced_accuracy": "logistic_balanced_accuracy",
            "accuracy": "logistic_accuracy",
            "weighted_f1": "logistic_weighted_f1",
        }
    )

    rf_keep = rf_selected[
        [
            "experiment",
            "macro_f1",
            "balanced_accuracy",
            "accuracy",
            "weighted_f1",
            "config_id",
        ]
    ].rename(
        columns={
            "macro_f1": "rf_macro_f1",
            "balanced_accuracy": "rf_balanced_accuracy",
            "accuracy": "rf_accuracy",
            "weighted_f1": "rf_weighted_f1",
        }
    )

    comparison = logistic_keep.merge(
        rf_keep,
        on="experiment",
        how="outer",
        validate="one_to_one",
    )

    comparison["delta_macro_f1_rf_minus_logistic"] = (
        comparison["rf_macro_f1"]
        - comparison["logistic_macro_f1"]
    )

    comparison[
        "delta_balanced_accuracy_rf_minus_logistic"
    ] = (
        comparison["rf_balanced_accuracy"]
        - comparison["logistic_balanced_accuracy"]
    )

    return comparison


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    total_start = time.time()

    section("F2.10 — Random Forest (VALIDAÇÃO)")

    print(
        "Validação: o conjunto de teste não é avaliado nesta etapa."
    )
    print(
        "Seleção de configuração é feita somente por macro-F1 de validação."
    )

    if args.smoke:
        print()
        print(
            "Modo smoke ativo: execução reduzida para verificação."
        )

    required = [
        BIO_PATH,
        TOPO_PATH,
        TARGET_PATH,
        BIO_SETS_PATH,
        TOPO_SETS_PATH,
        MAIN_SPLIT_PATH,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo necessário não encontrado: {path}"
            )

    # --------------------------------------------------------
    # 1. Dados
    # --------------------------------------------------------

    section("1. Carregando dados preparados")

    bio_sets = load_json(BIO_SETS_PATH)
    topo_sets = load_json(TOPO_SETS_PATH)

    bio = pd.read_csv(BIO_PATH, low_memory=False)
    topo = pd.read_csv(TOPO_PATH, low_memory=False)
    target = pd.read_csv(
        TARGET_PATH,
        usecols=["root_id", "super_class"],
        low_memory=False,
    )
    split = pd.read_csv(
        MAIN_SPLIT_PATH,
        usecols=["root_id", "split"],
        low_memory=False,
    )

    for name, df in [
        ("BIO", bio),
        ("TOPO", topo),
        ("TARGET", target),
        ("SPLIT", split),
    ]:
        if df["root_id"].duplicated().any():
            raise RuntimeError(
                f"{name} possui root_id duplicado."
            )

    data = (
        target
        .merge(
            split,
            on="root_id",
            validate="one_to_one",
        )
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

    if len(data) != len(target):
        raise RuntimeError(
            "Merge alterou o universo de neurônios."
        )

    train = data[
        data["split"] == "train"
    ].copy()

    val = data[
        data["split"] == "val"
    ].copy()

    test_ids = set(
        data.loc[
            data["split"] == "test",
            "root_id",
        ].astype(np.int64)
    )

    test_count = len(test_ids)

    if args.smoke:
        train = stratified_sample(
            train,
            "super_class",
            max_rows=6000,
            seed=RANDOM_STATE,
        )

        val = stratified_sample(
            val,
            "super_class",
            max_rows=2500,
            seed=RANDOM_STATE + 1,
        )

        grid = [RF_GRID[0]]
        n_estimators = 50
    else:
        grid = RF_GRID
        n_estimators = args.n_estimators

    labels = sorted(
        data["super_class"].unique()
    )

    if len(labels) != 10:
        raise RuntimeError(
            f"Esperadas 10 classes; encontradas {len(labels)}."
        )

    print(f"Train usado: {fmt_int(len(train))}")
    print(f"Val usado:   {fmt_int(len(val))}")
    print(f"Linhas reservadas para teste: {fmt_int(test_count)} (não avaliadas)")
    print(f"Árvores/configuração: {n_estimators}")
    print(f"Configurações: {len(grid)}")
    print(f"n_jobs: {args.n_jobs}")

    X_train = train.drop(
        columns=["super_class", "split"]
    )
    y_train = train["super_class"].to_numpy()

    X_val = val.drop(
        columns=["super_class", "split"]
    )
    y_val = val["super_class"].to_numpy()

    topo_feature_names = set(
        topo.columns
    ) - {"root_id"}

    # --------------------------------------------------------
    # 2. Modelos
    # --------------------------------------------------------

    section("2. Random Forest — grade de validação")

    result_rows = []
    selected_rows = []
    per_class_parts = []
    confusion_parts = []

    predictions = pd.DataFrame(
        {
            "root_id": val["root_id"].to_numpy(
                dtype=np.int64
            ),
            "y_true": y_val,
        }
    )

    for experiment in args.experiments:
        feature_cols = build_feature_spec(
            experiment=experiment,
            bio_sets=bio_sets,
            topo_sets=topo_sets,
        )

        missing_cols = [
            c
            for c in feature_cols
            if c not in X_train.columns
        ]

        if missing_cols:
            raise RuntimeError(
                f"{experiment}: features ausentes: {missing_cols}"
            )

        print()
        print("-" * 88)
        print(
            f"{experiment} | {len(feature_cols)} features brutas"
        )
        print("-" * 88)

        candidates = []

        for config in grid:
            print(
                "Iniciando fit: "
                f"experiment={experiment}, "
                f"config={config['config_id']}, "
                f"trees={n_estimators}",
                flush=True,
            )

            pipeline, y_pred, metrics = fit_random_forest(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                labels=labels,
                feature_cols=feature_cols,
                topo_feature_names=topo_feature_names,
                config=config,
                n_estimators=n_estimators,
                n_jobs=args.n_jobs,
            )

            row = {
                "experiment": experiment,
                "model": "random_forest_balanced_subsample",
                "config_id": config["config_id"],
                "n_features_raw": int(len(feature_cols)),
                "n_estimators": int(n_estimators),
                "max_features": str(config["max_features"]),
                "min_samples_leaf": int(
                    config["min_samples_leaf"]
                ),
                "max_depth": "None",
                "class_weight": "balanced_subsample",
                "complexity_rank": int(
                    config["complexity_rank"]
                ),
                **metrics,
                "selected": False,
            }

            result_rows.append(row)
            save_checkpoint(result_rows)

            candidates.append(
                {
                    "row": row,
                    "pipeline": pipeline,
                    "y_pred": y_pred,
                }
            )

            print(
                f"{config['config_id']:<12} "
                f"macro-F1={metrics['macro_f1']:.6f}  "
                f"bal.acc={metrics['balanced_accuracy']:.6f}  "
                f"acc={metrics['accuracy']:.6f}  "
                f"time={metrics['fit_seconds']:.2f}s"
            )

        best = max(
            candidates,
            key=lambda item: (
                item["row"]["macro_f1"],
                item["row"]["balanced_accuracy"],
                -item["row"]["complexity_rank"],
            ),
        )

        for row in result_rows:
            if (
                row["experiment"] == experiment
                and row["config_id"]
                == best["row"]["config_id"]
            ):
                row["selected"] = True

        save_checkpoint(result_rows)

        best_row = dict(best["row"])
        best_row["selected"] = True
        selected_rows.append(best_row)

        print(
            f"SELECIONADO: {best_row['config_id']} | "
            f"macro-F1={best_row['macro_f1']:.6f}"
        )

        per_class_parts.append(
            per_class_metrics(
                y_true=y_val,
                y_pred=best["y_pred"],
                labels=labels,
                experiment=experiment,
                config_id=best_row["config_id"],
            )
        )

        confusion_parts.append(
            confusion_long(
                y_true=y_val,
                y_pred=best["y_pred"],
                labels=labels,
                experiment=experiment,
                config_id=best_row["config_id"],
            )
        )

        predictions[experiment] = best["y_pred"]

    # --------------------------------------------------------
    # 3. Consolidação
    # --------------------------------------------------------

    section("3. Consolidando resultados")

    results = pd.DataFrame(result_rows)
    selected = pd.DataFrame(selected_rows)

    order = pd.Categorical(
        selected["experiment"],
        categories=EXPERIMENTS,
        ordered=True,
    )

    selected = (
        selected.assign(experiment_order=order)
        .sort_values("experiment_order")
        .drop(columns="experiment_order")
        .reset_index(drop=True)
    )

    per_class = pd.concat(
        per_class_parts,
        ignore_index=True,
    )

    confusions = pd.concat(
        confusion_parts,
        ignore_index=True,
    )

    if predictions["root_id"].duplicated().any():
        raise RuntimeError(
            "Predições possuem root_id duplicado."
        )

    if set(
        predictions["root_id"].astype(np.int64)
    ) & test_ids:
        raise RuntimeError(
            "Predições de validação contêm root_id do TESTE."
        )

    comparison = selected_comparison_with_logistic(
        selected
    )

    # --------------------------------------------------------
    # 4. Salva
    # --------------------------------------------------------

    section("4. Salvando outputs")

    results.to_csv(
        RESULTS_ALL_PATH,
        index=False,
    )

    selected.to_csv(
        SELECTED_PATH,
        index=False,
    )

    per_class.to_csv(
        PER_CLASS_PATH,
        index=False,
    )

    confusions.to_csv(
        CONFUSION_PATH,
        index=False,
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
        compression="gzip",
    )

    if comparison is not None:
        comparison.to_csv(
            COMPARISON_PATH,
            index=False,
        )

    selected_lookup = (
        selected.set_index("experiment")
        .to_dict(orient="index")
    )

    summary = {
        "stage": "validation_only_model_selection",
        "test_set_evaluated": False,
        "random_state": RANDOM_STATE,
        "nt_imputation_seed": NT_IMPUTATION_SEED,
        "train_rows_used": int(len(train)),
        "validation_rows_used": int(len(val)),
        "test_rows_locked": int(test_count),
        "n_estimators": int(n_estimators),
        "n_jobs": int(args.n_jobs),
        "criterion": "gini",
        "bootstrap": True,
        "class_weight": "balanced_subsample",
        "grid": grid,
        "selection_rule": (
            "maximize validation macro-F1; "
            "tie-break by balanced accuracy; "
            "then prefer lower complexity_rank"
        ),
        "selected_results": selected_lookup,
        "preprocessing": {
            "side": (
                "missing='unknown' then one-hot; categories fitted on train"
            ),
            "morphology": (
                "median imputation fitted on train; no scaling/log required for trees"
            ),
            "nt_probabilities": (
                "joint resampling of complete valid NT vectors from TRAIN only"
            ),
            "nt_available": "passthrough",
            "topology": (
                "raw frozen numeric topological features; no scaling/log required for trees"
            ),
        },
        "important_notes": [
            "The main TEST split is not evaluated in this stage.",
            (
                "BIO, NETWORK and BIO+NETWORK use the exact same "
                "train/validation rows."
            ),
            "BIO-strict excludes nt_available and nt_type_score.",
            (
                "The RF grid is intentionally small because the scientific goal "
                "is comparison of information blocks, not exhaustive algorithm competition."
            ),
            (
                "Impurity feature importance is not used as a primary scientific result; "
                "block-level predictive comparisons are primary."
            ),
        ],
        "logistic_comparison_created": bool(
            comparison is not None
        ),
        "smoke_mode": bool(args.smoke),
        "runtime_seconds": float(
            time.time() - total_start
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

    text = [
        "FlyWire FAFB v783 — F2.10 Random Forest",
        "",
        "ESTÁGIO",
        "Somente seleção em VALIDAÇÃO.",
        "O conjunto de TESTE NÃO foi avaliado.",
        "",
        f"Train usado: {len(train)}",
        f"Validation usado: {len(val)}",
        f"Test reservado: {test_count} (não avaliado)",
        f"Árvores por configuração: {n_estimators}",
        "Class weight: balanced_subsample",
        "",
        "RESULTADOS SELECIONADOS",
    ]

    for _, row in selected.iterrows():
        text.append(
            f"{row['experiment']}: "
            f"config={row['config_id']}, "
            f"max_features={row['max_features']}, "
            f"min_samples_leaf={row['min_samples_leaf']}, "
            f"macro-F1={row['macro_f1']:.6f}, "
            f"balanced_acc={row['balanced_accuracy']:.6f}, "
            f"accuracy={row['accuracy']:.6f}, "
            f"weighted_F1={row['weighted_f1']:.6f}"
        )

    if comparison is not None:
        text.extend(
            [
                "",
                "RF - LOGISTIC (DELTA MACRO-F1)",
            ]
        )

        for _, row in comparison.iterrows():
            if pd.notna(
                row["delta_macro_f1_rf_minus_logistic"]
            ):
                text.append(
                    f"{row['experiment']}: "
                    f"{row['delta_macro_f1_rf_minus_logistic']:+.6f}"
                )

    text.extend(
        [
            "",
            "REGRAS METODOLÓGICAS",
            "- seleção pelo macro-F1 da validação;",
            "- nenhuma métrica do TESTE foi consultada;",
            "- mesmas linhas de train/val do F2.09;",
            "- class_weight='balanced_subsample';",
            "- preprocessamento ajustado somente no treino;",
            "- BIO-strict não contém nt_available nem nt_type_score;",
            "- grade pequena e previamente definida;",
            "",
            (
                "Tempo total: "
                f"{(time.time() - total_start) / 60:.2f} min"
            ),
        ]
    )

    SUMMARY_TXT_PATH.write_text(
        "\n".join(text),
        encoding="utf-8",
    )

    print(
        selected[
            [
                "experiment",
                "config_id",
                "macro_f1",
                "balanced_accuracy",
                "accuracy",
                "weighted_f1",
            ]
        ].to_string(index=False)
    )

    if comparison is not None:
        print()
        print("Comparação RF - Logistic:")
        print(
            comparison[
                [
                    "experiment",
                    "logistic_macro_f1",
                    "rf_macro_f1",
                    "delta_macro_f1_rf_minus_logistic",
                ]
            ].to_string(index=False)
        )

    print()
    print(f"Todos os candidatos: {RESULTS_ALL_PATH}")
    print(f"Selecionados:        {SELECTED_PATH}")
    print(f"Por classe:          {PER_CLASS_PATH}")
    print(f"Confusões:           {CONFUSION_PATH}")
    print(f"Predições:           {PREDICTIONS_PATH}")

    if comparison is not None:
        print(f"RF vs Logistic:      {COMPARISON_PATH}")

    print(f"Resumo:              {SUMMARY_TXT_PATH}")

    section("Concluído")
    print(
        f"Tempo total: {(time.time() - total_start) / 60:.2f} min"
    )


if __name__ == "__main__":
    main()
