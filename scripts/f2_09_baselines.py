from pathlib import Path
import argparse
import json
import os
import time
import warnings

# Windows/Conda safety: avoid native BLAS/OpenMP oversubscription and
# occasional hard process termination inside scikit-learn solvers.
# These variables must be set before importing NumPy/SciPy/scikit-learn.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd

try:
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.compose import ColumnTransformer
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
except ImportError as exc:
    raise SystemExit(
        "Este script requer scikit-learn. Instale com:\n"
        "  conda install -c conda-forge scikit-learn\n"
        "ou adicione 'scikit-learn' ao environment.yml."
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

MODEL_DIR = ROOT / "reports" / "f2" / "models" / "logistic_validation"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_ALL_PATH = MODEL_DIR / "validation_results_all.csv"
SELECTED_PATH = MODEL_DIR / "selected_logistic_configs.csv"
PER_CLASS_PATH = MODEL_DIR / "validation_per_class_metrics.csv"
CONFUSION_PATH = MODEL_DIR / "validation_confusion_matrices.csv"
PREDICTIONS_PATH = MODEL_DIR / "validation_predictions.csv.gz"
SUMMARY_TXT_PATH = MODEL_DIR / "logistic_validation_summary.txt"
SUMMARY_JSON_PATH = MODEL_DIR / "logistic_validation_summary.json"


# ============================================================
# Configuração experimental
# ============================================================

DEFAULT_C_GRID = [0.1, 1.0, 10.0]

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

# Ordem dos experimentos no relatório.
LOGREG_EXPERIMENTS = [
    "nt_available_only",
    "bio_strict",
    "bio_plus_availability",
    "network_degree_strength",
    "network_plus_centrality",
    "network_plus_leiden",
    "network_full",
    "bio_strict_plus_network_full",
    "bio_plus_availability_plus_network_full",
]

# nt_available sozinho já é avaliado por um baseline condicional simples
# (majoritária dentro de availability=0/1). Mantemos a opção de rodar uma
# Logistic Regression de uma feature manualmente, mas ela não entra no
# conjunto padrão porque class_weight='balanced' pode produzir um baseline
# pouco interpretável quando há 10 classes e somente uma variável binária.
DEFAULT_LOGREG_EXPERIMENTS = [
    experiment
    for experiment in LOGREG_EXPERIMENTS
    if experiment != "nt_available_only"
]


# ============================================================
# Transformadores
# ============================================================

def log1p_array(x):
    """
    log(1+x) para features estritamente não negativas.
    """
    x = np.asarray(x, dtype=np.float64)

    if np.nanmin(x) < 0:
        raise ValueError(
            "log1p_array recebeu valor negativo."
        )

    return np.log1p(x)


class JointNTVectorImputer(BaseEstimator, TransformerMixin):
    """
    Imputação conjunta dos 6 scores/probabilidades de neurotransmissor.

    Os vetores inteiramente ausentes do BIO-strict foram convertidos para NaN
    em f2_07. Em vez de usar um vetor constante (que denunciaria missingness),
    esta transformação reamostra VETORES COMPLETOS observados somente no
    TREINO.

    Isso preserva:
      - correlação interna entre as 6 dimensões;
      - ausência de um marcador determinístico de missingness;
      - separação estrita entre treino e validação/teste.

    A seed é fixa para reprodutibilidade.

    Observação:
    Este procedimento remove o indicador EXPLÍCITO de disponibilidade de NT,
    mas não pretende remover toda associação indireta com observabilidade.
    """

    def __init__(self, random_state=NT_IMPUTATION_SEED):
        self.random_state = random_state

    def fit(self, X, y=None):
        x = np.asarray(X, dtype=np.float64)

        if x.ndim != 2 or x.shape[1] != 6:
            raise ValueError(
                "JointNTVectorImputer espera matriz com exatamente 6 colunas."
            )

        complete = np.isfinite(x).all(axis=1)
        partial = (
            np.isfinite(x).any(axis=1)
            & ~complete
        )

        if partial.any():
            raise ValueError(
                "Foram encontrados vetores NT parcialmente ausentes. "
                "O desenho esperado é bloco completo válido ou bloco completo NaN."
            )

        self.valid_vectors_ = x[
            complete
        ].copy()

        if len(self.valid_vectors_) == 0:
            raise ValueError(
                "Nenhum vetor NT válido disponível no treino."
            )

        self.n_features_in_ = x.shape[1]

        return self

    def transform(self, X):
        if not hasattr(
            self,
            "valid_vectors_",
        ):
            raise RuntimeError(
                "JointNTVectorImputer ainda não foi ajustado."
            )

        x = np.asarray(
            X,
            dtype=np.float64,
        ).copy()

        complete = np.isfinite(x).all(axis=1)
        partial = (
            np.isfinite(x).any(axis=1)
            & ~complete
        )

        if partial.any():
            raise ValueError(
                "Foram encontrados vetores NT parcialmente ausentes."
            )

        missing_idx = np.flatnonzero(
            ~complete
        )

        if len(missing_idx):
            # A amostragem depende somente da distribuição aprendida no treino.
            # Incluímos n_rows na seed para que train/val com tamanhos distintos
            # não recebam exatamente a mesma sequência de amostras.
            seed = (
                int(self.random_state)
                + 104729 * len(x)
            ) % (2**32 - 1)

            rng = np.random.default_rng(
                seed
            )

            sampled = rng.integers(
                0,
                len(self.valid_vectors_),
                size=len(missing_idx),
            )

            x[missing_idx] = (
                self.valid_vectors_[
                    sampled
                ]
            )

        if not np.isfinite(x).all():
            raise RuntimeError(
                "Imputação NT deixou NaN/infinito."
            )

        return x


# ============================================================
# Auxiliares
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "F2.09 — baselines + Logistic Regression em validação. "
            "O conjunto de TESTE não é avaliado neste script."
        )
    )

    parser.add_argument(
        "--c-grid",
        nargs="+",
        type=float,
        default=DEFAULT_C_GRID,
        help=(
            "Valores de C para Logistic Regression. "
            "Default: 0.1 1.0 10.0"
        ),
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=LOGREG_EXPERIMENTS,
        default=DEFAULT_LOGREG_EXPERIMENTS,
        help=(
            "Subconjunto de experimentos Logistic Regression a executar."
        ),
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=500,
        help="Máximo de iterações da Logistic Regression. Default: 500.",
    )

    parser.add_argument(
        "--solver",
        choices=["lbfgs", "saga"],
        default="lbfgs",
        help=(
            "Solver da Logistic Regression. Default: lbfgs. "
            "Solver alternativo disponível para ambientes com incompatibilidade nativa: saga."
        ),
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Execução reduzida: usa amostras estratificadas menores de train/val "
            "e força C=1.0 para verificação de execução."
        ),
    )

    args = parser.parse_args()

    if any(c <= 0 for c in args.c_grid):
        parser.error("Todos os valores de C devem ser > 0.")

    if args.max_iter < 1:
        parser.error("--max-iter deve ser >= 1.")

    return args


def section(title):
    print()
    print("=" * 84)
    print(title)
    print("=" * 84)


def fmt_int(x):
    return f"{int(x):,}".replace(",", ".")


def save_results_checkpoint(results_rows):
    """Salva resultados parciais para não perder tudo se houver falha nativa."""
    if not results_rows:
        return

    pd.DataFrame(results_rows).to_csv(
        RESULTS_ALL_PATH,
        index=False,
    )


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def stratified_sample(
    df,
    label_col,
    max_rows,
    seed,
):
    """
    Apenas para --smoke.
    Mantém aproximadamente a distribuição de classes.
    """
    if len(df) <= max_rows:
        return df.copy()

    rng = np.random.default_rng(seed)

    parts = []

    class_counts = (
        df[label_col]
        .value_counts()
        .sort_index()
    )

    for label, count in class_counts.items():
        target = max(
            1,
            int(
                round(
                    max_rows
                    * count
                    / len(df)
                )
            ),
        )

        group = df[
            df[label_col] == label
        ]

        target = min(
            target,
            len(group),
        )

        selected = rng.choice(
            group.index.to_numpy(),
            size=target,
            replace=False,
        )

        parts.append(
            df.loc[selected]
        )

    out = pd.concat(
        parts,
        ignore_index=True,
    )

    if len(out) > max_rows:
        selected = rng.choice(
            out.index.to_numpy(),
            size=max_rows,
            replace=False,
        )
        out = out.loc[
            selected
        ].copy()

    return out.reset_index(drop=True)


def build_feature_spec(
    experiment,
    bio_sets,
    topo_sets,
):
    """
    Retorna a whitelist exata para cada experimento.
    """
    if experiment == "nt_available_only":
        return [
            NT_AVAILABLE_COL
        ]

    if experiment == "bio_strict":
        return list(
            bio_sets[
                "bio_strict"
            ]
        )

    if experiment == "bio_plus_availability":
        return list(
            bio_sets[
                "bio_plus_availability"
            ]
        )

    if experiment in topo_sets:
        return list(
            topo_sets[
                experiment
            ]
        )

    if experiment == "bio_strict_plus_network_full":
        return (
            list(
                bio_sets[
                    "bio_strict"
                ]
            )
            + list(
                topo_sets[
                    "network_full"
                ]
            )
        )

    if experiment == "bio_plus_availability_plus_network_full":
        return (
            list(
                bio_sets[
                    "bio_plus_availability"
                ]
            )
            + list(
                topo_sets[
                    "network_full"
                ]
            )
        )

    raise ValueError(
        f"Experimento desconhecido: {experiment}"
    )


def infer_topology_preprocessing(
    topo_manifest,
    feature_cols,
):
    """
    Separa features topológicas conforme o manifesto congelado em F2.06.
    """
    relevant = topo_manifest[
        topo_manifest[
            "feature"
        ].isin(feature_cols)
    ]

    log_cols = relevant.loc[
        relevant[
            "recommended_preprocessing"
        ]
        == "log1p_then_scale",
        "feature",
    ].tolist()

    scale_cols = relevant.loc[
        relevant[
            "recommended_preprocessing"
        ]
        == "scale",
        "feature",
    ].tolist()

    covered = set(
        log_cols
        + scale_cols
    )

    expected = set(
        feature_cols
    ) & set(
        topo_manifest[
            "feature"
        ]
    )

    if covered != expected:
        missing = sorted(
            expected - covered
        )

        raise RuntimeError(
            "Manifesto topológico não define preprocessing para: "
            + ", ".join(missing)
        )

    return log_cols, scale_cols


def make_preprocessor(
    feature_cols,
    topo_manifest,
):
    """
    Cria o preprocessing para uma determinada whitelist.

    Tudo é ajustado exclusivamente nos dados passados ao .fit():
      side:
        missing -> 'unknown' -> one-hot
      morphology:
        mediana do treino -> log1p -> StandardScaler
      NT:
        reamostragem conjunta de vetores válidos do treino -> StandardScaler
      nt_available:
        passthrough
      topologia log:
        log1p -> StandardScaler
      topologia scale:
        StandardScaler
    """
    feature_cols = list(
        feature_cols
    )

    transformers = []

    if SIDE_COL in feature_cols:
        side_pipeline = Pipeline(
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
            (
                "side",
                side_pipeline,
                [SIDE_COL],
            )
        )

    morph_cols = [
        c
        for c in MORPHOLOGY_COLS
        if c in feature_cols
    ]

    if morph_cols:
        morph_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
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
        )

        transformers.append(
            (
                "morphology",
                morph_pipeline,
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
                "As probabilidades NT devem entrar como bloco completo de 6 colunas."
            )

        nt_pipeline = Pipeline(
            steps=[
                (
                    "joint_imputer",
                    JointNTVectorImputer(
                        random_state=NT_IMPUTATION_SEED,
                    ),
                ),
                (
                    "scale",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "nt_probabilities",
                nt_pipeline,
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

    topo_feature_cols = [
        c
        for c in feature_cols
        if c in set(
            topo_manifest[
                "feature"
            ]
        )
    ]

    if topo_feature_cols:
        topo_log_cols, topo_scale_cols = (
            infer_topology_preprocessing(
                topo_manifest=topo_manifest,
                feature_cols=topo_feature_cols,
            )
        )

        if topo_log_cols:
            topology_log_pipeline = Pipeline(
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
            )

            transformers.append(
                (
                    "topology_log",
                    topology_log_pipeline,
                    topo_log_cols,
                )
            )

        if topo_scale_cols:
            transformers.append(
                (
                    "topology_scale",
                    StandardScaler(),
                    topo_scale_cols,
                )
            )

    if not transformers:
        raise RuntimeError(
            "Nenhum transformador criado para o experimento."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def evaluate_predictions(
    y_true,
    y_pred,
    labels,
):
    accuracy = float(
        accuracy_score(
            y_true,
            y_pred,
        )
    )

    balanced = float(
        balanced_accuracy_score(
            y_true,
            y_pred,
        )
    )

    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )

    weighted_f1 = float(
        f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def per_class_metrics(
    y_true,
    y_pred,
    labels,
    experiment,
    model_name,
    C=None,
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
            "model": model_name,
            "C": C,
            "super_class": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(
                np.int64
            ),
        }
    )


def confusion_long(
    y_true,
    y_pred,
    labels,
    experiment,
    model_name,
    C=None,
):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    rows = []

    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            rows.append({
                "experiment": experiment,
                "model": model_name,
                "C": C,
                "true_class": true_label,
                "pred_class": pred_label,
                "count": int(
                    cm[i, j]
                ),
            })

    return pd.DataFrame(
        rows
    )


def fit_logistic(
    X_train,
    y_train,
    X_val,
    y_val,
    labels,
    feature_cols,
    topo_manifest,
    C,
    max_iter,
    solver,
):
    preprocessor = make_preprocessor(
        feature_cols=feature_cols,
        topo_manifest=topo_manifest,
    )

    model = LogisticRegression(
        C=C,
        solver=solver,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocess",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    fit_start = time.time()

    convergence_messages = []

    with warnings.catch_warnings(
        record=True
    ) as captured:
        warnings.simplefilter(
            "always",
            ConvergenceWarning,
        )

        pipeline.fit(
            X_train[
                feature_cols
            ],
            y_train,
        )

        for w in captured:
            if issubclass(
                w.category,
                ConvergenceWarning,
            ):
                convergence_messages.append(
                    str(w.message)
                )

    fit_seconds = (
        time.time()
        - fit_start
    )

    predict_start = time.time()

    y_pred = pipeline.predict(
        X_val[
            feature_cols
        ]
    )

    predict_seconds = (
        time.time()
        - predict_start
    )

    metrics = evaluate_predictions(
        y_true=y_val,
        y_pred=y_pred,
        labels=labels,
    )

    n_iter = int(
        np.max(
            pipeline.named_steps[
                "model"
            ].n_iter_
        )
    )

    metrics.update({
        "C": float(C),
        "fit_seconds": float(
            fit_seconds
        ),
        "predict_seconds": float(
            predict_seconds
        ),
        "n_iter": n_iter,
        "converged": (
            len(convergence_messages)
            == 0
        ),
        "convergence_warning": (
            " | ".join(
                convergence_messages
            )
            if convergence_messages
            else ""
        ),
    })

    return (
        pipeline,
        y_pred,
        metrics,
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    total_start = time.time()

    section(
        "F2.09 — Baselines e Logistic Regression (VALIDAÇÃO)"
    )

    print(
        "Validação: o conjunto de teste não é avaliado nesta etapa."
    )
    print(
        "Seleção de C e comparação entre blocos são feitas somente em validação."
    )

    if args.smoke:
        print()
        print(
            "Modo smoke ativo: execução reduzida para verificação."
        )

    required_paths = [
        BIO_PATH,
        TOPO_PATH,
        TARGET_PATH,
        BIO_SETS_PATH,
        TOPO_SETS_PATH,
        MAIN_SPLIT_PATH,
        FEATURE_DIR
        / "topological_feature_manifest.csv",
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo necessário não encontrado: {path}"
            )

    # --------------------------------------------------------
    # 1. Carrega dados preparados
    # --------------------------------------------------------

    section("1. Carregando features, target e split")

    bio_sets = load_json(
        BIO_SETS_PATH
    )
    topo_sets = load_json(
        TOPO_SETS_PATH
    )

    topo_manifest = pd.read_csv(
        FEATURE_DIR
        / "topological_feature_manifest.csv"
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
        low_memory=False,
    )

    split = pd.read_csv(
        MAIN_SPLIT_PATH,
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
        target[
            [
                "root_id",
                "super_class",
            ]
        ]
        .merge(
            split[
                [
                    "root_id",
                    "split",
                ]
            ],
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

    if data["super_class"].isna().any():
        raise RuntimeError(
            "Há target ausente."
        )

    # A partição de teste permanece no dataframe, mas não entra
    # em nenhum cálculo abaixo.
    train = data[
        data["split"] == "train"
    ].copy()

    val = data[
        data["split"] == "val"
    ].copy()

    test_count = int(
        (data["split"] == "test").sum()
    )

    if args.smoke:
        train = stratified_sample(
            train,
            label_col="super_class",
            max_rows=5000,
            seed=RANDOM_STATE,
        )

        val = stratified_sample(
            val,
            label_col="super_class",
            max_rows=2000,
            seed=RANDOM_STATE + 1,
        )

        c_grid = [1.0]
    else:
        c_grid = list(
            args.c_grid
        )

    labels = sorted(
        data[
            "super_class"
        ].unique()
    )

    if len(labels) != 10:
        raise RuntimeError(
            f"Esperadas 10 classes; encontradas {len(labels)}."
        )

    for subset_name, subset in [
        ("train", train),
        ("val", val),
    ]:
        missing_classes = sorted(
            set(labels)
            - set(
                subset[
                    "super_class"
                ].unique()
            )
        )

        if missing_classes:
            raise RuntimeError(
                f"{subset_name} sem classes: {missing_classes}"
            )

    print(
        f"Train usado: {fmt_int(len(train))}"
    )
    print(
        f"Val usado:   {fmt_int(len(val))}"
    )
    print(
        f"Linhas reservadas para teste: {fmt_int(test_count)} (não avaliadas)"
    )
    print(
        f"Classes:     {len(labels)}"
    )
    print(
        "C grid:      "
        + ", ".join(
            str(x)
            for x in c_grid
        )
    )
    print(
        f"Solver:      {args.solver}"
    )
    print(
        "Native threads: 1 (OMP/MKL/OpenBLAS/NumExpr)"
    )

    X_train = train.drop(
        columns=[
            "super_class",
            "split",
        ]
    )
    y_train = train[
        "super_class"
    ].to_numpy()

    X_val = val.drop(
        columns=[
            "super_class",
            "split",
        ]
    )
    y_val = val[
        "super_class"
    ].to_numpy()

    # --------------------------------------------------------
    # 2. Baseline global majoritário
    # --------------------------------------------------------

    section("2. Baseline majoritário global")

    majority_class = (
        pd.Series(
            y_train
        )
        .value_counts()
        .idxmax()
    )

    majority_pred = np.full(
        len(y_val),
        majority_class,
        dtype=object,
    )

    majority_metrics = (
        evaluate_predictions(
            y_true=y_val,
            y_pred=majority_pred,
            labels=labels,
        )
    )

    print(
        f"Classe majoritária: {majority_class}"
    )
    print(
        f"Accuracy:           {majority_metrics['accuracy']:.6f}"
    )
    print(
        f"Balanced accuracy:  {majority_metrics['balanced_accuracy']:.6f}"
    )
    print(
        f"Macro-F1:           {majority_metrics['macro_f1']:.6f}"
    )

    results_rows = [{
        "experiment": "majority_global",
        "model": "majority_baseline",
        "C": np.nan,
        "n_features_raw": 0,
        **majority_metrics,
        "fit_seconds": 0.0,
        "predict_seconds": 0.0,
        "n_iter": 0,
        "converged": True,
        "convergence_warning": "",
        "selected": True,
    }]

    per_class_parts = [
        per_class_metrics(
            y_true=y_val,
            y_pred=majority_pred,
            labels=labels,
            experiment="majority_global",
            model_name="majority_baseline",
            C=None,
        )
    ]

    confusion_parts = [
        confusion_long(
            y_true=y_val,
            y_pred=majority_pred,
            labels=labels,
            experiment="majority_global",
            model_name="majority_baseline",
            C=None,
        )
    ]

    selected_predictions = pd.DataFrame(
        {
            "root_id": val[
                "root_id"
            ].to_numpy(
                dtype=np.int64
            ),
            "y_true": y_val,
            "majority_global": majority_pred,
        }
    )

    # --------------------------------------------------------
    # 3. Baseline condicional por nt_available
    # --------------------------------------------------------

    section("3. Baseline condicional por nt_available")

    train_nt = train[
        [
            NT_AVAILABLE_COL,
            "super_class",
        ]
    ].copy()

    lookup = (
        train_nt.groupby(
            NT_AVAILABLE_COL,
            observed=True,
        )["super_class"]
        .agg(
            lambda s:
                s.value_counts()
                .idxmax()
        )
        .to_dict()
    )

    nt_lookup_pred = (
        val[
            NT_AVAILABLE_COL
        ]
        .map(lookup)
        .fillna(
            majority_class
        )
        .to_numpy()
    )

    nt_lookup_metrics = (
        evaluate_predictions(
            y_true=y_val,
            y_pred=nt_lookup_pred,
            labels=labels,
        )
    )

    print(
        "Regra aprendida no treino: "
        + ", ".join(
            f"{int(k)} -> {v}"
            for k, v in sorted(
                lookup.items()
            )
        )
    )
    print(
        f"Accuracy:           {nt_lookup_metrics['accuracy']:.6f}"
    )
    print(
        f"Balanced accuracy:  {nt_lookup_metrics['balanced_accuracy']:.6f}"
    )
    print(
        f"Macro-F1:           {nt_lookup_metrics['macro_f1']:.6f}"
    )

    results_rows.append({
        "experiment": "nt_available_lookup",
        "model": "conditional_majority_baseline",
        "C": np.nan,
        "n_features_raw": 1,
        **nt_lookup_metrics,
        "fit_seconds": 0.0,
        "predict_seconds": 0.0,
        "n_iter": 0,
        "converged": True,
        "convergence_warning": "",
        "selected": True,
    })

    # Checkpoint inicial: mesmo que um solver nativo encerre o processo depois,
    # os baselines já ficam registrados em disco.
    save_results_checkpoint(
        results_rows
    )

    per_class_parts.append(
        per_class_metrics(
            y_true=y_val,
            y_pred=nt_lookup_pred,
            labels=labels,
            experiment="nt_available_lookup",
            model_name="conditional_majority_baseline",
            C=None,
        )
    )

    confusion_parts.append(
        confusion_long(
            y_true=y_val,
            y_pred=nt_lookup_pred,
            labels=labels,
            experiment="nt_available_lookup",
            model_name="conditional_majority_baseline",
            C=None,
        )
    )

    selected_predictions[
        "nt_available_lookup"
    ] = nt_lookup_pred

    # --------------------------------------------------------
    # 4. Logistic Regression
    # --------------------------------------------------------

    section("4. Logistic Regression — validação")

    selected_rows = []

    for experiment in args.experiments:
        feature_cols = build_feature_spec(
            experiment=experiment,
            bio_sets=bio_sets,
            topo_sets=topo_sets,
        )

        missing_cols = [
            col
            for col in feature_cols
            if col not in X_train.columns
        ]

        if missing_cols:
            raise RuntimeError(
                f"{experiment}: colunas ausentes: {missing_cols}"
            )

        print()
        print("-" * 84)
        print(
            f"{experiment} | {len(feature_cols)} features brutas"
        )
        print("-" * 84)

        experiment_candidates = []

        for C in c_grid:
            print(
                f"Iniciando fit: experiment={experiment}, C={C}, solver={args.solver}",
                flush=True,
            )

            (
                pipeline,
                y_pred,
                metrics,
            ) = fit_logistic(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                labels=labels,
                feature_cols=feature_cols,
                topo_manifest=topo_manifest,
                C=C,
                max_iter=args.max_iter,
                solver=args.solver,
            )

            row = {
                "experiment": experiment,
                "model": "logistic_regression_balanced",
                "C": float(C),
                "n_features_raw": int(
                    len(feature_cols)
                ),
                **metrics,
                "selected": False,
            }

            results_rows.append(
                row
            )

            # Checkpoint após cada configuração concluída.
            save_results_checkpoint(
                results_rows
            )

            experiment_candidates.append(
                {
                    "row": row,
                    "pipeline": pipeline,
                    "y_pred": y_pred,
                }
            )

            print(
                f"C={C:<6g} "
                f"macro-F1={metrics['macro_f1']:.6f}  "
                f"bal.acc={metrics['balanced_accuracy']:.6f}  "
                f"acc={metrics['accuracy']:.6f}  "
                f"iter={metrics['n_iter']}  "
                f"converged={metrics['converged']}  "
                f"time={metrics['fit_seconds']:.2f}s"
            )

        # Seleção somente por macro-F1 na validação.
        # Desempate: balanced accuracy, depois menor C.
        best = max(
            experiment_candidates,
            key=lambda item: (
                item["row"][
                    "macro_f1"
                ],
                item["row"][
                    "balanced_accuracy"
                ],
                -item["row"]["C"],
            ),
        )

        # Marca como selected no objeto original em results_rows.
        for row in results_rows:
            if (
                row["experiment"]
                == experiment
                and row["model"]
                == "logistic_regression_balanced"
                and row["C"]
                == best["row"]["C"]
            ):
                row["selected"] = True

        best_row = dict(
            best["row"]
        )
        best_row["selected"] = True

        selected_rows.append(
            best_row
        )

        print(
            f"SELECIONADO: C={best_row['C']} | "
            f"macro-F1={best_row['macro_f1']:.6f}"
        )

        per_class_parts.append(
            per_class_metrics(
                y_true=y_val,
                y_pred=best["y_pred"],
                labels=labels,
                experiment=experiment,
                model_name="logistic_regression_balanced",
                C=best_row["C"],
            )
        )

        confusion_parts.append(
            confusion_long(
                y_true=y_val,
                y_pred=best["y_pred"],
                labels=labels,
                experiment=experiment,
                model_name="logistic_regression_balanced",
                C=best_row["C"],
            )
        )

        selected_predictions[
            experiment
        ] = best[
            "y_pred"
        ]

    # --------------------------------------------------------
    # 5. Consolidação e auditoria
    # --------------------------------------------------------

    section("5. Consolidando resultados")

    results = pd.DataFrame(
        results_rows
    )

    selected = pd.DataFrame(
        [
            row
            for row in results_rows
            if row["selected"]
        ]
    )

    # Ordenação amigável.
    experiment_order = [
        "majority_global",
        "nt_available_lookup",
    ] + LOGREG_EXPERIMENTS

    selected[
        "experiment_order"
    ] = pd.Categorical(
        selected[
            "experiment"
        ],
        categories=experiment_order,
        ordered=True,
    )

    selected = (
        selected.sort_values(
            "experiment_order"
        )
        .drop(
            columns=[
                "experiment_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    per_class = pd.concat(
        per_class_parts,
        ignore_index=True,
    )

    confusions = pd.concat(
        confusion_parts,
        ignore_index=True,
    )

    if selected_predictions[
        "root_id"
    ].duplicated().any():
        raise RuntimeError(
            "Predições possuem root_id duplicado."
        )

    if len(
        selected_predictions
    ) != len(val):
        raise RuntimeError(
            "Número de predições difere do validation set."
        )

    # TEST não foi avaliado.
    if set(
        selected_predictions[
            "root_id"
        ]
    ) & set(
        data.loc[
            data["split"] == "test",
            "root_id",
        ]
    ):
        raise RuntimeError(
            "Predições de validação contêm root_id do teste."
        )

    # --------------------------------------------------------
    # 6. Salva
    # --------------------------------------------------------

    section("6. Salvando resultados")

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

    selected_predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
        compression="gzip",
    )

    selected_lookup = (
        selected.set_index(
            "experiment"
        )
        .to_dict(
            orient="index"
        )
    )

    summary = {
        "stage": (
            "validation_only_model_selection"
        ),
        "test_set_evaluated": False,
        "random_state": RANDOM_STATE,
        "nt_imputation_seed": (
            NT_IMPUTATION_SEED
        ),
        "class_weight": "balanced",
        "solver": args.solver,
        "max_iter": args.max_iter,
        "C_grid": c_grid,
        "train_rows_used": int(
            len(train)
        ),
        "validation_rows_used": int(
            len(val)
        ),
        "test_rows_locked": int(
            test_count
        ),
        "labels": labels,
        "majority_class": str(
            majority_class
        ),
        "nt_available_lookup": {
            str(int(k)): str(v)
            for k, v in lookup.items()
        },
        "selected_results": (
            selected_lookup
        ),
        "selection_rule": (
            "maximize validation macro-F1; tie-break by balanced accuracy, then smaller C"
        ),
        "preprocessing": {
            "side": (
                "training-independent missing token 'unknown' then one-hot; "
                "categories fitted on train"
            ),
            "morphology": (
                "training median imputation -> log1p -> StandardScaler"
            ),
            "nt_probabilities": (
                "joint resampling of complete valid NT vectors from TRAIN only "
                "for missing 6D blocks -> StandardScaler"
            ),
            "nt_available": "passthrough",
            "topology_log_features": (
                "log1p -> StandardScaler, fitted on train"
            ),
            "topology_other_features": (
                "StandardScaler, fitted on train"
            ),
        },
        "important_notes": [
            (
                "The main test split is not evaluated in F2.09."
            ),
            (
                "BIO, NETWORK and BIO+NETWORK use the exact same train/validation rows."
            ),
            (
                "BIO-strict does not include nt_available or nt_type_score."
            ),
            (
                "Missing NT probability vectors are not replaced by zeros or a fixed mean; "
                "they are jointly resampled from valid training vectors to avoid an explicit "
                "deterministic missingness marker."
            ),
            (
                "class_weight='balanced' is used because macro-F1 is the primary metric "
                "and the target distribution is strongly imbalanced."
            ),
        ],
        "runtime_seconds": float(
            time.time()
            - total_start
        ),
        "smoke_mode": bool(
            args.smoke
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
        "FlyWire FAFB v783 — F2.09 Baselines + Logistic Regression",
        "",
        "ESTÁGIO",
        "Somente seleção em VALIDAÇÃO.",
        "O conjunto de TESTE NÃO foi avaliado.",
        "",
        f"Train usado: {len(train)}",
        f"Validation usado: {len(val)}",
        f"Test reservado: {test_count} (não avaliado)",
        f"Class weight: balanced",
        f"Solver: {args.solver}",
        f"C grid: {c_grid}",
        f"NT imputation seed: {NT_IMPUTATION_SEED}",
        "",
        "RESULTADOS SELECIONADOS",
    ]

    for _, row in selected.iterrows():
        c_text = (
            "-"
            if pd.isna(
                row["C"]
            )
            else str(
                row["C"]
            )
        )

        text.append(
            f"{row['experiment']}: "
            f"model={row['model']}, "
            f"C={c_text}, "
            f"macro-F1={row['macro_f1']:.6f}, "
            f"balanced_acc={row['balanced_accuracy']:.6f}, "
            f"accuracy={row['accuracy']:.6f}, "
            f"weighted_F1={row['weighted_f1']:.6f}"
        )

    text.extend([
        "",
        "REGRAS METODOLÓGICAS",
        "- seleção pelo macro-F1 de validação;",
        "- desempate por balanced accuracy e depois menor C;",
        "- nenhuma métrica do teste foi consultada;",
        "- BIO/REDE/combinado usam as mesmas linhas de treino/validação;",
        "- imputação e scaling ajustados somente no treino;",
        f"- Logistic Regression usa solver={args.solver} e class_weight='balanced';",
        "- threads nativas BLAS/OpenMP limitadas a 1 por estabilidade/reprodutibilidade;",
        "- BIO-strict não contém nt_available nem nt_type_score;",
        "",
        (
            "Tempo total: "
            f"{(time.time() - total_start) / 60:.2f} min"
        ),
    ])

    SUMMARY_TXT_PATH.write_text(
        "\n".join(text),
        encoding="utf-8",
    )

    print(
        selected[
            [
                "experiment",
                "model",
                "C",
                "macro_f1",
                "balanced_accuracy",
                "accuracy",
                "weighted_f1",
                "converged",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Todos os candidatos: {RESULTS_ALL_PATH}"
    )
    print(
        f"Selecionados:        {SELECTED_PATH}"
    )
    print(
        f"Por classe:          {PER_CLASS_PATH}"
    )
    print(
        f"Confusões:           {CONFUSION_PATH}"
    )
    print(
        f"Predições:           {PREDICTIONS_PATH}"
    )
    print(
        f"Resumo:              {SUMMARY_TXT_PATH}"
    )

    section("Concluído")

    print(
        f"Tempo total: "
        f"{(time.time() - total_start) / 60:.2f} min"
    )


if __name__ == "__main__":
    main()
