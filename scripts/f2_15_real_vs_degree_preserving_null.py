"""
F2.15 — Auditoria relacional com null model degree-preserving

Objetivo
--------
Testar se a informação preditiva sobre super_class está apenas no grau dos
neurônios ou também na organização específica de quem conecta com quem.

Desenho
-------
1) Constrói a projeção simples NÃO DIRIGIDA do conectoma usando apenas
   (pre_root_id, post_root_id). Pesos, neuropil e NT NÃO entram neste controle.
2) Computa um bloco estrutural idêntico na rede real e em N redes rewired:
      - undirected_degree
      - local_clustering
      - coreness
      - leiden_module_size
      - leiden_internal_degree_fraction
      - leiden_within_module_degree_z
3) Cada nulo é produzido por degree-preserving edge swaps. O grau de CADA nó
   deve ser exatamente igual ao da rede real; há asserts para isso.
4) Treina os mesmos paradigmas de modelo, SEM acessar o teste:
      - Logistic Regression
      - Random Forest
   Hiperparâmetros são herdados das configurações já congeladas de network_full.
5) Compara:
      real vs distribuição dos nulls
   em:
      - todas as superclasses
      - flow == intrinsic (controle principal condicionado a flow)

Interpretação
-------------
Se o desempenho da rede real for muito maior que o dos nulls, então o grau
sozinho não explica o sinal. Há informação em padrões relacionais de ordem
superior / organização mesoscópica.

IMPORTANTE:
Este experimento preserva grau na projeção não dirigida. Portanto ele sustenta
afirmações sobre estrutura relacional além do grau NÃO DIRIGIDO. Ele não testa
isoladamente direção e pesos sinápticos; isso pode ser auditado separadamente.
"""

from pathlib import Path
import argparse
import json
import math
import os
import random
import sys
import time
import warnings

# Evita oversubscription em BLAS/OpenMP.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import igraph as ig

from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FEATURES = ROOT / "reports" / "f2" / "features"
SPLITS = ROOT / "reports" / "f2" / "splits"
MODELS = ROOT / "reports" / "f2" / "models"

CLASSIFICATION = RAW / "classification.csv.gz"
PROCESSED_EDGES = PROCESSED / "flywire_fafb_v783_edges.csv.gz"
MAIN_SPLIT = SPLITS / "main_split.csv.gz"

LOG_SELECTED = MODELS / "logistic_validation" / "selected_logistic_configs.csv"
RF_SELECTED = MODELS / "random_forest_validation" / "selected_random_forest_configs.csv"

OUT = ROOT / "reports" / "f2" / "audit_relational_null"
OUT.mkdir(parents=True, exist_ok=True)

REAL_FEATURES_OUT = OUT / "real_unweighted_relational_features.csv.gz"
RESULTS_OUT = OUT / "real_vs_degree_preserving_null_results.csv"
NULL_SUMMARY_OUT = OUT / "null_distribution_summary.csv"
DEGREE_CHECK_OUT = OUT / "degree_preservation_checks.csv"
RUN_META_OUT = OUT / "run_metadata.json"
SUMMARY_OUT = OUT / "summary.txt"

SEED_BASE = 20261110
DEFAULT_N_NULLS = 10
DEFAULT_REWIRE_PER_EDGE = 5.0

RELATIONAL_FEATURES = [
    "undirected_degree",
    "local_clustering",
    "coreness",
    "leiden_module_size",
    "leiden_internal_degree_fraction",
    "leiden_within_module_degree_z",
]

DEGREE_ONLY_FEATURES = ["undirected_degree"]

SCOPES = {
    "global": None,
    "intrinsic": "intrinsic",
}


def section(title):
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def require(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo necessário não encontrado:\n{path}"
        )


def build_simple_undirected_graph():
    """
    Constrói projeção simples não dirigida preservando o universo completo de
    root_ids de classification.csv.gz. Isolados permanecem como vértices.
    """
    section("Construindo projeção simples não dirigida")

    cls = pd.read_csv(
        CLASSIFICATION,
        usecols=["root_id", "flow", "super_class"],
        low_memory=False,
    )

    if cls["root_id"].duplicated().any():
        raise RuntimeError("root_id duplicado em classification.csv.gz.")

    # Ordem determinística e mapa root_id -> vertex id.
    cls = cls.sort_values("root_id").reset_index(drop=True)
    node_ids = cls["root_id"].to_numpy(dtype=np.int64)

    edges = pd.read_csv(
        PROCESSED_EDGES,
        usecols=["pre_root_id", "post_root_id"],
        low_memory=False,
    )

    pre = edges["pre_root_id"].to_numpy(dtype=np.int64)
    post = edges["post_root_id"].to_numpy(dtype=np.int64)

    # Mapeamento vetorizado via searchsorted.
    src = np.searchsorted(node_ids, pre)
    dst = np.searchsorted(node_ids, post)

    if (
        (src >= len(node_ids)).any()
        or (dst >= len(node_ids)).any()
        or not np.array_equal(node_ids[src], pre)
        or not np.array_equal(node_ids[dst], post)
    ):
        raise RuntimeError(
            "Há endpoints de arestas que não pertencem ao universo de "
            "classification.csv.gz."
        )

    # Remove self-loops e converte pares dirigidos para pares não dirigidos.
    mask = src != dst
    src = src[mask]
    dst = dst[mask]

    u = np.minimum(src, dst)
    v = np.maximum(src, dst)

    pairs = pd.DataFrame(
        {
            "u": u.astype(np.int32, copy=False),
            "v": v.astype(np.int32, copy=False),
        }
    ).drop_duplicates(ignore_index=True)

    del edges, pre, post, src, dst, u, v

    print(f"Nós:                  {len(node_ids):,}")
    print(f"Arestas não dirigidas:{len(pairs):,}")

    # Passamos iterable para não criar uma lista Python gigantesca extra.
    g = ig.Graph(
        n=len(node_ids),
        edges=zip(
            pairs["u"].to_numpy(),
            pairs["v"].to_numpy(),
        ),
        directed=False,
    )

    if not g.is_simple():
        raise RuntimeError("Projeção deveria ser simples, mas não é.")

    print(f"igraph v{ig.__version__}")
    print(
        f"Isolados:             "
        f"{sum(d == 0 for d in g.degree()):,}"
    )

    return cls, node_ids, g


def leiden_membership(g, seed):
    """
    Leiden não ponderado, modularidade, resolução 1.
    """
    ig.set_random_number_generator(random.Random(seed))

    clustering = g.community_leiden(
        objective_function="modularity",
        weights=None,
        resolution=1.0,
        beta=0.01,
        initial_membership=None,
        n_iterations=-1,
    )

    membership = np.asarray(
        clustering.membership,
        dtype=np.int32,
    )

    return membership, float(clustering.modularity)


def within_module_z(membership, internal_degree):
    """
    z_i = (k_i^in - mean_module) / std_module
    """
    n_comm = int(membership.max()) + 1

    counts = np.bincount(
        membership,
        minlength=n_comm,
    ).astype(np.float64)

    sums = np.bincount(
        membership,
        weights=internal_degree,
        minlength=n_comm,
    ).astype(np.float64)

    sums_sq = np.bincount(
        membership,
        weights=internal_degree.astype(np.float64) ** 2,
        minlength=n_comm,
    ).astype(np.float64)

    means = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums),
        where=counts > 0,
    )

    second = np.divide(
        sums_sq,
        counts,
        out=np.zeros_like(sums_sq),
        where=counts > 0,
    )

    var = np.maximum(
        second - means**2,
        0.0,
    )
    std = np.sqrt(var)

    node_std = std[membership]
    z = np.zeros(len(membership), dtype=np.float64)

    valid = node_std > 0
    z[valid] = (
        internal_degree[valid]
        - means[membership[valid]]
    ) / node_std[valid]

    return z


def compute_relational_features(g, root_ids, seed, graph_label):
    """
    Mesmas features para real e null.

    Grau é explicitamente incluído para que o classificador possa usar a
    informação preservada pelo modelo nulo. A diferença real-null só pode
    aparecer nas features de ordem superior.
    """
    t0 = time.time()

    section(f"Features: {graph_label}")

    degree = np.asarray(
        g.degree(),
        dtype=np.int32,
    )

    print("  local clustering...")
    local_clustering = np.asarray(
        g.transitivity_local_undirected(
            vertices=None,
            mode="zero",
        ),
        dtype=np.float64,
    )

    local_clustering = np.nan_to_num(
        local_clustering,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    print("  coreness...")
    coreness = np.asarray(
        g.coreness(),
        dtype=np.int32,
    )

    print("  Leiden...")
    membership, modularity = leiden_membership(
        g,
        seed=seed,
    )

    n_comm = int(membership.max()) + 1
    module_sizes = np.bincount(
        membership,
        minlength=n_comm,
    ).astype(np.int32)

    module_size_node = module_sizes[membership]

    print(
        f"  comunidades={n_comm:,} | "
        f"Q={modularity:.6f}"
    )

    print("  métricas intramodulares...")
    edge_array = np.asarray(
        g.get_edgelist(),
        dtype=np.int32,
    )

    eu = edge_array[:, 0]
    ev = edge_array[:, 1]

    same = membership[eu] == membership[ev]

    internal_endpoints = np.concatenate(
        [eu[same], ev[same]]
    )

    internal_degree = np.bincount(
        internal_endpoints,
        minlength=g.vcount(),
    ).astype(np.float64)

    internal_fraction = np.divide(
        internal_degree,
        degree,
        out=np.zeros(
            len(degree),
            dtype=np.float64,
        ),
        where=degree > 0,
    )

    z = within_module_z(
        membership,
        internal_degree,
    )

    out = pd.DataFrame(
        {
            "root_id": root_ids,
            "undirected_degree": degree,
            "local_clustering": local_clustering,
            "coreness": coreness,
            "leiden_module_size": module_size_node,
            "leiden_internal_degree_fraction": internal_fraction,
            "leiden_within_module_degree_z": z,
        }
    )

    if out[RELATIONAL_FEATURES].isna().any().any():
        raise RuntimeError(
            f"NaN detectado nas features de {graph_label}."
        )

    metadata = {
        "graph_label": graph_label,
        "n_vertices": int(g.vcount()),
        "n_edges": int(g.ecount()),
        "n_communities": n_comm,
        "modularity": modularity,
        "mean_local_clustering": float(
            local_clustering.mean()
        ),
        "max_coreness": int(coreness.max()),
        "runtime_seconds": time.time() - t0,
    }

    print(
        f"  concluído em "
        f"{metadata['runtime_seconds'] / 60:.2f} min"
    )

    del edge_array, eu, ev, internal_endpoints

    return out, metadata


def load_frozen_hyperparameters():
    """
    Herda configs já escolhidas usando network_full. Não há novo tuning.
    """
    require(LOG_SELECTED)
    require(RF_SELECTED)

    log_df = pd.read_csv(LOG_SELECTED)
    rf_df = pd.read_csv(RF_SELECTED)

    log_row = log_df[
        log_df["experiment"] == "network_full"
    ]

    rf_row = rf_df[
        rf_df["experiment"] == "network_full"
    ]

    if len(log_row) != 1 or len(rf_row) != 1:
        raise RuntimeError(
            "Não foi possível localizar configuração congelada "
            "de network_full."
        )

    log_c = float(log_row.iloc[0]["C"])

    raw_max_features = str(
        rf_row.iloc[0]["max_features"]
    )

    rf_max_features = (
        "sqrt"
        if raw_max_features == "sqrt"
        else float(raw_max_features)
    )

    rf_leaf = int(
        rf_row.iloc[0]["min_samples_leaf"]
    )
    rf_trees = int(
        rf_row.iloc[0]["n_estimators"]
    )

    return {
        "logistic_C": log_c,
        "rf_max_features": rf_max_features,
        "rf_min_samples_leaf": rf_leaf,
        "rf_n_estimators": rf_trees,
    }


def metrics(y_true, y_pred, labels):
    return {
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
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


def prepare_scope(base, feature_df, scope):
    merged = (
        base
        .merge(
            feature_df,
            on="root_id",
            validate="one_to_one",
        )
    )

    if scope == "global":
        scoped = merged
    elif scope == "intrinsic":
        scoped = merged[
            merged["flow"] == "intrinsic"
        ].copy()
    else:
        raise ValueError(scope)

    train = scoped[
        scoped["split"] == "train"
    ].copy()

    val = scoped[
        scoped["split"] == "val"
    ].copy()

    # Não há acesso a TEST.
    labels = sorted(
        set(train["super_class"])
        & set(val["super_class"])
    )

    train = train[
        train["super_class"].isin(labels)
    ].copy()

    val = val[
        val["super_class"].isin(labels)
    ].copy()

    return train, val, labels


def fit_evaluate_models(
    base,
    feature_df,
    scope,
    graph_label,
    hyper,
    feature_block,
):
    train, val, labels = prepare_scope(
        base,
        feature_df,
        scope,
    )

    if feature_block == "degree_only":
        feature_cols = DEGREE_ONLY_FEATURES
    elif feature_block == "relational":
        feature_cols = RELATIONAL_FEATURES
    else:
        raise ValueError(feature_block)

    X_train = train[feature_cols].to_numpy(
        dtype=np.float64
    )
    X_val = val[feature_cols].to_numpy(
        dtype=np.float64
    )

    y_train = train["super_class"].to_numpy()
    y_val = val["super_class"].to_numpy()

    rows = []

    # Logistic: frozen C; standardization fit only on train.
    log_model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=hyper["logistic_C"],
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            ConvergenceWarning,
        )
        log_model.fit(X_train, y_train)

    pred = log_model.predict(X_val)

    rows.append(
        {
            "graph": graph_label,
            "scope": scope,
            "model_family": "logistic",
            "feature_block": feature_block,
            "train_rows": len(train),
            "val_rows": len(val),
            "n_classes": len(labels),
            "classes": ";".join(labels),
            **metrics(y_val, pred, labels),
        }
    )

    rf_model = RandomForestClassifier(
        n_estimators=hyper["rf_n_estimators"],
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=hyper[
            "rf_min_samples_leaf"
        ],
        max_features=hyper[
            "rf_max_features"
        ],
        bootstrap=True,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    rf_model.fit(X_train, y_train)
    pred = rf_model.predict(X_val)

    rows.append(
        {
            "graph": graph_label,
            "scope": scope,
            "model_family": "random_forest",
            "feature_block": feature_block,
            "train_rows": len(train),
            "val_rows": len(val),
            "n_classes": len(labels),
            "classes": ";".join(labels),
            **metrics(y_val, pred, labels),
        }
    )

    return rows


def summarize_nulls(results):
    """
    Resume real vs distribuição dos nulls.
    Sem p-valor inferencial forte: com apenas 10 nulls, o menor p empírico
    one-sided seria 1/11 ~= 0.091. O foco aqui é efeito/deslocamento.
    """
    rows = []

    keys = [
        "scope",
        "model_family",
        "feature_block",
    ]

    for key, grp in results.groupby(keys):
        scope, family, block = key

        real = grp[
            grp["graph"] == "real"
        ]

        null = grp[
            grp["graph"].str.startswith("null_")
        ]

        if len(real) != 1 or len(null) == 0:
            continue

        real_f1 = float(
            real.iloc[0]["macro_f1"]
        )

        null_vals = null[
            "macro_f1"
        ].to_numpy(dtype=float)

        mean = float(null_vals.mean())
        std = float(
            null_vals.std(ddof=1)
            if len(null_vals) > 1
            else 0.0
        )

        z = (
            (real_f1 - mean) / std
            if std > 0
            else math.inf
        )

        empirical_tail_fraction = float(
            np.mean(null_vals >= real_f1)
        )

        rows.append(
            {
                "scope": scope,
                "model_family": family,
                "feature_block": block,
                "real_macro_f1": real_f1,
                "null_n": len(null_vals),
                "null_mean_macro_f1": mean,
                "null_std_macro_f1": std,
                "null_min_macro_f1": float(
                    null_vals.min()
                ),
                "null_max_macro_f1": float(
                    null_vals.max()
                ),
                "real_minus_null_mean": (
                    real_f1 - mean
                ),
                "z_vs_null_descriptive": z,
                "fraction_null_ge_real": (
                    empirical_tail_fraction
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-nulls",
        type=int,
        default=DEFAULT_N_NULLS,
    )
    parser.add_argument(
        "--rewire-per-edge",
        type=float,
        default=DEFAULT_REWIRE_PER_EDGE,
        help=(
            "Número de tentativas de rewiring por aresta. "
            "Default=5."
        ),
    )
    parser.add_argument(
        "--save-null-features",
        action="store_true",
        help=(
            "Salva CSV de features de cada nulo. "
            "Não é necessário para a análise padrão."
        ),
    )
    args = parser.parse_args()

    if args.n_nulls < 1:
        raise ValueError("--n-nulls deve ser >= 1.")
    if args.rewire_per_edge <= 0:
        raise ValueError("--rewire-per-edge deve ser > 0.")

    t_global = time.time()

    section("F2.15 — REAL vs DEGREE-PRESERVING NULL")
    print("TEST FINAL: BLOQUEADO / NÃO ACESSADO")
    print(f"N nulls: {args.n_nulls}")
    print(
        f"Rewire trials por aresta: "
        f"{args.rewire_per_edge}"
    )

    for p in [
        CLASSIFICATION,
        PROCESSED_EDGES,
        MAIN_SPLIT,
        LOG_SELECTED,
        RF_SELECTED,
    ]:
        require(p)

    cls, root_ids, g_real = (
        build_simple_undirected_graph()
    )

    split = pd.read_csv(
        MAIN_SPLIT,
        usecols=["root_id", "split"],
    )

    base = cls.merge(
        split,
        on="root_id",
        validate="one_to_one",
    )

    test_rows = int(
        (base["split"] == "test").sum()
    )
    print(f"Test rows presentes no arquivo: {test_rows:,}")
    print(
        "Eles são carregados apenas para validar o split, "
        "mas NUNCA entram em fit/predict."
    )

    hyper = load_frozen_hyperparameters()

    print()
    print("Hiperparâmetros congelados:")
    print(json.dumps(
        hyper,
        indent=2,
        ensure_ascii=False,
    ))

    # ------------------------------------------------------------------
    # REAL
    # ------------------------------------------------------------------
    real_features, real_meta = (
        compute_relational_features(
            g_real,
            root_ids,
            seed=SEED_BASE,
            graph_label="real",
        )
    )

    real_features.to_csv(
        REAL_FEATURES_OUT,
        index=False,
        compression="gzip",
    )

    real_degree = real_features[
        "undirected_degree"
    ].to_numpy(dtype=np.int32)

    results_rows = []

    # degree_only precisa ser avaliado apenas uma vez; é idêntico em todos os
    # nulls por construção.
    for scope in SCOPES:
        results_rows.extend(
            fit_evaluate_models(
                base,
                real_features,
                scope,
                "real",
                hyper,
                "degree_only",
            )
        )

        results_rows.extend(
            fit_evaluate_models(
                base,
                real_features,
                scope,
                "real",
                hyper,
                "relational",
            )
        )

    degree_checks = [
        {
            "graph": "real",
            "degree_identical_to_real": True,
            "n_vertices": g_real.vcount(),
            "n_edges": g_real.ecount(),
            "degree_sum": int(real_degree.sum()),
            "degree_max": int(real_degree.max()),
        }
    ]

    graph_meta = [real_meta]

    # ------------------------------------------------------------------
    # NULLS
    # ------------------------------------------------------------------
    n_rewire_trials = int(
        round(
            args.rewire_per_edge
            * g_real.ecount()
        )
    )

    for i in range(1, args.n_nulls + 1):
        graph_label = f"null_{i:02d}"
        seed = SEED_BASE + i

        section(
            f"{graph_label}: rewiring "
            f"({n_rewire_trials:,} tentativas)"
        )

        # Copiar real a cada iteração evita que null_i seja derivado do null_{i-1}.
        g_null = g_real.copy()

        ig.set_random_number_generator(
            random.Random(seed)
        )

        t_rewire = time.time()

        g_null.rewire(
            n=n_rewire_trials,
            mode="simple",
        )

        print(
            f"Rewiring: "
            f"{(time.time() - t_rewire) / 60:.2f} min"
        )

        null_degree = np.asarray(
            g_null.degree(),
            dtype=np.int32,
        )

        degree_equal = np.array_equal(
            null_degree,
            real_degree,
        )

        if not degree_equal:
            bad = int(
                np.sum(
                    null_degree != real_degree
                )
            )
            raise RuntimeError(
                f"{graph_label}: degree sequence mudou "
                f"em {bad} nós."
            )

        if g_null.ecount() != g_real.ecount():
            raise RuntimeError(
                f"{graph_label}: número de arestas mudou."
            )

        degree_checks.append(
            {
                "graph": graph_label,
                "degree_identical_to_real": True,
                "n_vertices": g_null.vcount(),
                "n_edges": g_null.ecount(),
                "degree_sum": int(
                    null_degree.sum()
                ),
                "degree_max": int(
                    null_degree.max()
                ),
            }
        )

        print(
            "Degree preservation: PASS "
            "(idêntico nó a nó)"
        )

        null_features, null_meta = (
            compute_relational_features(
                g_null,
                root_ids,
                seed=seed + 100000,
                graph_label=graph_label,
            )
        )

        graph_meta.append(null_meta)

        if not np.array_equal(
            null_features[
                "undirected_degree"
            ].to_numpy(dtype=np.int32),
            real_degree,
        ):
            raise RuntimeError(
                f"{graph_label}: feature de grau "
                "não preservada."
            )

        if args.save_null_features:
            null_features.to_csv(
                OUT
                / f"{graph_label}_unweighted_relational_features.csv.gz",
                index=False,
                compression="gzip",
            )

        # Não repetimos degree_only porque é matematicamente idêntico.
        for scope in SCOPES:
            results_rows.extend(
                fit_evaluate_models(
                    base,
                    null_features,
                    scope,
                    graph_label,
                    hyper,
                    "relational",
                )
            )

        del g_null, null_features

    results = pd.DataFrame(results_rows)

    # Para facilitar comparação, replicamos o degree-only real como baseline
    # separado, mas não fingimos que é uma distribuição null.
    results.to_csv(
        RESULTS_OUT,
        index=False,
    )

    degree_check_df = pd.DataFrame(
        degree_checks
    )
    degree_check_df.to_csv(
        DEGREE_CHECK_OUT,
        index=False,
    )

    summary = summarize_nulls(
        results[
            results["feature_block"]
            == "relational"
        ].copy()
    )

    summary.to_csv(
        NULL_SUMMARY_OUT,
        index=False,
    )

    metadata = {
        "experiment": (
            "real_vs_degree_preserving_rewired_"
            "unweighted_projection"
        ),
        "test_accessed_for_fit_or_prediction": False,
        "n_nulls": args.n_nulls,
        "rewire_trials_per_edge": (
            args.rewire_per_edge
        ),
        "rewire_trials_per_null": (
            n_rewire_trials
        ),
        "seed_base": SEED_BASE,
        "real_graph": real_meta,
        "graph_metadata": graph_meta,
        "relational_features": (
            RELATIONAL_FEATURES
        ),
        "degree_only_features": (
            DEGREE_ONLY_FEATURES
        ),
        "scopes": list(SCOPES.keys()),
        "frozen_hyperparameters": hyper,
        "interpretation_boundary": (
            "This audit preserves the exact undirected degree of every node "
            "and destroys partner identity / higher-order organization. "
            "It does not independently preserve/test directed in/out degree "
            "or weighted node strengths."
        ),
        "runtime_seconds": (
            time.time() - t_global
        ),
    }

    with RUN_META_OUT.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    section("RESULTADO PRINCIPAL")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # Pretty summary
    lines = [
        "F2.15 — Real vs degree-preserving null",
        "",
        "TEST FINAL NÃO ACESSADO.",
        "",
        "NULL MODEL",
        (
            "Projeção simples não dirigida; degree-preserving edge swaps; "
            "grau de cada nó preservado exatamente."
        ),
        "",
        "FEATURES DO CONTROLE",
        ", ".join(RELATIONAL_FEATURES),
        "",
        "RESULTADOS",
    ]

    for _, r in summary.iterrows():
        lines.append(
            f"{r['scope']} | {r['model_family']} | "
            f"real={r['real_macro_f1']:.6f} | "
            f"null={r['null_mean_macro_f1']:.6f}"
            f"±{r['null_std_macro_f1']:.6f} | "
            f"delta={r['real_minus_null_mean']:+.6f} | "
            f"z_desc={r['z_vs_null_descriptive']:.3f}"
        )

    lines.extend(
        [
            "",
            "INTERPRETAÇÃO",
            (
                "Se real >> null, a sequência de graus não explica sozinha "
                "a informação preditiva. A identidade dos parceiros e a "
                "organização de ordem superior/mesoscópica carregam sinal."
            ),
            (
                "Este controle é não dirigido e não ponderado; direção e "
                "pesos requerem auditoria separada para sustentar afirmações "
                "específicas sobre esses componentes."
            ),
            "",
            f"Resultados: {RESULTS_OUT.relative_to(ROOT)}",
            f"Resumo null: {NULL_SUMMARY_OUT.relative_to(ROOT)}",
            f"Degree checks: {DEGREE_CHECK_OUT.relative_to(ROOT)}",
            f"Metadata: {RUN_META_OUT.relative_to(ROOT)}",
            (
                f"Runtime total: "
                f"{(time.time() - t_global) / 60:.2f} min"
            ),
        ]
    )

    SUMMARY_OUT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print(
        "Degree preservation checks: "
        + (
            "PASS"
            if degree_check_df[
                "degree_identical_to_real"
            ].all()
            else "FAIL"
        )
    )

    print()
    print(f"Resultados: {RESULTS_OUT}")
    print(f"Resumo:     {NULL_SUMMARY_OUT}")
    print(f"Checks:     {DEGREE_CHECK_OUT}")
    print(f"Metadata:   {RUN_META_OUT}")
    print(f"TXT:        {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
