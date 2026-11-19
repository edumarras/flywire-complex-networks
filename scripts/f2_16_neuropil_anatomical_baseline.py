"""
F2.16 — Baseline anatômico por neuropil

Pergunta
--------
Quanto de super_class é recuperável quando fornecemos explicitamente a localização
anatômica das sinapses, e a topologia ainda acrescenta informação depois disso?

Este é DELIBERADAMENTE um baseline "quase definicional":
- usa a distribuição de INPUTS por neuropil;
- usa a distribuição de OUTPUTS por neuropil;
- NÃO usa super_class, flow, class, nerve, tract etc. como features;
- NÃO usa o conjunto de teste final.

Experimentos
------------
1. network_full
2. neuropil_only
3. network_full + neuropil

Escopos
-------
- global: todas as superclasses
- intrinsic: apenas flow == intrinsic

Modelos
-------
- Logistic Regression
- Random Forest

Os hiperparâmetros são os já congelados para network_full. Não há novo tuning.

Interpretação principal
-----------------------
Se:
    neuropil_only >> network_full
então a localização anatômica explícita é mais próxima da definição do target.

Se:
    network_full + neuropil > neuropil_only
então a topologia contém informação incremental mesmo após fornecer explicitamente
a distribuição espacial das sinapses.

Uso analítico
------------
Este baseline é diagnóstico. As features de neuropil são analisadas separadamente
do modelo principal por serem próximas dos critérios anatômicos usados na definição
de várias superclasses.
"""

from pathlib import Path
import argparse
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
from scipy import sparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler, normalize


ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
FEATURES = ROOT / "reports" / "f2" / "features"
SPLITS = ROOT / "reports" / "f2" / "splits"
MODELS = ROOT / "reports" / "f2" / "models"

CONNECTIONS = RAW / "connections_princeton.csv.gz"
CLASSIFICATION = RAW / "classification.csv.gz"

TOPO_FEATURES = FEATURES / "node_topological_features.csv.gz"
TOPO_SETS = FEATURES / "topological_feature_sets.json"
TOPO_MANIFEST = FEATURES / "topological_feature_manifest.csv"

MAIN_SPLIT = SPLITS / "main_split.csv.gz"

LOG_SELECTED = MODELS / "logistic_validation" / "selected_logistic_configs.csv"
RF_SELECTED = MODELS / "random_forest_validation" / "selected_random_forest_configs.csv"

OUT = ROOT / "reports" / "f2" / "audit_neuropil_baseline"
OUT.mkdir(parents=True, exist_ok=True)

NEUROPIL_CACHE = OUT / "neuropil_fraction_features.npz"
NEUROPIL_MANIFEST = OUT / "neuropil_feature_manifest.csv"
NEUROPIL_META = OUT / "neuropil_feature_metadata.json"

RESULTS_OUT = OUT / "neuropil_baseline_results.csv"
SUMMARY_OUT = OUT / "neuropil_baseline_summary.csv"
SUMMARY_TXT = OUT / "summary.txt"
RUN_META = OUT / "run_metadata.json"

CHUNK_SIZE = 500_000
RANDOM_STATE = 42

SCOPES = {
    "global": None,
    "intrinsic": "intrinsic",
}

EXPERIMENTS = [
    "network_full",
    "neuropil_only",
    "network_full_plus_neuropil",
]


def section(title):
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def require(path):
    if not path.exists():
        raise FileNotFoundError(f"Arquivo necessário não encontrado:\n{path}")


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_name(x):
    s = str(x)
    for ch in [" ", "/", "\\", ":", ";", ",", "(", ")", "[", "]", "{", "}"]:
        s = s.replace(ch, "_")
    return s


def discover_neuropils():
    section("Descobrindo neuropils")

    values = set()

    for chunk in pd.read_csv(
        CONNECTIONS,
        usecols=["neuropil"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        vals = chunk["neuropil"].fillna("__MISSING__").astype(str).unique()
        values.update(vals.tolist())

    neuropils = sorted(values)

    print(f"Neuropils distintos: {len(neuropils)}")
    print(", ".join(neuropils[:20]) + (" ..." if len(neuropils) > 20 else ""))

    return neuropils


def build_neuropil_matrix(node_ids, neuropils):
    """
    Produz CSR com:
        [OUT fractions por neuropil | IN fractions por neuropil]

    Cada bloco é normalizado por linha para soma 1 quando o neurônio possui
    sinapses naquela direção. Portanto o baseline contém distribuição espacial,
    não força total.
    """
    section("Construindo features anatômicas por neuropil")

    n_nodes = len(node_ids)
    n_neuropils = len(neuropils)

    neuropil_to_col = {
        n: i for i, n in enumerate(neuropils)
    }

    out_acc = sparse.csr_matrix(
        (n_nodes, n_neuropils),
        dtype=np.float32,
    )
    in_acc = sparse.csr_matrix(
        (n_nodes, n_neuropils),
        dtype=np.float32,
    )

    n_rows = 0
    total_syn_count = 0.0
    t0 = time.time()

    for idx, chunk in enumerate(
        pd.read_csv(
            CONNECTIONS,
            usecols=[
                "pre_root_id",
                "post_root_id",
                "neuropil",
                "syn_count",
            ],
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ),
        start=1,
    ):
        n_rows += len(chunk)

        neuropil_series = (
            chunk["neuropil"]
            .fillna("__MISSING__")
            .astype(str)
        )

        cols = neuropil_series.map(
            neuropil_to_col
        ).to_numpy(dtype=np.int32)

        pre = chunk["pre_root_id"].to_numpy(
            dtype=np.int64
        )
        post = chunk["post_root_id"].to_numpy(
            dtype=np.int64
        )
        weights = chunk["syn_count"].to_numpy(
            dtype=np.float32
        )

        total_syn_count += float(weights.sum())

        pre_idx = np.searchsorted(node_ids, pre)
        post_idx = np.searchsorted(node_ids, post)

        if (
            (pre_idx >= n_nodes).any()
            or (post_idx >= n_nodes).any()
            or not np.array_equal(node_ids[pre_idx], pre)
            or not np.array_equal(node_ids[post_idx], post)
        ):
            raise RuntimeError(
                "Endpoint de conexão fora do universo de classification."
            )

        out_chunk = sparse.coo_matrix(
            (
                weights,
                (pre_idx, cols),
            ),
            shape=(n_nodes, n_neuropils),
            dtype=np.float32,
        ).tocsr()

        in_chunk = sparse.coo_matrix(
            (
                weights,
                (post_idx, cols),
            ),
            shape=(n_nodes, n_neuropils),
            dtype=np.float32,
        ).tocsr()

        out_chunk.sum_duplicates()
        in_chunk.sum_duplicates()

        out_acc = out_acc + out_chunk
        in_acc = in_acc + in_chunk

        if idx % 2 == 0:
            print(
                f"  chunks={idx:02d} | "
                f"rows={n_rows:,} | "
                f"tempo={(time.time() - t0) / 60:.2f} min"
            )

    out_acc.sum_duplicates()
    in_acc.sum_duplicates()

    # Converte syn_count absoluto em distribuição anatômica relativa.
    out_frac = normalize(
        out_acc,
        norm="l1",
        axis=1,
        copy=True,
    ).astype(np.float32)

    in_frac = normalize(
        in_acc,
        norm="l1",
        axis=1,
        copy=True,
    ).astype(np.float32)

    X = sparse.hstack(
        [out_frac, in_frac],
        format="csr",
        dtype=np.float32,
    )

    feature_names = (
        [f"out_frac__{sanitize_name(n)}" for n in neuropils]
        + [f"in_frac__{sanitize_name(n)}" for n in neuropils]
    )

    zero_out = int(
        np.sum(
            np.asarray(out_acc.sum(axis=1)).ravel() == 0
        )
    )
    zero_in = int(
        np.sum(
            np.asarray(in_acc.sum(axis=1)).ravel() == 0
        )
    )

    meta = {
        "connection_rows": int(n_rows),
        "total_syn_count_from_rows": float(total_syn_count),
        "n_nodes": int(n_nodes),
        "n_neuropils": int(n_neuropils),
        "n_features": int(X.shape[1]),
        "matrix_nnz": int(X.nnz),
        "zero_out_nodes": zero_out,
        "zero_in_nodes": zero_in,
        "feature_definition": (
            "outgoing and incoming syn_count fractions by neuropil, "
            "L1-normalized separately by direction"
        ),
        "includes_total_strength": False,
        "includes_super_class": False,
        "includes_flow_as_feature": False,
        "runtime_seconds": time.time() - t0,
    }

    return X, feature_names, meta


def load_or_build_neuropil(node_ids, force_rebuild=False):
    if (
        NEUROPIL_CACHE.exists()
        and NEUROPIL_MANIFEST.exists()
        and NEUROPIL_META.exists()
        and not force_rebuild
    ):
        section("Carregando cache de neuropil")
        X = sparse.load_npz(NEUROPIL_CACHE).tocsr()
        manifest = pd.read_csv(NEUROPIL_MANIFEST)
        with NEUROPIL_META.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        if X.shape[0] != len(node_ids):
            raise RuntimeError(
                "Cache de neuropil incompatível com o universo atual de nós."
            )

        return X, manifest["feature"].tolist(), meta

    neuropils = discover_neuropils()

    X, feature_names, meta = build_neuropil_matrix(
        node_ids,
        neuropils,
    )

    sparse.save_npz(
        NEUROPIL_CACHE,
        X,
        compressed=True,
    )

    manifest = pd.DataFrame(
        {
            "feature": feature_names,
            "block": [
                "neuropil_out_fraction"
            ] * len(neuropils)
            + [
                "neuropil_in_fraction"
            ] * len(neuropils),
            "source": "connections_princeton.csv.gz",
            "uses_syn_count": True,
            "uses_neuropil": True,
            "diagnostic_only": True,
        }
    )
    manifest.to_csv(
        NEUROPIL_MANIFEST,
        index=False,
    )

    with NEUROPIL_META.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            meta,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return X, feature_names, meta


def get_frozen_hyperparameters():
    """
    Usa exatamente as configs já selecionadas para network_full.
    Não otimiza parâmetros para neuropil.
    """
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
            "Configuração congelada de network_full não encontrada."
        )

    log_c = float(log_row.iloc[0]["C"])

    raw_mf = str(
        rf_row.iloc[0]["max_features"]
    )
    rf_mf = (
        "sqrt"
        if raw_mf == "sqrt"
        else float(raw_mf)
    )

    return {
        "logistic_C": log_c,
        "rf_n_estimators": int(
            rf_row.iloc[0]["n_estimators"]
        ),
        "rf_max_features": rf_mf,
        "rf_min_samples_leaf": int(
            rf_row.iloc[0]["min_samples_leaf"]
        ),
    }


def prepare_network_matrix(node_ids):
    """
    Carrega network_full na MESMA ordem de node_ids.
    """
    topo_sets = load_json(TOPO_SETS)
    network_cols = list(
        topo_sets["network_full"]
    )

    topo = pd.read_csv(
        TOPO_FEATURES,
        usecols=["root_id"] + network_cols,
        low_memory=False,
    )

    topo = (
        topo
        .set_index("root_id")
        .loc[node_ids]
    )

    if topo.isna().any().any():
        raise RuntimeError(
            "NaN em network_full."
        )

    X = sparse.csr_matrix(
        topo[network_cols].to_numpy(
            dtype=np.float32
        )
    )

    return X, network_cols


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


def build_experiment_matrix(
    experiment,
    X_network,
    X_neuropil,
):
    if experiment == "network_full":
        return X_network
    if experiment == "neuropil_only":
        return X_neuropil
    if experiment == "network_full_plus_neuropil":
        return sparse.hstack(
            [X_network, X_neuropil],
            format="csr",
            dtype=np.float32,
        )
    raise ValueError(experiment)


def evaluate_scope(
    base,
    X_network,
    X_neuropil,
    scope,
    hyper,
):
    if scope == "global":
        scope_mask = np.ones(
            len(base),
            dtype=bool,
        )
    elif scope == "intrinsic":
        scope_mask = (
            base["flow"].to_numpy()
            == "intrinsic"
        )
    else:
        raise ValueError(scope)

    split_arr = base["split"].to_numpy()
    y_all = base["super_class"].to_numpy()

    train_mask = (
        scope_mask
        & (split_arr == "train")
    )
    val_mask = (
        scope_mask
        & (split_arr == "val")
    )

    labels = sorted(
        set(y_all[train_mask])
        & set(y_all[val_mask])
    )

    train_mask &= np.isin(
        y_all,
        labels,
    )
    val_mask &= np.isin(
        y_all,
        labels,
    )

    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(val_mask)

    y_train = y_all[train_idx]
    y_val = y_all[val_idx]

    print(
        f"{scope}: train={len(train_idx):,} | "
        f"val={len(val_idx):,} | "
        f"classes={len(labels)}"
    )

    rows = []

    for experiment in EXPERIMENTS:
        section(
            f"{scope} | {experiment}"
        )

        X = build_experiment_matrix(
            experiment,
            X_network,
            X_neuropil,
        )

        X_train = X[train_idx]
        X_val = X[val_idx]

        print(
            f"Features={X.shape[1]:,} | "
            f"nnz={X.nnz:,}"
        )

        # Logistic:
        # with_mean=False mantém sparse. O mesmo preprocessing é usado em todos
        # os blocos desta auditoria; compare os experimentos DENTRO desta auditoria.
        log_model = PipelineSparseLogistic(
            C=hyper["logistic_C"],
        )

        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore",
                ConvergenceWarning,
            )
            log_model.fit(
                X_train,
                y_train,
            )

        pred = log_model.predict(X_val)
        m = metrics(
            y_val,
            pred,
            labels,
        )

        rows.append(
            {
                "scope": scope,
                "model_family": "logistic",
                "experiment": experiment,
                "train_rows": len(train_idx),
                "val_rows": len(val_idx),
                "n_classes": len(labels),
                "classes": ";".join(labels),
                "n_features": X.shape[1],
                **m,
            }
        )

        print(
            f"[LOG] macro-F1={m['macro_f1']:.6f}"
        )

        rf_model = RandomForestClassifier(
            n_estimators=hyper[
                "rf_n_estimators"
            ],
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
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        rf_model.fit(
            X_train,
            y_train,
        )

        pred = rf_model.predict(X_val)
        m = metrics(
            y_val,
            pred,
            labels,
        )

        rows.append(
            {
                "scope": scope,
                "model_family": "random_forest",
                "experiment": experiment,
                "train_rows": len(train_idx),
                "val_rows": len(val_idx),
                "n_classes": len(labels),
                "classes": ";".join(labels),
                "n_features": X.shape[1],
                **m,
            }
        )

        print(
            f"[RF ] macro-F1={m['macro_f1']:.6f}"
        )

    return rows


class PipelineSparseLogistic:
    """
    Wrapper simples para:
        StandardScaler(with_mean=False)
        LogisticRegression

    Evita densificar a matriz de neuropils.
    """
    def __init__(self, C):
        self.C = C
        self.scaler = StandardScaler(
            with_mean=False,
        )
        self.model = LogisticRegression(
            C=C,
            solver="lbfgs",
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        )

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        return self

    def predict(self, X):
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs)


def summarize(results):
    rows = []

    for (
        scope,
        family,
    ), grp in results.groupby(
        ["scope", "model_family"]
    ):
        score = {
            r["experiment"]: r["macro_f1"]
            for _, r in grp.iterrows()
        }

        required = set(EXPERIMENTS)
        if not required.issubset(score):
            continue

        rows.append(
            {
                "scope": scope,
                "model_family": family,
                "network_macro_f1": score[
                    "network_full"
                ],
                "neuropil_macro_f1": score[
                    "neuropil_only"
                ],
                "network_plus_neuropil_macro_f1": score[
                    "network_full_plus_neuropil"
                ],
                "neuropil_minus_network": (
                    score["neuropil_only"]
                    - score["network_full"]
                ),
                "network_increment_over_neuropil": (
                    score[
                        "network_full_plus_neuropil"
                    ]
                    - score["neuropil_only"]
                ),
                "neuropil_increment_over_network": (
                    score[
                        "network_full_plus_neuropil"
                    ]
                    - score["network_full"]
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force-rebuild-neuropil",
        action="store_true",
    )
    args = parser.parse_args()

    t0 = time.time()

    section("F2.16 — BASELINE ANATÔMICO POR NEUROPIL")
    print("Escopo: train/validation do split principal.")
    print(
        "Este baseline é deliberadamente próximo da anatomia "
        "usada na definição de super_class."
    )

    for p in [
        CONNECTIONS,
        CLASSIFICATION,
        TOPO_FEATURES,
        TOPO_SETS,
        MAIN_SPLIT,
        LOG_SELECTED,
        RF_SELECTED,
    ]:
        require(p)

    cls = pd.read_csv(
        CLASSIFICATION,
        usecols=[
            "root_id",
            "flow",
            "super_class",
        ],
        low_memory=False,
    )

    if cls["root_id"].duplicated().any():
        raise RuntimeError(
            "root_id duplicado em classification."
        )

    # Ordem canônica para todas as matrizes.
    cls = cls.sort_values(
        "root_id"
    ).reset_index(drop=True)

    node_ids = cls[
        "root_id"
    ].to_numpy(dtype=np.int64)

    split = pd.read_csv(
        MAIN_SPLIT,
        usecols=[
            "root_id",
            "split",
        ],
    )

    split = (
        split
        .set_index("root_id")
        .loc[node_ids]
        .reset_index()
    )

    base = cls.copy()
    base["split"] = split["split"].to_numpy()

    n_test = int(
        (base["split"] == "test").sum()
    )
    print(
        f"Test rows no split: {n_test:,} "
        "(não usados em fit/predict)"
    )

    X_neuropil, neuropil_names, neuropil_meta = (
        load_or_build_neuropil(
            node_ids,
            force_rebuild=args.force_rebuild_neuropil,
        )
    )

    X_network, network_names = (
        prepare_network_matrix(
            node_ids
        )
    )

    if not (
        X_neuropil.shape[0]
        == X_network.shape[0]
        == len(base)
    ):
        raise RuntimeError(
            "Número de linhas incompatível."
        )

    hyper = get_frozen_hyperparameters()

    section("Hiperparâmetros congelados")
    print(json.dumps(
        hyper,
        indent=2,
        ensure_ascii=False,
    ))

    rows = []

    for scope in SCOPES:
        rows.extend(
            evaluate_scope(
                base,
                X_network,
                X_neuropil,
                scope,
                hyper,
            )
        )

    results = pd.DataFrame(rows)
    results.to_csv(
        RESULTS_OUT,
        index=False,
    )

    summary = summarize(results)
    summary.to_csv(
        SUMMARY_OUT,
        index=False,
    )

    section("RESULTADO PRINCIPAL")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    run_meta = {
        "experiment": "neuropil_anatomical_baseline",
        "test_accessed_for_fit_or_prediction": False,
        "neuropil_feature_definition": neuropil_meta,
        "n_network_features": len(network_names),
        "n_neuropil_features": len(neuropil_names),
        "experiments": EXPERIMENTS,
        "scopes": list(SCOPES.keys()),
        "frozen_hyperparameters": hyper,
        "interpretation": {
            "neuropil_only": (
                "near-definitional anatomical baseline: explicit spatial "
                "distribution of incoming/outgoing synaptic weight"
            ),
            "network_plus_neuropil": (
                "tests whether topology adds predictive information after "
                "explicit synapse-location information is supplied"
            ),
        },
        "runtime_seconds": time.time() - t0,
    }

    with RUN_META.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            run_meta,
            f,
            indent=2,
            ensure_ascii=False,
        )

    lines = [
        "F2.16 — Baseline anatômico por neuropil",
        "",
        "Escopo: train/validation do split principal.",
        "",
        (
            "Features neuropil = fração do syn_count de INPUT e OUTPUT "
            "em cada neuropil, normalizada separadamente por direção."
        ),
        (
            "Força total NÃO entra neste baseline; apenas distribuição espacial."
        ),
        "",
        "RESULTADOS",
    ]

    for _, r in summary.iterrows():
        lines.append(
            f"{r['scope']} | {r['model_family']} | "
            f"REDE={r['network_macro_f1']:.6f} | "
            f"NEUROPIL={r['neuropil_macro_f1']:.6f} | "
            f"REDE+NEUROPIL={r['network_plus_neuropil_macro_f1']:.6f} | "
            f"ganho_REDE_sobre_NEUROPIL="
            f"{r['network_increment_over_neuropil']:+.6f}"
        )

    lines.extend(
        [
            "",
            "LEITURA",
            (
                "Se NEUROPIL >> REDE, a anatomia espacial explícita está mais "
                "próxima da definição do target, como esperado."
            ),
            (
                "Se REDE+NEUROPIL > NEUROPIL, a topologia contém informação "
                "incremental mesmo controlando localização anatômica explícita."
            ),
            "",
            f"Resultados: {RESULTS_OUT.relative_to(ROOT)}",
            f"Resumo: {SUMMARY_OUT.relative_to(ROOT)}",
            f"Manifesto neuropil: {NEUROPIL_MANIFEST.relative_to(ROOT)}",
            f"Metadata: {RUN_META.relative_to(ROOT)}",
            f"Runtime total: {(time.time() - t0)/60:.2f} min",
        ]
    )

    SUMMARY_TXT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print(f"Resultados: {RESULTS_OUT}")
    print(f"Resumo:     {SUMMARY_OUT}")
    print(f"Manifesto:  {NEUROPIL_MANIFEST}")
    print(f"Metadata:   {RUN_META}")
    print(f"TXT:        {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
