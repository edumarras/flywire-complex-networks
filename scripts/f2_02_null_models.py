from pathlib import Path
import argparse
import gc
import json
import random
import time

import igraph as ig
import numpy as np
import pandas as pd


# ============================================================
# Caminhos
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed"
F2_TABLE_DIR = ROOT / "reports" / "f2" / "tables"
REPORT_DIR = ROOT / "reports" / "f2"
NULL_DIR = REPORT_DIR / "null_models"

EDGES_PATH = PROCESSED_DIR / "flywire_fafb_v783_edges.csv.gz"
NODE_FEATURES_PATH = F2_TABLE_DIR / "node_structural_features.csv.gz"

REALIZATIONS_PATH = NULL_DIR / "null_model_realizations.csv"
SUMMARY_CSV_PATH = NULL_DIR / "null_model_summary.csv"
SUMMARY_JSON_PATH = NULL_DIR / "null_model_summary.json"
SUMMARY_TXT_PATH = NULL_DIR / "null_model_summary.txt"

NULL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Funções auxiliares
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Modelo nulo degree-preserving para a projeção simples não dirigida "
            "do conectoma FlyWire."
        )
    )

    parser.add_argument(
        "--n-null",
        type=int,
        default=10,
        help="Número de realizações nulas. Default: 10.",
    )
    parser.add_argument(
        "--swap-factor",
        type=float,
        default=10.0,
        help=(
            "Número de tentativas de rewiring por aresta. "
            "Default: 10.0."
        ),
    )
    parser.add_argument(
        "--distance-sources",
        type=int,
        default=150,
        help=(
            "Número de origens usadas para estimar distância média "
            "na maior componente. Default: 150."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed base. Default: 42.",
    )

    args = parser.parse_args()

    if args.n_null < 1:
        parser.error("--n-null deve ser >= 1.")
    if args.swap_factor <= 0:
        parser.error("--swap-factor deve ser > 0.")
    if args.distance_sources < 1:
        parser.error("--distance-sources deve ser >= 1.")

    return args


def section(title):
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


def fmt_int(x):
    return f"{int(x):,}".replace(",", ".")


def fmt_float(x, digits=6):
    return f"{float(x):.{digits}f}".replace(".", ",")


def elapsed(start):
    return f"{time.time() - start:.2f} s"


def largest_component_info(graph):
    components = graph.connected_components()
    sizes = np.asarray(components.sizes(), dtype=np.int64)

    largest_idx = int(np.argmax(sizes))
    membership = np.asarray(components.membership, dtype=np.int64)
    vertex_ids = np.flatnonzero(membership == largest_idx)

    giant = graph.induced_subgraph(vertex_ids)

    # Guarda o índice original do vértice para permitir usar as mesmas
    # origens entre o grafo real e os nulos quando possível.
    giant.vs["original_vertex_id"] = vertex_ids.tolist()

    return giant, int(sizes[largest_idx]), int(len(sizes))


def estimate_mean_distance(
    graph,
    n_sources,
    seed,
    preferred_original_sources=None,
):
    """
    Estima a distância média em um grafo conexo não dirigido.

    Se preferred_original_sources for fornecido, tenta usar as mesmas origens
    (IDs de vértices do grafo original) nas diferentes realizações.
    """
    if graph.vcount() <= 1:
        raise RuntimeError("A maior componente possui menos de dois vértices.")

    original_ids = np.asarray(
        graph.vs["original_vertex_id"],
        dtype=np.int64,
    )
    original_to_local = {
        int(orig): i
        for i, orig in enumerate(original_ids)
    }

    rng = np.random.default_rng(seed)

    if preferred_original_sources is None:
        k = min(n_sources, graph.vcount())
        local_sources = rng.choice(
            graph.vcount(),
            size=k,
            replace=False,
        ).astype(np.int64)
        used_original_sources = original_ids[local_sources]
    else:
        available = [
            int(orig)
            for orig in preferred_original_sources
            if int(orig) in original_to_local
        ]

        if len(available) >= min(n_sources, graph.vcount()):
            used_original_sources = np.asarray(
                available[: min(n_sources, graph.vcount())],
                dtype=np.int64,
            )
        else:
            # Usa todas as origens preferenciais ainda presentes e completa,
            # se necessário, com outras origens da componente.
            used = set(available)
            remaining = [
                int(orig)
                for orig in original_ids
                if int(orig) not in used
            ]

            need = min(n_sources, graph.vcount()) - len(available)

            if need > 0:
                extra = rng.choice(
                    np.asarray(remaining, dtype=np.int64),
                    size=need,
                    replace=False,
                ).tolist()
            else:
                extra = []

            used_original_sources = np.asarray(
                available + extra,
                dtype=np.int64,
            )

        local_sources = np.asarray(
            [original_to_local[int(orig)] for orig in used_original_sources],
            dtype=np.int64,
        )

    source_means = []
    source_eccentricities = []
    bfs_times = []

    for i, source in enumerate(local_sources, start=1):
        start = time.perf_counter()

        distances = graph.distances(
            source=int(source),
            mode="all",
        )[0]

        bfs_times.append(time.perf_counter() - start)

        distances = np.asarray(distances, dtype=np.float64)

        if not np.isfinite(distances).all():
            raise RuntimeError(
                "Foi encontrado vértice inalcançável dentro da maior componente."
            )

        values = distances[distances > 0]

        source_means.append(float(values.mean()))
        source_eccentricities.append(int(values.max()))

        if i % 25 == 0 or i == len(local_sources):
            print(
                f"\rBFS: {i}/{len(local_sources)}",
                end="",
                flush=True,
            )

    print()

    source_means = np.asarray(source_means, dtype=np.float64)
    source_eccentricities = np.asarray(
        source_eccentricities,
        dtype=np.int64,
    )
    bfs_times = np.asarray(bfs_times, dtype=np.float64)

    mean_distance = float(source_means.mean())

    if len(source_means) > 1:
        se = float(
            source_means.std(ddof=1)
            / np.sqrt(len(source_means))
        )
    else:
        se = 0.0

    return {
        "sampled_sources": int(len(local_sources)),
        "mean_distance": mean_distance,
        "distance_se": se,
        "distance_ci95_low": float(mean_distance - 1.96 * se),
        "distance_ci95_high": float(mean_distance + 1.96 * se),
        "maximum_distance_observed": int(source_eccentricities.max()),
        "mean_bfs_seconds": float(bfs_times.mean()),
        "original_sources": used_original_sources.tolist(),
    }


def graph_clustering_metrics(graph, eligible_mask):
    global_transitivity = float(
        graph.transitivity_undirected(mode="zero")
    )

    local = np.asarray(
        graph.transitivity_local_undirected(mode="zero"),
        dtype=np.float64,
    )

    mean_local = float(local[eligible_mask].mean())

    return {
        "global_transitivity": global_transitivity,
        "mean_local_clustering": mean_local,
    }


def rewire_simple_degree_preserving(graph, n_trials):
    """
    Compatibilidade entre versões do python-igraph.

    Versões mais novas usam allowed_edge_types="simple".
    Versões 0.10/0.11 usam mode="simple".
    """
    try:
        graph.rewire(
            n=n_trials,
            allowed_edge_types="simple",
        )
    except TypeError:
        graph.rewire(
            n=n_trials,
            mode="simple",
        )


def describe_null(values):
    x = np.asarray(values, dtype=np.float64)

    if len(x) > 1:
        std = float(x.std(ddof=1))
    else:
        std = 0.0

    return {
        "mean": float(x.mean()),
        "std": std,
        "min": float(x.min()),
        "max": float(x.max()),
    }


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    total_start = time.time()

    section("Parâmetros")

    print(f"Nulos:              {args.n_null}")
    print(f"Swap factor:        {args.swap_factor}")
    print(f"Origens distância:  {args.distance_sources}")
    print(f"Seed base:          {args.seed}")

    if not EDGES_PATH.exists():
        raise FileNotFoundError(f"Não encontrado: {EDGES_PATH}")

    if not NODE_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Não encontrado: {NODE_FEATURES_PATH}\n"
            "Dependência gerada por scripts/f2_01_structural_metrics.py."
        )

    # --------------------------------------------------------
    # 1. Carrega nós e arestas
    # --------------------------------------------------------

    section("1. Carregando dados")

    start = time.time()

    nodes = pd.read_csv(
        NODE_FEATURES_PATH,
        usecols=["root_id", "undirected_degree"],
        dtype={
            "root_id": "int64",
            "undirected_degree": "int64",
        },
    )

    if nodes["root_id"].duplicated().any():
        raise RuntimeError("Há root_id duplicado na tabela F2.")

    edges = pd.read_csv(
        EDGES_PATH,
        usecols=["pre_root_id", "post_root_id"],
        dtype={
            "pre_root_id": "int64",
            "post_root_id": "int64",
        },
    )

    n_nodes = len(nodes)
    n_directed_edges = len(edges)

    print(f"Nós:               {fmt_int(n_nodes)}")
    print(f"Arestas dirigidas: {fmt_int(n_directed_edges)}")
    print(f"Tempo:              {elapsed(start)}")

    # --------------------------------------------------------
    # 2. Constrói projeção simples não dirigida
    # --------------------------------------------------------

    section("2. Construindo projeção simples não dirigida")

    start = time.time()

    node_index = pd.Index(
        nodes["root_id"].to_numpy(dtype=np.int64)
    )

    source = node_index.get_indexer(
        edges["pre_root_id"].to_numpy(dtype=np.int64)
    )
    target = node_index.get_indexer(
        edges["post_root_id"].to_numpy(dtype=np.int64)
    )

    if (source < 0).any() or (target < 0).any():
        raise RuntimeError(
            "Há extremos de aresta ausentes da tabela de nós."
        )

    graph_edges = pd.DataFrame(
        {
            "source": source.astype(np.int64),
            "target": target.astype(np.int64),
        }
    )

    vertices = pd.DataFrame(
        {
            "vertex_id": np.arange(n_nodes, dtype=np.int64),
            "root_id": nodes["root_id"].to_numpy(dtype=np.int64),
        }
    )

    g = ig.Graph.DataFrame(
        graph_edges,
        directed=True,
        vertices=vertices,
        use_vids=True,
    )

    if not g.is_simple():
        raise RuntimeError(
            "O grafo dirigido deveria ser simples após agregação por par."
        )

    u = g.as_undirected(mode="collapse")

    if not u.is_simple():
        raise RuntimeError(
            "A projeção não dirigida deveria ser simples."
        )

    del g, graph_edges, vertices, edges, source, target
    gc.collect()

    n_undirected_edges = u.ecount()

    real_degree = np.asarray(
        u.degree(),
        dtype=np.int64,
    )

    expected_degree = nodes["undirected_degree"].to_numpy(
        dtype=np.int64
    )

    if not np.array_equal(real_degree, expected_degree):
        raise RuntimeError(
            "Os graus da projeção não coincidem com undirected_degree da F2."
        )

    eligible = real_degree >= 2

    print(f"Vértices:          {fmt_int(u.vcount())}")
    print(f"Arestas:           {fmt_int(u.ecount())}")
    print(f"Grau médio:        {real_degree.mean():.6f}")
    print(f"Grau máximo:       {fmt_int(real_degree.max())}")
    print(f"Vértices grau>=2:  {fmt_int(eligible.sum())}")
    print(f"Tempo:             {elapsed(start)}")

    # --------------------------------------------------------
    # 3. Métricas do grafo real
    # --------------------------------------------------------

    section("3. Métricas da rede real")

    start = time.time()

    real_cluster = graph_clustering_metrics(
        u,
        eligible_mask=eligible,
    )

    real_giant, real_giant_size, real_components = (
        largest_component_info(u)
    )

    real_distance = estimate_mean_distance(
        graph=real_giant,
        n_sources=args.distance_sources,
        seed=args.seed,
        preferred_original_sources=None,
    )

    preferred_sources = real_distance["original_sources"]

    print(
        "Transitividade:          "
        f"{real_cluster['global_transitivity']:.9f}"
    )
    print(
        "Clustering local médio:  "
        f"{real_cluster['mean_local_clustering']:.9f}"
    )
    print(
        "Componentes:             "
        f"{fmt_int(real_components)}"
    )
    print(
        "Maior componente:        "
        f"{fmt_int(real_giant_size)}"
    )
    print(
        "Distância média estimada:"
        f" {real_distance['mean_distance']:.6f}"
    )
    print(
        "IC95% aproximado:        "
        f"[{real_distance['distance_ci95_low']:.6f}, "
        f"{real_distance['distance_ci95_high']:.6f}]"
    )
    print(f"Tempo:                    {elapsed(start)}")

    del real_giant
    gc.collect()

    # --------------------------------------------------------
    # 4. Aproximação analítica do modelo de configuração
    # --------------------------------------------------------

    section("4. Aproximação analítica para transitividade")

    mean_k = float(real_degree.mean())
    mean_k2 = float(np.mean(real_degree.astype(np.float64) ** 2))

    analytic_transitivity = float(
        ((mean_k2 - mean_k) ** 2)
        / (n_nodes * (mean_k ** 3))
    )

    print(f"<k>:                    {mean_k:.9f}")
    print(f"<k²>:                   {mean_k2:.9f}")
    print(
        "C_config analítico:     "
        f"{analytic_transitivity:.9f}"
    )
    print(
        "Real / analítico:       "
        f"{real_cluster['global_transitivity'] / analytic_transitivity:.4f}x"
    )

    # --------------------------------------------------------
    # 5. Realizações nulas
    # --------------------------------------------------------

    section("5. Realizações nulas degree-preserving")

    n_rewire_trials = max(
        1,
        int(round(args.swap_factor * n_undirected_edges)),
    )

    print(
        "Tentativas de rewiring por realização: "
        f"{fmt_int(n_rewire_trials)}"
    )
    print(
        "Cada realização parte novamente da rede real e usa "
        "rewiring simples, preservando exatamente o grau de cada vértice."
    )

    realization_rows = []

    # Força o igraph a usar o módulo random do Python,
    # permitindo controlar a seed explicitamente.
    ig.set_random_number_generator(random)

    for realization in range(1, args.n_null + 1):
        section(
            f"Realização nula {realization}/{args.n_null}"
        )

        realization_start = time.time()

        null_seed = args.seed + 1000 * realization
        random.seed(null_seed)

        null = u.copy()

        degree_before = np.asarray(
            null.degree(),
            dtype=np.int64,
        )

        rewire_start = time.time()

        rewire_simple_degree_preserving(
            null,
            n_trials=n_rewire_trials,
        )

        rewire_seconds = time.time() - rewire_start

        if not null.is_simple():
            raise RuntimeError(
                f"Nulo {realization}: rewiring gerou grafo não simples."
            )

        if null.vcount() != u.vcount():
            raise RuntimeError(
                f"Nulo {realization}: número de vértices mudou."
            )

        if null.ecount() != u.ecount():
            raise RuntimeError(
                f"Nulo {realization}: número de arestas mudou."
            )

        degree_after = np.asarray(
            null.degree(),
            dtype=np.int64,
        )

        if not np.array_equal(
            degree_before,
            degree_after,
        ):
            raise RuntimeError(
                f"Nulo {realization}: sequência de graus não foi preservada."
            )

        null_cluster = graph_clustering_metrics(
            null,
            eligible_mask=eligible,
        )

        null_giant, null_giant_size, null_components = (
            largest_component_info(null)
        )

        null_distance = estimate_mean_distance(
            graph=null_giant,
            n_sources=args.distance_sources,
            seed=null_seed + 17,
            preferred_original_sources=preferred_sources,
        )

        sigma_transitivity = float(
            (
                real_cluster["global_transitivity"]
                / null_cluster["global_transitivity"]
            )
            /
            (
                real_distance["mean_distance"]
                / null_distance["mean_distance"]
            )
        )

        sigma_local = float(
            (
                real_cluster["mean_local_clustering"]
                / null_cluster["mean_local_clustering"]
            )
            /
            (
                real_distance["mean_distance"]
                / null_distance["mean_distance"]
            )
        )

        row = {
            "realization": realization,
            "seed": null_seed,
            "rewire_trials": n_rewire_trials,
            "rewire_seconds": rewire_seconds,
            "vertices": null.vcount(),
            "edges": null.ecount(),
            "components": null_components,
            "largest_component_vertices": null_giant_size,
            "global_transitivity": (
                null_cluster["global_transitivity"]
            ),
            "mean_local_clustering": (
                null_cluster["mean_local_clustering"]
            ),
            "sampled_sources": (
                null_distance["sampled_sources"]
            ),
            "mean_distance": (
                null_distance["mean_distance"]
            ),
            "distance_se": (
                null_distance["distance_se"]
            ),
            "distance_ci95_low": (
                null_distance["distance_ci95_low"]
            ),
            "distance_ci95_high": (
                null_distance["distance_ci95_high"]
            ),
            "maximum_distance_observed": (
                null_distance["maximum_distance_observed"]
            ),
            "sigma_transitivity": sigma_transitivity,
            "sigma_local_clustering": sigma_local,
            "total_seconds": (
                time.time() - realization_start
            ),
        }

        realization_rows.append(row)

        # Salva progresso após cada realização para não perder
        # tudo se houver interrupção posterior.
        pd.DataFrame(realization_rows).to_csv(
            REALIZATIONS_PATH,
            index=False,
        )

        print(
            "Transitividade:         "
            f"{row['global_transitivity']:.9f}"
        )
        print(
            "Clustering local médio: "
            f"{row['mean_local_clustering']:.9f}"
        )
        print(
            "Distância média:        "
            f"{row['mean_distance']:.6f}"
        )
        print(
            "Maior componente:       "
            f"{fmt_int(row['largest_component_vertices'])}"
        )
        print(
            "Sigma (transitividade): "
            f"{row['sigma_transitivity']:.4f}"
        )
        print(
            "Sigma (clustering):     "
            f"{row['sigma_local_clustering']:.4f}"
        )
        print(
            "Rewiring:               "
            f"{rewire_seconds:.2f} s"
        )
        print(
            "Tempo total realização: "
            f"{row['total_seconds']:.2f} s"
        )

        del null_giant, null
        gc.collect()

    # Restaura RNG interno padrão do igraph.
    ig.set_random_number_generator(None)

    # --------------------------------------------------------
    # 6. Consolidação
    # --------------------------------------------------------

    section("6. Consolidando resultados")

    realizations = pd.DataFrame(realization_rows)

    metric_names = [
        "global_transitivity",
        "mean_local_clustering",
        "mean_distance",
        "largest_component_vertices",
        "sigma_transitivity",
        "sigma_local_clustering",
    ]

    summary_rows = []

    for metric in metric_names:
        stats = describe_null(
            realizations[metric].to_numpy(dtype=np.float64)
        )
        summary_rows.append(
            {
                "metric": metric,
                "real_value": (
                    real_cluster["global_transitivity"]
                    if metric == "global_transitivity"
                    else real_cluster["mean_local_clustering"]
                    if metric == "mean_local_clustering"
                    else real_distance["mean_distance"]
                    if metric == "mean_distance"
                    else real_giant_size
                    if metric == "largest_component_vertices"
                    else np.nan
                ),
                "null_mean": stats["mean"],
                "null_std": stats["std"],
                "null_min": stats["min"],
                "null_max": stats["max"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    # Razões usando a média das realizações nulas.
    null_trans_mean = float(
        realizations["global_transitivity"].mean()
    )
    null_local_mean = float(
        realizations["mean_local_clustering"].mean()
    )
    null_distance_mean = float(
        realizations["mean_distance"].mean()
    )

    trans_ratio = float(
        real_cluster["global_transitivity"]
        / null_trans_mean
    )
    local_ratio = float(
        real_cluster["mean_local_clustering"]
        / null_local_mean
    )
    distance_ratio = float(
        real_distance["mean_distance"]
        / null_distance_mean
    )

    sigma_from_means_trans = float(
        trans_ratio / distance_ratio
    )
    sigma_from_means_local = float(
        local_ratio / distance_ratio
    )

    sigma_trans_stats = describe_null(
        realizations["sigma_transitivity"]
    )
    sigma_local_stats = describe_null(
        realizations["sigma_local_clustering"]
    )

    results = {
        "parameters": {
            "n_null": args.n_null,
            "swap_factor": args.swap_factor,
            "rewire_trials_per_realization": (
                n_rewire_trials
            ),
            "distance_sources": args.distance_sources,
            "seed": args.seed,
        },
        "graph": {
            "vertices": u.vcount(),
            "directed_edges_original": n_directed_edges,
            "undirected_edges": u.ecount(),
            "degree_mean": mean_k,
            "degree_second_moment": mean_k2,
            "degree_max": int(real_degree.max()),
            "clustering_eligible_vertices": int(
                eligible.sum()
            ),
        },
        "real": {
            "global_transitivity": (
                real_cluster["global_transitivity"]
            ),
            "mean_local_clustering": (
                real_cluster["mean_local_clustering"]
            ),
            "components": real_components,
            "largest_component_vertices": real_giant_size,
            "mean_distance": (
                real_distance["mean_distance"]
            ),
            "distance_se": real_distance["distance_se"],
            "distance_ci95_low": (
                real_distance["distance_ci95_low"]
            ),
            "distance_ci95_high": (
                real_distance["distance_ci95_high"]
            ),
        },
        "analytic_configuration_model": {
            "expected_transitivity_approx": (
                analytic_transitivity
            ),
            "real_to_analytic_ratio": (
                real_cluster["global_transitivity"]
                / analytic_transitivity
            ),
        },
        "null_mean": {
            "global_transitivity": null_trans_mean,
            "mean_local_clustering": null_local_mean,
            "mean_distance": null_distance_mean,
            "largest_component_vertices": float(
                realizations[
                    "largest_component_vertices"
                ].mean()
            ),
        },
        "null_std": {
            "global_transitivity": float(
                realizations[
                    "global_transitivity"
                ].std(ddof=1)
                if args.n_null > 1
                else 0.0
            ),
            "mean_local_clustering": float(
                realizations[
                    "mean_local_clustering"
                ].std(ddof=1)
                if args.n_null > 1
                else 0.0
            ),
            "mean_distance": float(
                realizations[
                    "mean_distance"
                ].std(ddof=1)
                if args.n_null > 1
                else 0.0
            ),
        },
        "ratios_real_to_null_mean": {
            "transitivity": trans_ratio,
            "mean_local_clustering": local_ratio,
            "mean_distance": distance_ratio,
        },
        "small_world_sigma": {
            "from_null_means_transitivity": (
                sigma_from_means_trans
            ),
            "from_null_means_local_clustering": (
                sigma_from_means_local
            ),
            "per_realization_transitivity_mean": (
                sigma_trans_stats["mean"]
            ),
            "per_realization_transitivity_std": (
                sigma_trans_stats["std"]
            ),
            "per_realization_local_mean": (
                sigma_local_stats["mean"]
            ),
            "per_realization_local_std": (
                sigma_local_stats["std"]
            ),
        },
        "runtime_seconds": float(
            time.time() - total_start
        ),
    }

    realizations.to_csv(
        REALIZATIONS_PATH,
        index=False,
    )
    summary_df.to_csv(
        SUMMARY_CSV_PATH,
        index=False,
    )

    with SUMMARY_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    summary_text = f"""
FlyWire FAFB v783 — modelo nulo degree-preserving

PARÂMETROS
Nulos: {args.n_null}
Swap factor: {args.swap_factor}
Tentativas de rewiring por realização: {n_rewire_trials}
Origens para distância: {args.distance_sources}
Seed base: {args.seed}

PROJEÇÃO NÃO DIRIGIDA REAL
Vértices: {u.vcount()}
Arestas: {u.ecount()}
Grau médio: {mean_k:.9f}
<k²>: {mean_k2:.9f}
Vértices com grau >= 2: {int(eligible.sum())}

REDE REAL
Transitividade global: {real_cluster['global_transitivity']:.9f}
Clustering local médio (grau >= 2): {real_cluster['mean_local_clustering']:.9f}
Componentes: {real_components}
Maior componente: {real_giant_size}
Distância média estimada: {real_distance['mean_distance']:.6f}
IC95% aproximado da distância: [{real_distance['distance_ci95_low']:.6f}, {real_distance['distance_ci95_high']:.6f}]

APROXIMAÇÃO ANALÍTICA DO MODELO DE CONFIGURAÇÃO
Transitividade aproximada: {analytic_transitivity:.9f}
Real / analítico: {real_cluster['global_transitivity'] / analytic_transitivity:.4f}x

NULOS — MÉDIA ± DP
Transitividade: {null_trans_mean:.9f} ± {results['null_std']['global_transitivity']:.9f}
Clustering local médio: {null_local_mean:.9f} ± {results['null_std']['mean_local_clustering']:.9f}
Distância média: {null_distance_mean:.6f} ± {results['null_std']['mean_distance']:.6f}

RAZÕES REAL / NULO
Transitividade: {trans_ratio:.4f}x
Clustering local médio: {local_ratio:.4f}x
Distância média: {distance_ratio:.4f}x

SMALL-WORLD SIGMA
Usando transitividade: {sigma_from_means_trans:.4f}
Usando clustering local médio: {sigma_from_means_local:.4f}

Sigma por realização — transitividade:
{sigma_trans_stats['mean']:.4f} ± {sigma_trans_stats['std']:.4f}

Sigma por realização — clustering local:
{sigma_local_stats['mean']:.4f} ± {sigma_local_stats['std']:.4f}

Observação metodológica:
Os nulos são produzidos por rewiring simples da projeção não dirigida.
O número de vértices, o número de arestas e o grau de CADA vértice são
verificados após cada realização. A distância é estimada na maior componente
usando as mesmas origens da rede real sempre que elas permanecem na maior
componente nula.

Tempo total: {results['runtime_seconds'] / 60:.2f} min
""".strip()

    SUMMARY_TXT_PATH.write_text(
        summary_text,
        encoding="utf-8",
    )

    section("Concluído")

    print(summary_text)
    print()
    print(f"Realizações: {REALIZATIONS_PATH}")
    print(f"Resumo CSV:  {SUMMARY_CSV_PATH}")
    print(f"Resumo JSON: {SUMMARY_JSON_PATH}")
    print(f"Resumo TXT:  {SUMMARY_TXT_PATH}")


if __name__ == "__main__":
    main()
