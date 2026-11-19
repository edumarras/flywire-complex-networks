from pathlib import Path
import argparse
import gc
import json
import random
import time

import igraph as ig
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# Caminhos
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "f2"
TABLE_DIR = REPORT_DIR / "tables"
COMMUNITY_DIR = REPORT_DIR / "communities"
FIGURE_DIR = COMMUNITY_DIR / "figures"

EDGES_PATH = PROCESSED_DIR / "flywire_fafb_v783_edges.csv.gz"
NODE_FEATURES_PATH = TABLE_DIR / "node_structural_features.csv.gz"

ASSIGNMENTS_PATH = COMMUNITY_DIR / "community_assignments.csv.gz"
LEIDEN_SIZES_PATH = COMMUNITY_DIR / "leiden_community_sizes.csv"
INFOMAP_SIZES_PATH = COMMUNITY_DIR / "infomap_community_sizes.csv"
SUMMARY_CSV_PATH = COMMUNITY_DIR / "community_detection_summary.csv"
SUMMARY_JSON_PATH = COMMUNITY_DIR / "community_detection_summary.json"
SUMMARY_TXT_PATH = COMMUNITY_DIR / "community_detection_summary.txt"
LEIDEN_TRIALS_PATH = COMMUNITY_DIR / "leiden_trials.csv"

SIZE_CCDF_FIGURE = FIGURE_DIR / "community_size_ccdf.png"
RANK_SIZE_FIGURE = FIGURE_DIR / "community_rank_size.png"

COMMUNITY_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Funções auxiliares
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Detecção de comunidades no conectoma FlyWire: "
            "Leiden na projeção ponderada não dirigida e Infomap "
            "no grafo ponderado dirigido."
        )
    )

    parser.add_argument(
        "--leiden-trials",
        type=int,
        default=5,
        help=(
            "Número de inicializações independentes do Leiden. "
            "A partição com maior modularidade ponderada é mantida. Default: 5."
        ),
    )
    parser.add_argument(
        "--infomap-trials",
        type=int,
        default=10,
        help=(
            "Número de tentativas internas do Infomap. Default: 10."
        ),
    )
    parser.add_argument(
        "--leiden-resolution",
        type=float,
        default=1.0,
        help=(
            "Resolução fixa do Leiden sob modularidade. "
            "Resolução fixa do Leiden. Default: 1.0; independente de super_class."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed base. Default: 42.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Desativa a geração de arquivos PNG.",
    )

    args = parser.parse_args()

    if args.leiden_trials < 1:
        parser.error("--leiden-trials deve ser >= 1.")
    if args.infomap_trials < 1:
        parser.error("--infomap-trials deve ser >= 1.")
    if args.leiden_resolution <= 0:
        parser.error("--leiden-resolution deve ser > 0.")

    return args


def section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fmt_int(x):
    return f"{int(x):,}".replace(",", ".")


def elapsed(start):
    return f"{time.time() - start:.2f} s"


def canonicalize_membership(membership):
    """
    Renumera comunidades por tamanho:
      0 = maior, 1 = segunda maior, ...

    Isso não muda a partição; apenas torna os IDs mais legíveis e estáveis.
    """
    membership = np.asarray(membership, dtype=np.int64)

    labels, counts = np.unique(
        membership,
        return_counts=True,
    )

    order = sorted(
        zip(labels.tolist(), counts.tolist()),
        key=lambda x: (-x[1], x[0]),
    )

    mapping = {
        int(old_label): int(new_label)
        for new_label, (old_label, _) in enumerate(order)
    }

    canonical = np.fromiter(
        (mapping[int(x)] for x in membership),
        dtype=np.int64,
        count=len(membership),
    )

    return canonical


def community_size_table(membership, algorithm):
    membership = np.asarray(membership, dtype=np.int64)

    communities, counts = np.unique(
        membership,
        return_counts=True,
    )

    out = pd.DataFrame(
        {
            "community": communities,
            "size": counts,
        }
    ).sort_values(
        ["size", "community"],
        ascending=[False, True],
    ).reset_index(drop=True)

    out["rank"] = np.arange(1, len(out) + 1, dtype=np.int64)
    out["fraction_of_active_nodes"] = out["size"] / len(membership)
    out.insert(0, "algorithm", algorithm)

    return out[
        [
            "algorithm",
            "community",
            "rank",
            "size",
            "fraction_of_active_nodes",
        ]
    ]


def size_stats(size_df):
    sizes = size_df["size"].to_numpy(dtype=np.int64)

    return {
        "communities": int(len(sizes)),
        "singleton_communities": int(np.sum(sizes == 1)),
        "communities_size_ge_10": int(np.sum(sizes >= 10)),
        "communities_size_ge_100": int(np.sum(sizes >= 100)),
        "communities_size_ge_1000": int(np.sum(sizes >= 1000)),
        "largest_size": int(sizes.max()),
        "largest_fraction": float(sizes.max() / sizes.sum()),
        "median_size": float(np.median(sizes)),
        "mean_size": float(np.mean(sizes)),
        "p90_size": float(np.quantile(sizes, 0.90)),
        "p95_size": float(np.quantile(sizes, 0.95)),
        "p99_size": float(np.quantile(sizes, 0.99)),
    }


def save_figures(leiden_sizes, infomap_sizes):
    # CCDF dos tamanhos das comunidades
    fig = go.Figure()

    for df, label in [
        (leiden_sizes, "Leiden"),
        (infomap_sizes, "Infomap"),
    ]:
        sizes = np.sort(df["size"].to_numpy(dtype=np.int64))
        unique_sizes, counts = np.unique(sizes, return_counts=True)
        ccdf = counts[::-1].cumsum()[::-1] / counts.sum()

        fig.add_trace(
            go.Scatter(
                x=unique_sizes,
                y=ccdf,
                mode="markers",
                name=label,
                marker={"size": 6},
                hovertemplate=(
                    "Tamanho: %{x}<br>"
                    "P(S >= s): %{y:.6g}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="CCDF dos tamanhos das comunidades",
        xaxis_title="Tamanho da comunidade",
        yaxis_title="CCDF — P(S ≥ s)",
        xaxis_type="log",
        yaxis_type="log",
        template="simple_white",
        width=1000,
        height=700,
    )

    fig.write_image(SIZE_CCDF_FIGURE, scale=2)

    # Rank-size
    fig = go.Figure()

    for df, label in [
        (leiden_sizes, "Leiden"),
        (infomap_sizes, "Infomap"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=df["rank"],
                y=df["size"],
                mode="markers",
                name=label,
                marker={"size": 6},
                hovertemplate=(
                    "Rank: %{x}<br>"
                    "Tamanho: %{y}<br>"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Rank-size das comunidades",
        xaxis_title="Rank da comunidade por tamanho",
        yaxis_title="Número de neurônios",
        xaxis_type="log",
        yaxis_type="log",
        template="simple_white",
        width=1000,
        height=700,
    )

    fig.write_image(RANK_SIZE_FIGURE, scale=2)


def run_leiden_trials(graph, trials, resolution, seed):
    """
    Executa Leiden várias vezes com seeds distintas e escolhe a partição
    de maior modularidade ponderada. Nenhum rótulo biológico é usado.
    """
    trial_rows = []
    best_membership = None
    best_modularity = -np.inf
    best_trial = None

    ig.set_random_number_generator(random)

    try:
        for trial in range(1, trials + 1):
            trial_seed = seed + trial * 1000
            random.seed(trial_seed)

            start = time.time()

            clustering = graph.community_leiden(
                objective_function="modularity",
                weights="weight",
                resolution=resolution,
                n_iterations=-1,
            )

            membership = np.asarray(
                clustering.membership,
                dtype=np.int64,
            )

            modularity = float(
                graph.modularity(
                    membership,
                    weights="weight",
                    resolution=resolution,
                    directed=False,
                )
            )

            runtime = time.time() - start
            n_communities = int(len(np.unique(membership)))

            trial_rows.append(
                {
                    "trial": trial,
                    "seed": trial_seed,
                    "communities": n_communities,
                    "weighted_undirected_modularity": modularity,
                    "runtime_seconds": runtime,
                }
            )

            print(
                f"Leiden {trial}/{trials}: "
                f"Q={modularity:.9f}, "
                f"comunidades={fmt_int(n_communities)}, "
                f"tempo={runtime:.2f} s"
            )

            if modularity > best_modularity:
                best_modularity = modularity
                best_membership = membership.copy()
                best_trial = trial

            del clustering, membership
            gc.collect()

    finally:
        ig.set_random_number_generator(None)

    if best_membership is None:
        raise RuntimeError("Leiden não produziu nenhuma partição.")

    return (
        best_membership,
        float(best_modularity),
        int(best_trial),
        pd.DataFrame(trial_rows),
    )


def run_infomap(graph, trials, seed):
    """
    Executa Infomap no grafo dirigido e ponderado.
    O próprio algoritmo usa 'trials' tentativas e mantém a melhor solução.
    """
    ig.set_random_number_generator(random)
    random.seed(seed)

    try:
        start = time.time()

        clustering = graph.community_infomap(
            edge_weights="weight",
            trials=trials,
        )

        runtime = time.time() - start

        membership = np.asarray(
            clustering.membership,
            dtype=np.int64,
        )

        codelength = float(
            getattr(clustering, "codelength", np.nan)
        )

    finally:
        ig.set_random_number_generator(None)

    return membership, codelength, runtime


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    total_start = time.time()

    section("Parâmetros")
    print(f"Leiden trials:       {args.leiden_trials}")
    print(f"Infomap trials:      {args.infomap_trials}")
    print(f"Leiden resolution:   {args.leiden_resolution}")
    print(f"Seed base:           {args.seed}")
    print(f"Gerar figuras:       {not args.no_figures}")

    for path in [EDGES_PATH, NODE_FEATURES_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    # --------------------------------------------------------
    # 1. Carrega os nós e remove apenas isolados para detecção
    # --------------------------------------------------------

    section("1. Carregando nós")
    start = time.time()

    nodes = pd.read_csv(
        NODE_FEATURES_PATH,
        usecols=["root_id", "isolated"],
        dtype={"root_id": "int64"},
    )

    if nodes["root_id"].duplicated().any():
        raise RuntimeError("Há root_id duplicado na tabela estrutural.")

    # Aceita bool, 0/1 ou strings convertíveis de forma segura.
    if nodes["isolated"].dtype != bool:
        if set(pd.unique(nodes["isolated"].dropna())).issubset({0, 1}):
            nodes["isolated"] = nodes["isolated"].astype(bool)
        else:
            nodes["isolated"] = (
                nodes["isolated"].astype(str).str.lower().map(
                    {"true": True, "false": False}
                )
            )

    if nodes["isolated"].isna().any():
        raise RuntimeError("Não foi possível interpretar a coluna isolated.")

    n_total = len(nodes)
    n_isolates = int(nodes["isolated"].sum())

    active_mask = ~nodes["isolated"].to_numpy(dtype=bool)
    active_nodes = nodes.loc[active_mask, ["root_id"]].reset_index(drop=True)
    n_active = len(active_nodes)

    print(f"Nós totais:          {fmt_int(n_total)}")
    print(f"Nós ativos:          {fmt_int(n_active)}")
    print(f"Isolados:            {fmt_int(n_isolates)}")
    print(f"Tempo:               {elapsed(start)}")

    # --------------------------------------------------------
    # 2. Carrega arestas e constrói grafo dirigido ponderado
    # --------------------------------------------------------

    section("2. Construindo grafo ativo dirigido ponderado")
    start = time.time()

    edges = pd.read_csv(
        EDGES_PATH,
        dtype={
            "pre_root_id": "int64",
            "post_root_id": "int64",
            "weight": "int32",
        },
    )

    required = {"pre_root_id", "post_root_id", "weight"}
    missing = required.difference(edges.columns)
    if missing:
        raise RuntimeError(f"Colunas ausentes em edges: {sorted(missing)}")

    if (edges["weight"] <= 0).any():
        raise RuntimeError("Há pesos não positivos.")

    n_directed_edges = len(edges)
    total_weight_directed = int(edges["weight"].sum())

    active_index = pd.Index(
        active_nodes["root_id"].to_numpy(dtype=np.int64)
    )

    source = active_index.get_indexer(
        edges["pre_root_id"].to_numpy(dtype=np.int64)
    )
    target = active_index.get_indexer(
        edges["post_root_id"].to_numpy(dtype=np.int64)
    )

    if (source < 0).any() or (target < 0).any():
        raise RuntimeError(
            "Há extremos de aresta ausentes do conjunto de nós ativos."
        )

    graph_edges = pd.DataFrame(
        {
            "source": source.astype(np.int64),
            "target": target.astype(np.int64),
            "weight": edges["weight"].to_numpy(dtype=np.int32),
        }
    )

    vertices = pd.DataFrame(
        {
            "vertex_id": np.arange(n_active, dtype=np.int64),
            "root_id": active_nodes["root_id"].to_numpy(dtype=np.int64),
        }
    )

    del source, target, edges
    gc.collect()

    g = ig.Graph.DataFrame(
        graph_edges,
        directed=True,
        vertices=vertices,
        use_vids=True,
    )

    del graph_edges, vertices
    gc.collect()

    if g.vcount() != n_active:
        raise RuntimeError("Número inesperado de vértices no grafo dirigido.")
    if g.ecount() != n_directed_edges:
        raise RuntimeError("Número inesperado de arestas no grafo dirigido.")
    if not g.is_simple():
        raise RuntimeError("O grafo dirigido deveria ser simples.")
    if "weight" not in g.es.attributes():
        raise RuntimeError("O atributo de aresta 'weight' não foi importado.")

    print(f"Vértices ativos:     {fmt_int(g.vcount())}")
    print(f"Arestas dirigidas:   {fmt_int(g.ecount())}")
    print(f"Peso total:          {fmt_int(total_weight_directed)}")
    print(f"Simples:             {g.is_simple()}")
    print(f"Tempo:               {elapsed(start)}")

    # --------------------------------------------------------
    # 3. Infomap — dirigido e ponderado
    # --------------------------------------------------------

    section("3. Infomap — grafo dirigido ponderado")

    infomap_membership_raw, infomap_codelength, infomap_runtime = run_infomap(
        g,
        trials=args.infomap_trials,
        seed=args.seed + 500_000,
    )

    if len(infomap_membership_raw) != n_active:
        raise RuntimeError("Membership do Infomap com tamanho inesperado.")

    infomap_directed_modularity = float(
        g.modularity(
            infomap_membership_raw,
            weights="weight",
            resolution=1.0,
            directed=True,
        )
    )

    infomap_membership = canonicalize_membership(
        infomap_membership_raw
    )
    infomap_sizes = community_size_table(
        infomap_membership,
        "infomap",
    )
    infomap_stats = size_stats(infomap_sizes)

    print(f"Comunidades:         {fmt_int(infomap_stats['communities'])}")
    print(f"Maior comunidade:    {fmt_int(infomap_stats['largest_size'])}")
    print(f"Codelength:          {infomap_codelength:.9f}")
    print(f"Q dirigido (descr.): {infomap_directed_modularity:.9f}")
    print(f"Tempo:               {infomap_runtime:.2f} s")

    # --------------------------------------------------------
    # 4. Projeção ponderada não dirigida
    # --------------------------------------------------------

    section("4. Construindo projeção ponderada não dirigida")
    start = time.time()

    u = g.as_undirected(
        mode="collapse",
        combine_edges="sum",
    )

    if not u.is_simple():
        raise RuntimeError(
            "A projeção ponderada não dirigida deveria ser simples."
        )

    n_undirected_edges = u.ecount()
    total_weight_undirected = int(
        round(float(np.sum(u.es["weight"])))
    )

    if total_weight_undirected != total_weight_directed:
        raise RuntimeError(
            "A soma dos pesos mudou ao construir a projeção não dirigida."
        )

    print(f"Vértices ativos:     {fmt_int(u.vcount())}")
    print(f"Arestas não dirig.:  {fmt_int(u.ecount())}")
    print(f"Peso total:          {fmt_int(total_weight_undirected)}")
    print(f"Tempo:               {elapsed(start)}")

    # Infomap avaliado também sob a modularidade da mesma projeção,
    # apenas para comparação descritiva entre partições.
    infomap_undirected_modularity = float(
        u.modularity(
            infomap_membership_raw,
            weights="weight",
            resolution=1.0,
            directed=False,
        )
    )

    # O grafo dirigido não é mais necessário.
    del g, infomap_membership_raw
    gc.collect()

    # --------------------------------------------------------
    # 5. Leiden — projeção não dirigida ponderada
    # --------------------------------------------------------

    section("5. Leiden — projeção não dirigida ponderada")

    (
        leiden_membership_raw,
        leiden_modularity,
        best_leiden_trial,
        leiden_trials_df,
    ) = run_leiden_trials(
        u,
        trials=args.leiden_trials,
        resolution=args.leiden_resolution,
        seed=args.seed,
    )

    if len(leiden_membership_raw) != n_active:
        raise RuntimeError("Membership do Leiden com tamanho inesperado.")

    leiden_membership = canonicalize_membership(
        leiden_membership_raw
    )
    leiden_sizes = community_size_table(
        leiden_membership,
        "leiden",
    )
    leiden_stats = size_stats(leiden_sizes)

    print(f"Melhor trial:        {best_leiden_trial}")
    print(f"Comunidades:         {fmt_int(leiden_stats['communities'])}")
    print(f"Maior comunidade:    {fmt_int(leiden_stats['largest_size'])}")
    print(f"Q ponderado:         {leiden_modularity:.9f}")

    # --------------------------------------------------------
    # 6. Tabela de memberships no universo total de nós
    # --------------------------------------------------------

    section("6. Construindo tabela de comunidades por root_id")

    leiden_all = np.full(n_total, -1, dtype=np.int64)
    infomap_all = np.full(n_total, -1, dtype=np.int64)

    leiden_all[active_mask] = leiden_membership
    infomap_all[active_mask] = infomap_membership

    assignments = pd.DataFrame(
        {
            "root_id": nodes["root_id"].to_numpy(dtype=np.int64),
            "isolated": nodes["isolated"].to_numpy(dtype=bool),
            "leiden_community": leiden_all,
            "infomap_community": infomap_all,
        }
    )

    if assignments["root_id"].duplicated().any():
        raise RuntimeError("root_id duplicado na tabela de memberships.")

    if not (assignments.loc[assignments["isolated"], "leiden_community"] == -1).all():
        raise RuntimeError("Isolados receberam comunidade Leiden por engano.")

    if not (assignments.loc[assignments["isolated"], "infomap_community"] == -1).all():
        raise RuntimeError("Isolados receberam comunidade Infomap por engano.")

    if (assignments.loc[~assignments["isolated"], "leiden_community"] < 0).any():
        raise RuntimeError("Há nó ativo sem comunidade Leiden.")

    if (assignments.loc[~assignments["isolated"], "infomap_community"] < 0).any():
        raise RuntimeError("Há nó ativo sem comunidade Infomap.")

    assignments.to_csv(
        ASSIGNMENTS_PATH,
        index=False,
        compression="gzip",
    )

    leiden_sizes.to_csv(LEIDEN_SIZES_PATH, index=False)
    infomap_sizes.to_csv(INFOMAP_SIZES_PATH, index=False)
    leiden_trials_df.to_csv(LEIDEN_TRIALS_PATH, index=False)

    print(f"Memberships:         {ASSIGNMENTS_PATH}")
    print(f"Tamanhos Leiden:     {LEIDEN_SIZES_PATH}")
    print(f"Tamanhos Infomap:    {INFOMAP_SIZES_PATH}")

    # --------------------------------------------------------
    # 7. Resumo comparativo
    # --------------------------------------------------------

    section("7. Resumo")

    summary_df = pd.DataFrame(
        [
            {
                "algorithm": "leiden",
                "representation": "weighted undirected projection",
                "primary_objective": "modularity",
                "communities": leiden_stats["communities"],
                "singleton_communities": leiden_stats["singleton_communities"],
                "largest_size": leiden_stats["largest_size"],
                "largest_fraction_active": leiden_stats["largest_fraction"],
                "median_size": leiden_stats["median_size"],
                "mean_size": leiden_stats["mean_size"],
                "primary_score": leiden_modularity,
                "primary_score_name": "weighted_undirected_modularity",
                "descriptive_modularity": leiden_modularity,
                "descriptive_modularity_type": "weighted_undirected",
            },
            {
                "algorithm": "infomap",
                "representation": "weighted directed graph",
                "primary_objective": "map equation",
                "communities": infomap_stats["communities"],
                "singleton_communities": infomap_stats["singleton_communities"],
                "largest_size": infomap_stats["largest_size"],
                "largest_fraction_active": infomap_stats["largest_fraction"],
                "median_size": infomap_stats["median_size"],
                "mean_size": infomap_stats["mean_size"],
                "primary_score": infomap_codelength,
                "primary_score_name": "codelength_lower_is_better",
                "descriptive_modularity": infomap_directed_modularity,
                "descriptive_modularity_type": "weighted_directed",
            },
        ]
    )

    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)

    results = {
        "parameters": {
            "seed": args.seed,
            "leiden_trials": args.leiden_trials,
            "leiden_resolution": args.leiden_resolution,
            "infomap_trials": args.infomap_trials,
            "labels_used_for_detection": False,
        },
        "graph": {
            "total_vertices": n_total,
            "active_vertices": n_active,
            "isolated_vertices_excluded_from_detection": n_isolates,
            "directed_edges": n_directed_edges,
            "undirected_projection_edges": n_undirected_edges,
            "total_synapse_weight": total_weight_directed,
        },
        "leiden": {
            "representation": "weighted undirected projection",
            "objective": "modularity",
            "resolution": args.leiden_resolution,
            "best_trial": best_leiden_trial,
            "weighted_undirected_modularity": leiden_modularity,
            **leiden_stats,
        },
        "infomap": {
            "representation": "weighted directed graph",
            "objective": "map equation",
            "trials": args.infomap_trials,
            "codelength": infomap_codelength,
            "weighted_directed_modularity_descriptive": infomap_directed_modularity,
            "weighted_undirected_modularity_same_partition_descriptive": infomap_undirected_modularity,
            "runtime_seconds": infomap_runtime,
            **infomap_stats,
        },
        "notes": [
            "Community detection uses no biological labels.",
            "Leiden uses the weighted undirected projection with resolution fixed a priori.",
            "Infomap uses the weighted directed graph and optimizes map-equation codelength.",
            "Modularity and codelength are different objectives and should not be compared numerically as if they were the same score.",
            "Isolated vertices are excluded from community detection and receive community id -1 in the output table.",
            "Community IDs are renumbered by decreasing size; the integers have no ordinal meaning.",
        ],
        "runtime_seconds": float(time.time() - total_start),
    }

    with SUMMARY_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary_text = f"""
FlyWire FAFB v783 — detecção de comunidades

PRINCÍPIO METODOLÓGICO
Nenhum rótulo biológico foi usado na detecção ou seleção das partições.
Leiden e Infomap usam representações e objetivos adequados a cada método.

GRAFO
Nós totais: {n_total}
Nós ativos usados na detecção: {n_active}
Isolados excluídos da detecção: {n_isolates}
Arestas dirigidas: {n_directed_edges}
Arestas da projeção não dirigida: {n_undirected_edges}
Peso total de sinapses: {total_weight_directed}

LEIDEN
Representação: projeção ponderada não dirigida
Objetivo: modularidade
Resolução fixada a priori: {args.leiden_resolution}
Trials independentes: {args.leiden_trials}
Melhor trial: {best_leiden_trial}
Comunidades: {leiden_stats['communities']}
Comunidades singleton: {leiden_stats['singleton_communities']}
Maior comunidade: {leiden_stats['largest_size']} ({100 * leiden_stats['largest_fraction']:.3f}% dos nós ativos)
Tamanho mediano: {leiden_stats['median_size']:.3f}
Tamanho médio: {leiden_stats['mean_size']:.3f}
Modularidade ponderada não dirigida: {leiden_modularity:.9f}

INFOMAP
Representação: grafo ponderado dirigido
Objetivo: map equation
Trials internos: {args.infomap_trials}
Comunidades: {infomap_stats['communities']}
Comunidades singleton: {infomap_stats['singleton_communities']}
Maior comunidade: {infomap_stats['largest_size']} ({100 * infomap_stats['largest_fraction']:.3f}% dos nós ativos)
Tamanho mediano: {infomap_stats['median_size']:.3f}
Tamanho médio: {infomap_stats['mean_size']:.3f}
Codelength: {infomap_codelength:.9f}
Modularidade ponderada dirigida da partição (descritiva): {infomap_directed_modularity:.9f}
Modularidade da mesma partição na projeção não dirigida (descritiva): {infomap_undirected_modularity:.9f}

OBSERVAÇÕES
- Isolados recebem community = -1 e não entram na detecção.
- IDs de comunidade foram renumerados por tamanho: 0 é a maior comunidade.
- Os números dos IDs não têm significado ordinal.
- Leiden otimiza modularidade; Infomap otimiza codelength. Os scores não são diretamente comparáveis.
- NMI/ARI com super_class serão calculados depois, como RESULTADOS, e não usados para ajustar as partições.

Tempo total: {(time.time() - total_start) / 60:.2f} min
""".strip()

    SUMMARY_TXT_PATH.write_text(summary_text, encoding="utf-8")

    print(summary_text)

    # --------------------------------------------------------
    # 8. Figuras
    # --------------------------------------------------------

    if not args.no_figures:
        section("8. Gerando figuras")
        start = time.time()

        save_figures(
            leiden_sizes=leiden_sizes,
            infomap_sizes=infomap_sizes,
        )

        print(f"CCDF:                {SIZE_CCDF_FIGURE}")
        print(f"Rank-size:           {RANK_SIZE_FIGURE}")
        print(f"Tempo:               {elapsed(start)}")

    # --------------------------------------------------------
    # 9. Sanidade final
    # --------------------------------------------------------

    section("9. Checagens finais")

    if len(assignments) != n_total:
        raise RuntimeError("Tabela final não cobre todos os nós.")

    if assignments[["leiden_community", "infomap_community"]].isna().any().any():
        raise RuntimeError("Há NaN na tabela de memberships.")

    if int(leiden_sizes["size"].sum()) != n_active:
        raise RuntimeError("Tamanhos Leiden não somam os nós ativos.")

    if int(infomap_sizes["size"].sum()) != n_active:
        raise RuntimeError("Tamanhos Infomap não somam os nós ativos.")

    print("Cobertura de nós:                 OK")
    print("Isolados = -1:                    OK")
    print("Tamanhos das comunidades:         OK")
    print("Nenhum rótulo biológico carregado: OK")

    print()
    print("Concluído.")
    print(f"Resumo TXT:          {SUMMARY_TXT_PATH}")
    print(f"Resumo CSV:          {SUMMARY_CSV_PATH}")
    print(f"Resumo JSON:         {SUMMARY_JSON_PATH}")
    print(f"Memberships:         {ASSIGNMENTS_PATH}")

    del u
    gc.collect()


if __name__ == "__main__":
    main()
