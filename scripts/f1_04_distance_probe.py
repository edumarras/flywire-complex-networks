from pathlib import Path
import gc
import json
import time

import igraph as ig
import numpy as np
import pandas as pd


# Caminhos e parâmetros

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "f1"

NEURONS_PATH = RAW_DIR / "neurons.csv.gz"
EDGES_PATH = PROCESSED_DIR / "flywire_fafb_v783_edges.csv.gz"

OUTPUT_JSON = REPORT_DIR / "distance_probe.json"
OUTPUT_TXT = REPORT_DIR / "distance_probe.txt"

N_SOURCES = 100
RANDOM_SEED = 42


# Funções auxiliares

def section(title):
    print()
    print(title)


def fmt_int(x):
    return f"{int(x):,}".replace(",", ".")


def fmt_float(x, casas=6):
    return f"{x:.{casas}f}".replace(".", ",")


def probe_distances(graph, directed, n_sources, seed):
    rng = np.random.default_rng(seed)

    sources = rng.choice(
        graph.vcount(),
        size=min(n_sources, graph.vcount()),
        replace=False,
    )

    source_means = []
    source_eccentricities = []
    bfs_times = []

    for i, source in enumerate(sources, start=1):
        start = time.perf_counter()

        distances = graph.distances(
            source=int(source),
            mode="out" if directed else "all",
        )[0]

        elapsed = time.perf_counter() - start
        bfs_times.append(elapsed)

        distances = np.asarray(
            distances,
            dtype=np.float64,
        )

        # Como a componente é conexa (ou fortemente conexa),
        # todas as distâncias devem ser finitas.
        finite = np.isfinite(distances)

        if not finite.all():
            raise RuntimeError(
                "Foi encontrado um vértice inalcançável dentro da componente."
            )

        # Exclui a distância da origem até ela mesma.
        values = distances[distances > 0]

        source_means.append(
            float(values.mean())
        )

        source_eccentricities.append(
            int(values.max())
        )

        if i % 10 == 0 or i == len(sources):
            print(
                f"\rOrigens processadas: "
                f"{i}/{len(sources)}",
                end="",
                flush=True,
            )

    print()

    source_means = np.asarray(source_means)
    source_eccentricities = np.asarray(
        source_eccentricities
    )
    bfs_times = np.asarray(bfs_times)

    mean_distance = float(
        source_means.mean()
    )

    if len(source_means) > 1:
        se = float(
            source_means.std(ddof=1)
            / np.sqrt(len(source_means))
        )
    else:
        se = 0.0

    ci_low = mean_distance - 1.96 * se
    ci_high = mean_distance + 1.96 * se

    mean_bfs_time = float(
        bfs_times.mean()
    )

    # Estimativa do tempo necessário para executar uma BFS
    # a partir de todos os vértices.
    estimated_exact_seconds = (
        mean_bfs_time * graph.vcount()
    )

    return {
        "vertices": graph.vcount(),
        "edges": graph.ecount(),
        "sampled_sources": len(sources),

        "estimated_mean_distance": mean_distance,
        "approx_ci95_low": ci_low,
        "approx_ci95_high": ci_high,

        "maximum_distance_observed": int(
            source_eccentricities.max()
        ),

        "mean_bfs_seconds": mean_bfs_time,

        "estimated_exact_minutes": (
            estimated_exact_seconds / 60
        ),

        "estimated_exact_hours": (
            estimated_exact_seconds / 3600
        ),
    }


# Carrega os dados

section("1. Carregando o grafo")

nodes = pd.read_csv(
    NEURONS_PATH,
    usecols=["root_id"],
    dtype={"root_id": "int64"},
)

edges = pd.read_csv(
    EDGES_PATH,
    dtype={
        "pre_root_id": "int64",
        "post_root_id": "int64",
        "weight": "int64",
    },
)

node_index = pd.Index(
    nodes["root_id"].to_numpy()
)

source = node_index.get_indexer(
    edges["pre_root_id"].to_numpy()
)

target = node_index.get_indexer(
    edges["post_root_id"].to_numpy()
)

if (source < 0).any() or (target < 0).any():
    raise RuntimeError(
        "Um dos extremos de uma aresta não está no conjunto de nós."
    )

graph_edges = list(
    zip(
        source.tolist(),
        target.tolist(),
    )
)

g = ig.Graph(
    n=len(nodes),
    edges=graph_edges,
    directed=True,
)

print(f"Vértices: {fmt_int(g.vcount())}")
print(f"Arestas:  {fmt_int(g.ecount())}")

del edges
del graph_edges
del source
del target
gc.collect()


# Grafo direcionado: maior componente fortemente conexa

section("2. Maior componente fortemente conexa")

strong_components = g.connected_components(
    mode="strong"
)

scc = strong_components.giant()

print(f"Vértices: {fmt_int(scc.vcount())}")
print(f"Arestas:  {fmt_int(scc.ecount())}")

section("3. Amostragem de distâncias no grafo direcionado")

directed_result = probe_distances(
    graph=scc,
    directed=True,
    n_sources=N_SOURCES,
    seed=RANDOM_SEED,
)

labels = {
    "vertices": "Vértices",
    "edges": "Arestas",
    "sampled_sources": "Origens amostradas",
    "estimated_mean_distance": "Distância média estimada",
    "approx_ci95_low": "Limite inferior aproximado (95%)",
    "approx_ci95_high": "Limite superior aproximado (95%)",
    "maximum_distance_observed": "Maior distância observada",
    "mean_bfs_seconds": "Tempo médio por BFS (s)",
    "estimated_exact_minutes": "Tempo estimado do cálculo exato (min)",
    "estimated_exact_hours": "Tempo estimado do cálculo exato (h)",
}

for key, value in directed_result.items():
    label = labels.get(key, key)
    if isinstance(value, float):
        print(f"{label}: {fmt_float(value, 6)}")
    else:
        print(f"{label}: {fmt_int(value)}")

del scc
del strong_components
gc.collect()


# Projeção não direcionada: maior componente conexa

section("4. Projeção não direcionada")

u = g.as_undirected(
    mode="collapse"
)

del g
gc.collect()

components = u.connected_components()

giant = components.giant()

print(f"Vértices: {fmt_int(giant.vcount())}")
print(f"Arestas:  {fmt_int(giant.ecount())}")

section("5. Amostragem de distâncias na projeção não direcionada")

undirected_result = probe_distances(
    graph=giant,
    directed=False,
    n_sources=N_SOURCES,
    seed=RANDOM_SEED,
)

for key, value in undirected_result.items():
    label = labels.get(key, key)
    if isinstance(value, float):
        print(f"{label}: {fmt_float(value, 6)}")
    else:
        print(f"{label}: {fmt_int(value)}")


# Salva os resultados

# As chaves do JSON permanecem em inglês para manter compatibilidade com os outros scripts.
results = {
    "sample_size": N_SOURCES,
    "seed": RANDOM_SEED,
    "directed_largest_scc": directed_result,
    "undirected_giant_component": undirected_result,
}

with OUTPUT_JSON.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False,
    )


summary = f"""
FlyWire FAFB v783 - amostragem das distâncias

Número de origens amostradas: {N_SOURCES}
Seed: {RANDOM_SEED}

Maior componente fortemente conexa

Vértices: {fmt_int(directed_result['vertices'])}
Arestas: {fmt_int(directed_result['edges'])}

Distância média estimada:
{fmt_float(directed_result['estimated_mean_distance'], 6)}

Intervalo aproximado de 95%:
[{fmt_float(directed_result['approx_ci95_low'], 6)},
 {fmt_float(directed_result['approx_ci95_high'], 6)}]

Maior distância observada:
{directed_result['maximum_distance_observed']}

Tempo médio por BFS:
{fmt_float(directed_result['mean_bfs_seconds'], 6)} s

Tempo estimado para o cálculo exato:
{fmt_float(directed_result['estimated_exact_minutes'], 2)} min
({fmt_float(directed_result['estimated_exact_hours'], 2)} h)


Maior componente da projeção não direcionada

Vértices: {fmt_int(undirected_result['vertices'])}
Arestas: {fmt_int(undirected_result['edges'])}

Distância média estimada:
{fmt_float(undirected_result['estimated_mean_distance'], 6)}

Intervalo aproximado de 95%:
[{fmt_float(undirected_result['approx_ci95_low'], 6)},
 {fmt_float(undirected_result['approx_ci95_high'], 6)}]

Maior distância observada:
{undirected_result['maximum_distance_observed']}

Tempo médio por BFS:
{fmt_float(undirected_result['mean_bfs_seconds'], 6)} s

Tempo estimado para o cálculo exato:
{fmt_float(undirected_result['estimated_exact_minutes'], 2)} min
({fmt_float(undirected_result['estimated_exact_hours'], 2)} h)
""".strip()

OUTPUT_TXT.write_text(
    summary,
    encoding="utf-8",
)

section("Concluído")

print(summary)
print()
print(f"Relatório salvo em: {OUTPUT_TXT}")