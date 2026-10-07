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

CHECKPOINT_PATH = REPORT_DIR / "distance_exact_checkpoint.json"
OUTPUT_JSON = REPORT_DIR / "distance_exact.json"
OUTPUT_TXT = REPORT_DIR / "distance_exact.txt"

CHECKPOINT_EVERY = 1000


# Funções auxiliares

def section(title):
    print()
    print(title)


def fmt_int(x):
    return f"{int(x):,}".replace(",", ".")


def fmt_float(x, casas=6):
    return f"{x:.{casas}f}".replace(".", ",")


def fmt_percent(x, casas=2):
    return f"{100 * x:.{casas}f}".replace(".", ",") + "%"


def save_checkpoint(
    next_source,
    total_distance,
    diameter,
    elapsed_seconds,
    n_vertices,
    n_edges,
):
    data = {
        "next_source": next_source,
        "total_distance": int(total_distance),
        "diameter": int(diameter),
        "elapsed_seconds": float(elapsed_seconds),
        "vertices": int(n_vertices),
        "edges": int(n_edges),
    }

    with CHECKPOINT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
        )


# Carrega o grafo

section("1. Carregando o grafo")

nodes = pd.read_csv(
    NEURONS_PATH,
    usecols=["root_id"],
    dtype={"root_id": "int64"},
)

edges = pd.read_csv(
    EDGES_PATH,
    usecols=[
        "pre_root_id",
        "post_root_id",
    ],
    dtype={
        "pre_root_id": "int64",
        "post_root_id": "int64",
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

g = ig.Graph(
    n=len(nodes),
    edges=zip(source, target),
    directed=True,
)

print(f"Vértices no grafo completo: {fmt_int(g.vcount())}")
print(f"Arestas no grafo completo:  {fmt_int(g.ecount())}")

del nodes
del edges
del node_index
del source
del target

gc.collect()


# Maior componente fortemente conexa

section("2. Maior componente fortemente conexa")

components = g.connected_components(
    mode="strong"
)

scc = components.giant()

del components
del g
gc.collect()

n = scc.vcount()
m = scc.ecount()

print(f"Vértices: {fmt_int(n)}")
print(f"Arestas:  {fmt_int(m)}")


# Retoma o cálculo a partir do checkpoint, se existir

section("3. Menores caminhos direcionados")

start_source = 0
total_distance = 0
diameter = 0
previous_elapsed = 0.0

if CHECKPOINT_PATH.exists():

    with CHECKPOINT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        checkpoint = json.load(f)

    if (
        checkpoint["vertices"] != n
        or checkpoint["edges"] != m
    ):
        raise RuntimeError(
            "O checkpoint pertence a outro grafo."
        )

    start_source = checkpoint["next_source"]
    total_distance = checkpoint["total_distance"]
    diameter = checkpoint["diameter"]
    previous_elapsed = checkpoint["elapsed_seconds"]

    print(
        f"Retomando a partir da origem "
        f"{fmt_int(start_source)}/{fmt_int(n)}"
    )

    print(
        f"Diâmetro atual: {diameter}"
    )

else:
    print("Iniciando um novo cálculo exato.")


# Menores caminhos a partir de todos os vértices

run_start = time.perf_counter()

for source_id in range(start_source, n):

    distances = scc.distances(
        source=source_id,
        mode="out",
        weights=None,
    )[0]

    # As distâncias são contagens inteiras de saltos.
    distances = np.asarray(
        distances,
        dtype=np.int16,
    )

    # A distância do vértice até ele mesmo é zero e não altera a soma.
    total_distance += int(
        distances.sum(dtype=np.int64)
    )

    source_max = int(
        distances.max()
    )

    if source_max > diameter:
        diameter = source_max
        print()
        print(
            f"Novo limite inferior para o diâmetro: "
            f"{diameter} "
            f"(origem {fmt_int(source_id)})"
        )

    completed = source_id + 1

    if (
        completed % CHECKPOINT_EVERY == 0
        or completed == n
    ):
        current_run_elapsed = (
            time.perf_counter() - run_start
        )

        total_elapsed = (
            previous_elapsed
            + current_run_elapsed
        )

        mean_seconds_per_source = (
            total_elapsed / completed
        )

        remaining = n - completed

        eta_seconds = (
            remaining
            * mean_seconds_per_source
        )

        print(
            f"\rProcessados: "
            f"{fmt_int(completed)}/{fmt_int(n)} "
            f"({fmt_percent(completed / n, 2)}) | "
            f"diâmetro={diameter} | "
            f"tempo={fmt_float(total_elapsed / 60, 1)} min | "
            f"restante≈{fmt_float(eta_seconds / 60, 1)} min",
            end="",
            flush=True,
        )

        save_checkpoint(
            next_source=completed,
            total_distance=total_distance,
            diameter=diameter,
            elapsed_seconds=total_elapsed,
            n_vertices=n,
            n_edges=m,
        )


print()


# Métricas exatas

section("4. Resultados exatos")

total_elapsed = (
    previous_elapsed
    + time.perf_counter()
    - run_start
)

# Na componente fortemente conexa direcionada existem
# n * (n - 1) pares ordenados de vértices distintos.
#
ordered_pairs = n * (n - 1)

average_distance = (
    total_distance / ordered_pairs
)

print(f"Vértices:          {fmt_int(n)}")
print(f"Arestas:           {fmt_int(m)}")
print(f"Pares ordenados:   {fmt_int(ordered_pairs)}")
print(f"Soma das distâncias: {fmt_int(total_distance)}")
print(f"Distância média:   {fmt_float(average_distance, 10)}")
print(f"Diâmetro:          {diameter}")
print(f"Tempo total:       {fmt_float(total_elapsed / 60, 2)} min")
print(f"Tempo total:       {fmt_float(total_elapsed / 3600, 2)} h")


# Salva os resultados

# As chaves e os valores descritivos do JSON foram mantidos para compatibilidade.
results = {
    "graph": "FlyWire FAFB v783",
    "representation": (
        "largest strongly connected component "
        "of the directed unweighted graph"
    ),
    "distance_definition": (
        "directed unweighted shortest-path hop count"
    ),
    "vertices": n,
    "edges": m,
    "ordered_distinct_vertex_pairs": ordered_pairs,
    "distance_sum": total_distance,
    "average_distance": average_distance,
    "diameter": diameter,
    "runtime_seconds": total_elapsed,
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
FlyWire FAFB v783 - análise exata das distâncias

Representação
Maior componente fortemente conexa do grafo
direcionado e não ponderado do FlyWire.

Vértices: {fmt_int(n)}
Arestas: {fmt_int(m)}

Definição de distância
Número de arestas no menor caminho direcionado.

Como a componente é fortemente conexa, todos os pares
ordenados de vértices distintos são alcançáveis.

Pares ordenados: {fmt_int(ordered_pairs)}

Resultados
Distância média exata: {fmt_float(average_distance, 10)}
Diâmetro exato: {diameter}

Tempo de execução
{fmt_float(total_elapsed / 60, 2)} minutos
({fmt_float(total_elapsed / 3600, 2)} horas)
""".strip()

OUTPUT_TXT.write_text(
    summary,
    encoding="utf-8",
)

section("5. Concluído")

print(summary)

print()
print(f"Relatório salvo em: {OUTPUT_TXT}")
print(f"JSON salvo em:      {OUTPUT_JSON}")


# Remove o checkpoint após a conclusão

if CHECKPOINT_PATH.exists():
    CHECKPOINT_PATH.unlink()

print("Checkpoint removido após a conclusão.")