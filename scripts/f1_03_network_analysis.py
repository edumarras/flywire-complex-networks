from pathlib import Path
import gc
import json
import time

import igraph as ig
import matplotlib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Caminhos

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

REPORT_DIR = ROOT / "reports" / "f1"
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"

NEURONS_PATH = RAW_DIR / "neurons.csv.gz"
EDGES_PATH = PROCESSED_DIR / "flywire_fafb_v783_edges.csv.gz"

SUMMARY_JSON = REPORT_DIR / "network_summary.json"
SUMMARY_TXT = REPORT_DIR / "network_summary.txt"

NODE_METRICS_PATH = TABLE_DIR / "node_metrics.csv.gz"
DEGREE_DIST_PATH = TABLE_DIR / "degree_distribution.csv"
COMPONENTS_PATH = TABLE_DIR / "component_sizes.csv"

DEGREE_FIGURE = FIGURE_DIR / "degree_distribution.png"
CLUSTERING_FIGURE = FIGURE_DIR / "clustering_distribution.png"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# Funções auxiliares

def section(title):
    print()
    print(title)


def fmt(x):
    return f"{int(x):,}".replace(",", ".")


def fmt_float(x, casas=6):
    return f"{x:.{casas}f}".replace(".", ",")


def fmt_percent(x, casas=6):
    return f"{100 * x:.{casas}f}".replace(".", ",") + "%"


# Carrega os nós

section("1. Carregando os nós")

nodes_df = pd.read_csv(
    NEURONS_PATH,
    usecols=["root_id"], # cada root_id representa um nó
    dtype={"root_id": "int64"},
)

if nodes_df["root_id"].duplicated().any():
    raise RuntimeError("Foram encontrados valores de root_id duplicados.")

nodes_df = nodes_df.reset_index(drop=True)

n_nodes = len(nodes_df)

print(f"Vértices: {fmt(n_nodes)}")


# Carrega a lista de arestas agregada

section("2. Carregando a lista de arestas")

edges_df = pd.read_csv(
    EDGES_PATH,
    dtype={
        "pre_root_id": "int64",
        "post_root_id": "int64",
        "weight": "int64",
    },
)

n_edges = len(edges_df)

print(f"Arestas: {fmt(n_edges)}")


# Converte os IDs do FlyWire para índices contínuos do igraph

section("3. Mapeando os IDs do FlyWire")

node_index = pd.Index(nodes_df["root_id"].to_numpy())

source = node_index.get_indexer(
    edges_df["pre_root_id"].to_numpy()
)

target = node_index.get_indexer(
    edges_df["post_root_id"].to_numpy()
)

if (source < 0).any() or (target < 0).any():
    raise RuntimeError(
        "Pelo menos um extremo de aresta não está no conjunto de nós."
    )

graph_edges = pd.DataFrame(
    {
        "source": source,
        "target": target,
        "weight": edges_df["weight"].to_numpy(),
    }
)

vertices = pd.DataFrame(
    {
        "vertex_id": np.arange(n_nodes, dtype=np.int64),
        "root_id": nodes_df["root_id"].to_numpy(),
    }
)

del source, target
gc.collect()


# Constrói o grafo direcionado

section("4. Construindo o grafo direcionado")

start = time.time()

g = ig.Graph.DataFrame(
    graph_edges,
    directed=True,
    vertices=vertices,
    use_vids=True,
)

print(f"Vértices: {fmt(g.vcount())}")
print(f"Arestas:  {fmt(g.ecount())}")
print(f"Simples:  {g.is_simple()}")
print(f"Tempo:    {fmt_float(time.time() - start, 2)} s")

if g.vcount() != n_nodes:
    raise RuntimeError("O número de vértices não corresponde ao esperado.")

if g.ecount() != n_edges:
    raise RuntimeError("O número de arestas não corresponde ao esperado.")

if not g.is_simple():
    raise RuntimeError(
        "O grafo direcionado contém autolaços ou "
        "arestas paralelas."
    )

# Os dataframes originais não são mais necessários.
del graph_edges
del edges_df
gc.collect()


# Grau e força no grafo direcionado

section("5. Grau e força")

in_degree = np.asarray(g.degree(mode="in"), dtype=np.int64)
out_degree = np.asarray(g.degree(mode="out"), dtype=np.int64)
total_degree = in_degree + out_degree

in_strength = np.asarray(
    g.strength(mode="in", weights="weight"),
    dtype=np.float64,
)

out_strength = np.asarray(
    g.strength(mode="out", weights="weight"),
    dtype=np.float64,
)

total_strength = in_strength + out_strength

mean_in_degree = float(in_degree.mean())
mean_out_degree = float(out_degree.mean())
mean_total_degree = float(total_degree.mean())

density_directed = float(g.density(loops=False))

isolated = total_degree == 0
n_isolated = int(isolated.sum())

reciprocity = float(
    g.reciprocity(ignore_loops=True)
)

print(f"Grau médio de entrada: {fmt_float(mean_in_degree, 6)}")
print(f"Grau médio de saída:   {fmt_float(mean_out_degree, 6)}")
print(f"Grau médio total:      {fmt_float(mean_total_degree, 6)}")
print(f"Densidade:             {fmt_float(density_directed, 12)}")
print(f"Vértices isolados:     {fmt(n_isolated)}")
print(f"Reciprocidade:         {fmt_float(reciprocity, 6)}")


# Componentes

section("6. Componentes")

start = time.time()

weak_components = g.connected_components(mode="weak")
weak_sizes = np.asarray(weak_components.sizes(), dtype=np.int64)

strong_components = g.connected_components(mode="strong")
strong_sizes = np.asarray(strong_components.sizes(), dtype=np.int64)

n_weak = len(weak_sizes)
n_strong = len(strong_sizes)

largest_weak = int(weak_sizes.max())
largest_strong = int(strong_sizes.max())

print(f"Componentes fracas:             {fmt(n_weak)}")
print(f"Maior componente fraca:         {fmt(largest_weak)}")
print(
    f"Fração na maior componente fraca: "
    f"{fmt_percent(largest_weak / n_nodes, 6)}"
)

print(f"Componentes fortes:             {fmt(n_strong)}")
print(f"Maior componente forte:         {fmt(largest_strong)}")
print(
    f"Fração na maior componente forte: "
    f"{fmt_percent(largest_strong / n_nodes, 6)}"
)

print(
    f"Tempo da análise de componentes: "
    f"{fmt_float(time.time() - start, 2)} s"
)


component_df = pd.concat(
    [
        pd.DataFrame(
            {
                "type": "weak",
                "size": weak_sizes,
            }
        ),
        pd.DataFrame(
            {
                "type": "strong",
                "size": strong_sizes,
            }
        ),
    ],
    ignore_index=True,
)

component_df.to_csv(
    COMPONENTS_PATH,
    index=False,
)


# Projeção não direcionada
# Para os coeficientes de clusterização, usamos um grafo simples
# e não direcionado, como definido na disciplina.
# A -> B ou B -> A passa a ser A -- B.
# Arestas recíprocas são unidas e os pesos são somados.
# Os coeficientes de clusterização abaixo não usam os pesos.

section("7. Projeção não direcionada")

start = time.time()

u = g.as_undirected(
    mode="collapse",
    combine_edges="sum",
)

print(f"Vértices: {fmt(u.vcount())}")
print(f"Arestas:  {fmt(u.ecount())}")
print(f"Simples:  {u.is_simple()}")
print(f"Tempo:    {fmt_float(time.time() - start, 2)} s")

if not u.is_simple():
    raise RuntimeError(
        "A projeção não direcionada deveria ser um grafo simples."
    )

undirected_edges = u.ecount()
undirected_density = float(u.density(loops=False))

# O grafo direcionado não é mais necessário a partir daqui.
del g
del weak_components
del strong_components
gc.collect()


# Clusterização local

section("8. Coeficientes de clusterização local")

start = time.time()

undirected_degree = np.asarray(
    u.degree(),
    dtype=np.int64,
)

local_clustering = np.asarray(
    u.transitivity_local_undirected(
        mode="zero"
    ),
    dtype=np.float64,
)

# Convenção usada na disciplina:
# para grau 0 ou 1, o coeficiente local pode ser representado por zero,
# mas esses vértices não entram no cálculo da média da rede.

eligible_clustering = undirected_degree >= 2

n_clustering_eligible = int(
    eligible_clustering.sum()
)

mean_local_clustering = float(
    local_clustering[eligible_clustering].mean()
)

print(
    f"Vértices com grau >= 2: "
    f"{fmt(n_clustering_eligible)}"
)

print(
    f"Clusterização local média: "
    f"{fmt_float(mean_local_clustering, 8)}"
)

print(
    f"Tempo da clusterização local: "
    f"{fmt_float(time.time() - start, 2)} s"
)


# Clusterização global: transitividade

section("9. Transitividade global")

start = time.time()

global_transitivity = float(
    u.transitivity_undirected(
        mode="zero"
    )
)

print(
    f"Transitividade global: "
    f"{fmt_float(global_transitivity, 8)}"
)

print(
    f"Tempo da transitividade: "
    f"{fmt_float(time.time() - start, 2)} s"
)


# Métricas por nó

section("10. Salvando as métricas dos nós")

node_metrics = pd.DataFrame(
    {
        "root_id": nodes_df["root_id"].to_numpy(),
        "in_degree": in_degree,
        "out_degree": out_degree,
        "total_degree": total_degree,
        "in_strength": in_strength,
        "out_strength": out_strength,
        "total_strength": total_strength,
        "undirected_degree": undirected_degree,
        "local_clustering": local_clustering,
        "isolated": isolated,
    }
)

node_metrics.to_csv(
    NODE_METRICS_PATH,
    index=False,
    compression="gzip",
)

print(f"Salvo em: {NODE_METRICS_PATH}")


# Tabela da distribuição de graus

section("11. Distribuição de graus")

def degree_counts(values, name):
    return (
        pd.Series(values)
        .value_counts()
        .sort_index()
        .rename(name)
    )


degree_distribution = pd.concat(
    [
        degree_counts(in_degree, "in_count"),
        degree_counts(out_degree, "out_count"),
        degree_counts(total_degree, "total_count"),
    ],
    axis=1,
).fillna(0)

degree_distribution.index.name = "degree"

degree_distribution = degree_distribution.astype(
    {
        "in_count": "int64",
        "out_count": "int64",
        "total_count": "int64",
    }
)

degree_distribution["in_probability"] = (
    degree_distribution["in_count"] / n_nodes
)

degree_distribution["out_probability"] = (
    degree_distribution["out_count"] / n_nodes
)

degree_distribution["total_probability"] = (
    degree_distribution["total_count"] / n_nodes
)

degree_distribution.reset_index().to_csv(
    DEGREE_DIST_PATH,
    index=False,
)

print(f"Salvo em: {DEGREE_DIST_PATH}")

# Salva o resumo antes de gerar as figuras
# As chaves do JSON e das tabelas continuam em inglês para não quebrar os próximos scripts.

section("11b. Salvando o resumo da rede")

summary = {
    "graph": "FlyWire FAFB v783",

    "representation": {
        "directed": True,
        "weighted": True,
        "vertices": int(n_nodes),
        "directed_edges": int(n_edges),
        "isolated_vertices": n_isolated,
    },

    "degree": {
        "mean_in_degree": mean_in_degree,
        "mean_out_degree": mean_out_degree,
        "mean_total_degree": mean_total_degree,
    },

    "density": {
        "directed": density_directed,
        "undirected_projection": undirected_density,
    },

    "reciprocity": reciprocity,

    "components": {
        "weak_count": int(n_weak),
        "largest_weak": largest_weak,
        "largest_weak_fraction": largest_weak / n_nodes,

        "strong_count": int(n_strong),
        "largest_strong": largest_strong,
        "largest_strong_fraction": largest_strong / n_nodes,
    },

    "undirected_projection": {
        "edges": int(undirected_edges),
    },

    "clustering": {
        "convention": (
            "simple unweighted undirected projection"
        ),
        "eligible_vertices_degree_ge_2": (
            n_clustering_eligible
        ),
        "mean_local_clustering": (
            mean_local_clustering
        ),
        "global_transitivity": (
            global_transitivity
        ),
    },

    "distances": {
        "average_distance": None,
        "diameter": None,
        "status": "computed separately",
    },
}

with SUMMARY_JSON.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False,
    )


summary_text = f"""
FlyWire FAFB v783 - análise inicial da rede

Grafo direcionado
Vértices: {fmt(n_nodes)}
Arestas direcionadas: {fmt(n_edges)}
Vértices isolados: {fmt(n_isolated)}

Graus
Grau médio de entrada: {fmt_float(mean_in_degree, 6)}
Grau médio de saída: {fmt_float(mean_out_degree, 6)}
Grau médio total: {fmt_float(mean_total_degree, 6)}

Densidade
Densidade direcionada: {fmt_float(density_directed, 12)}

Reciprocidade
{fmt_float(reciprocity, 8)}

Componentes
Componentes fracas: {fmt(n_weak)}
Maior componente fraca: {fmt(largest_weak)}
Fração na maior componente fraca: {fmt_percent(largest_weak / n_nodes, 6)}

Componentes fortes: {fmt(n_strong)}
Maior componente forte: {fmt(largest_strong)}
Fração na maior componente forte: {fmt_percent(largest_strong / n_nodes, 6)}

Projeção não direcionada
Arestas: {fmt(undirected_edges)}
Densidade: {fmt_float(undirected_density, 12)}

Clusterização
Vértices com grau >= 2: {fmt(n_clustering_eligible)}
Clusterização local média: {fmt_float(mean_local_clustering, 8)}
Transitividade global: {fmt_float(global_transitivity, 8)}
""".strip()

SUMMARY_TXT.write_text(
    summary_text,
    encoding="utf-8",
)

print(f"Salvo em: {SUMMARY_JSON}")
print(f"Salvo em: {SUMMARY_TXT}")

# Figura da distribuição de graus

section("12. Gerando a distribuição de graus")

fig, ax = plt.subplots(figsize=(7, 5))

degree_values = degree_distribution.index.to_numpy()

for probability_column, label in [
    ("in_probability", "Grau de entrada"),
    ("out_probability", "Grau de saída"),
    ("total_probability", "Grau total"),
]:
    probability = (
        degree_distribution[
            probability_column
        ].to_numpy()
    )

    mask = (
        (degree_values > 0)
        & (probability > 0)
    )

    ax.loglog(
        degree_values[mask],
        probability[mask],
        marker=".",
        linestyle="none",
        markersize=3,
        alpha=0.7,
        label=label,
    )

ax.set_xlabel("Grau")
ax.set_ylabel("Probabilidade")
ax.set_title("Distribuição dos graus")
ax.legend()
ax.grid(
    True,
    which="both",
    alpha=0.2,
)

fig.tight_layout()

fig.savefig(
    DEGREE_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Salvo em: {DEGREE_FIGURE}")


# Figura da distribuição da clusterização

section("13. Gerando a distribuição da clusterização local")

fig, ax = plt.subplots(figsize=(7, 5))

bins = np.linspace(
    0.0,
    1.0,
    51,
)

ax.hist(
    local_clustering,
    bins=bins,
)

ax.set_xlabel("Coeficiente de clusterização local")
ax.set_ylabel("Número de vértices")
ax.set_title(
    "Distribuição dos coeficientes de clusterização local"
)

# Escala logarítmica para facilitar a visualização
# da concentração de valores próximos de zero.
ax.set_yscale("log")

fig.tight_layout()

fig.savefig(
    CLUSTERING_FIGURE,
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)

print(f"Salvo em: {CLUSTERING_FIGURE}")


# Resumo

section("14. Resumo")

summary = {
    "graph": "FlyWire FAFB v783",

    "representation": {
        "directed": True,
        "weighted": True,
        "vertices": int(n_nodes),
        "directed_edges": int(n_edges),
        "isolated_vertices": n_isolated,
    },

    "degree": {
        "mean_in_degree": mean_in_degree,
        "mean_out_degree": mean_out_degree,
        "mean_total_degree": mean_total_degree,
    },

    "density": {
        "directed": density_directed,
        "undirected_projection": undirected_density,
    },

    "reciprocity": reciprocity,

    "components": {
        "weak_count": int(n_weak),
        "largest_weak": largest_weak,
        "largest_weak_fraction": (
            largest_weak / n_nodes
        ),
        "strong_count": int(n_strong),
        "largest_strong": largest_strong,
        "largest_strong_fraction": (
            largest_strong / n_nodes
        ),
    },

    "undirected_projection": {
        "edges": int(undirected_edges),
    },

    "clustering": {
        "convention": (
            "simple unweighted undirected projection"
        ),
        "eligible_vertices_degree_ge_2": (
            n_clustering_eligible
        ),
        "mean_local_clustering": (
            mean_local_clustering
        ),
        "global_transitivity": (
            global_transitivity
        ),
    },

    "distances": {
        "average_distance": None,
        "diameter": None,
        "status": (
            "pending component-aware analysis"
        ),
    },
}

with SUMMARY_JSON.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False,
    )


summary_text = f"""
FlyWire FAFB v783 - análise inicial da rede

Grafo direcionado
Vértices: {fmt(n_nodes)}
Arestas direcionadas: {fmt(n_edges)}
Vértices isolados: {fmt(n_isolated)}

Graus
Grau médio de entrada: {fmt_float(mean_in_degree, 6)}
Grau médio de saída: {fmt_float(mean_out_degree, 6)}
Grau médio total: {fmt_float(mean_total_degree, 6)}

Densidade
Densidade direcionada: {fmt_float(density_directed, 12)}

Reciprocidade
{fmt_float(reciprocity, 8)}

Componentes
Componentes fracas: {fmt(n_weak)}
Maior componente fraca: {fmt(largest_weak)}
Fração na maior componente fraca: {fmt_percent(largest_weak / n_nodes, 6)}

Componentes fortes: {fmt(n_strong)}
Maior componente forte: {fmt(largest_strong)}
Fração na maior componente forte: {fmt_percent(largest_strong / n_nodes, 6)}

Projeção não direcionada
Arestas: {fmt(undirected_edges)}
Densidade: {fmt_float(undirected_density, 12)}

Clusterização
Projeção simples, não ponderada e não direcionada.
Vértices com grau >= 2: {fmt(n_clustering_eligible)}
Coeficiente local médio: {fmt_float(mean_local_clustering, 8)}
Transitividade global: {fmt_float(global_transitivity, 8)}

Distâncias
Distância média: PENDENTE
Diâmetro: PENDENTE
""".strip()

SUMMARY_TXT.write_text(
    summary_text,
    encoding="utf-8",
)

print(summary_text)

print()
print(f"Resumo: {SUMMARY_TXT}")
print(f"JSON:    {SUMMARY_JSON}")


# Limpeza

del u
gc.collect()

section("Concluído")