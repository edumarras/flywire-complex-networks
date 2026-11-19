from pathlib import Path
import gc
import json
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
F1_TABLE_DIR = ROOT / "reports" / "f1" / "tables"

REPORT_DIR = ROOT / "reports" / "f2"
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"

EDGES_PATH = PROCESSED_DIR / "flywire_fafb_v783_edges.csv.gz"
F1_NODE_METRICS_PATH = F1_TABLE_DIR / "node_metrics.csv.gz"

NODE_STRUCTURAL_PATH = TABLE_DIR / "node_structural_features.csv.gz"
STRUCTURAL_SUMMARY_PATH = TABLE_DIR / "structural_summary.csv"
EDGE_WEIGHT_DIST_PATH = TABLE_DIR / "edge_weight_distribution.csv"
CORENESS_DIST_PATH = TABLE_DIR / "coreness_distribution.csv"
TOP_HUBS_PATH = TABLE_DIR / "top_hubs.csv"
SUMMARY_JSON_PATH = REPORT_DIR / "structural_metrics_summary.json"
SUMMARY_TXT_PATH = REPORT_DIR / "structural_metrics_summary.txt"

DEGREE_CCDF_FIGURE = FIGURE_DIR / "degree_ccdf.png"
STRENGTH_CCDF_FIGURE = FIGURE_DIR / "strength_ccdf.png"
EDGE_WEIGHT_CCDF_FIGURE = FIGURE_DIR / "edge_weight_ccdf.png"
CORENESS_FIGURE = FIGURE_DIR / "coreness_distribution.png"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Funções auxiliares
# ============================================================

def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def fmt_int(x):
    return f"{int(x):,}".replace(",", ".")


def fmt_float(x, digits=6):
    return f"{float(x):.{digits}f}".replace(".", ",")


def elapsed(start):
    return f"{time.time() - start:.2f} s"


def metric_summary(values, metric):
    x = np.asarray(values)
    x = x[np.isfinite(x)]

    if x.size == 0:
        raise RuntimeError(f"A métrica {metric} não possui valores finitos.")

    q = np.quantile(x, [0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

    return {
        "metric": metric,
        "n": int(x.size),
        "zeros": int(np.sum(x == 0)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "p25": float(q[0]),
        "median": float(q[1]),
        "p75": float(q[2]),
        "p90": float(q[3]),
        "p95": float(q[4]),
        "p99": float(q[5]),
        "max": float(np.max(x)),
    }


def discrete_distribution(values, value_name):
    s = pd.Series(np.asarray(values))
    counts = s.value_counts().sort_index()
    total = int(counts.sum())

    out = counts.rename("count").rename_axis(value_name).reset_index()
    out["probability"] = out["count"] / total
    out["ccdf"] = out["count"][::-1].cumsum()[::-1] / total
    return out


def add_ccdf_trace(fig, values, name):
    dist = discrete_distribution(values, "value")
    dist = dist[(dist["value"] > 0) & (dist["ccdf"] > 0)]

    fig.add_trace(
        go.Scatter(
            x=dist["value"],
            y=dist["ccdf"],
            mode="markers",
            name=name,
            marker={"size": 5},
            hovertemplate=(
                "Valor: %{x}<br>"
                "P(X ≥ x): %{y:.6g}"
                "<extra></extra>"
            ),
        )
    )


def save_ccdf_figure(series_list, title, x_title, output_path):
    fig = go.Figure()

    for values, label in series_list:
        add_ccdf_trace(fig, values, label)

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title="CCDF — P(X ≥ x)",
        xaxis_type="log",
        yaxis_type="log",
        template="simple_white",
        width=1000,
        height=700,
    )

    fig.write_image(output_path, scale=2)


# ============================================================
# 1. Carrega métricas já calculadas na F1
# ============================================================

section("1. Carregando métricas por nó da F1")

required_node_columns = {
    "root_id",
    "in_degree",
    "out_degree",
    "total_degree",
    "in_strength",
    "out_strength",
    "total_strength",
    "undirected_degree",
    "local_clustering",
    "isolated",
}

node = pd.read_csv(F1_NODE_METRICS_PATH)

missing = required_node_columns.difference(node.columns)
if missing:
    raise RuntimeError(
        "node_metrics.csv.gz não possui todas as colunas esperadas. "
        f"Faltando: {sorted(missing)}"
    )

if node["root_id"].duplicated().any():
    raise RuntimeError("Há root_id duplicado em node_metrics.csv.gz.")

node = node.reset_index(drop=True)
n_nodes = len(node)

# Força é soma de pesos inteiros. Mantemos como int64 na tabela F2.
for col in ["in_strength", "out_strength", "total_strength"]:
    rounded = np.rint(node[col].to_numpy(dtype=np.float64))
    if not np.allclose(node[col].to_numpy(dtype=np.float64), rounded):
        raise RuntimeError(f"A coluna {col} possui valores não inteiros inesperados.")
    node[col] = rounded.astype(np.int64)

print(f"Nós: {fmt_int(n_nodes)}")
print(f"Isolados: {fmt_int(node['isolated'].sum())}")


# ============================================================
# 2. Carrega a lista de arestas agregada
# ============================================================

section("2. Carregando arestas agregadas")

start = time.time()

edges = pd.read_csv(
    EDGES_PATH,
    dtype={
        "pre_root_id": "int64",
        "post_root_id": "int64",
        "weight": "int32",
    },
)

required_edge_columns = {"pre_root_id", "post_root_id", "weight"}
missing = required_edge_columns.difference(edges.columns)
if missing:
    raise RuntimeError(
        "A tabela de arestas não possui todas as colunas esperadas. "
        f"Faltando: {sorted(missing)}"
    )

n_edges = len(edges)
weights = edges["weight"].to_numpy(dtype=np.int64)

if np.any(weights <= 0):
    raise RuntimeError("Foram encontrados pesos de aresta não positivos.")

print(f"Arestas dirigidas: {fmt_int(n_edges)}")
print(f"Peso mínimo:       {fmt_int(weights.min())}")
print(f"Peso máximo:       {fmt_int(weights.max())}")
print(f"Sinapses totais:   {fmt_int(weights.sum())}")
print(f"Tempo:             {elapsed(start)}")


# ============================================================
# 3. Distribuição dos pesos
# ============================================================

section("3. Distribuição dos pesos das arestas")

edge_weight_dist = discrete_distribution(weights, "weight")
edge_weight_dist.to_csv(EDGE_WEIGHT_DIST_PATH, index=False)

fraction_weight_5 = float(np.mean(weights == 5))
fraction_weight_le_10 = float(np.mean(weights <= 10))

print(f"Peso = 5:   {100 * fraction_weight_5:.2f}% das arestas")
print(f"Peso <= 10: {100 * fraction_weight_le_10:.2f}% das arestas")
print(f"Tabela salva em: {EDGE_WEIGHT_DIST_PATH}")


# ============================================================
# 4. Constrói o grafo dirigido
# ============================================================

section("4. Construindo grafo dirigido")

start = time.time()

node_index = pd.Index(node["root_id"].to_numpy(dtype=np.int64))
source = node_index.get_indexer(edges["pre_root_id"].to_numpy(dtype=np.int64))
target = node_index.get_indexer(edges["post_root_id"].to_numpy(dtype=np.int64))

if (source < 0).any() or (target < 0).any():
    raise RuntimeError(
        "Pelo menos um extremo de aresta não está presente em node_metrics.csv.gz."
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
        "vertex_id": np.arange(n_nodes, dtype=np.int64),
        "root_id": node["root_id"].to_numpy(dtype=np.int64),
    }
)

g = ig.Graph.DataFrame(
    graph_edges,
    directed=True,
    vertices=vertices,
    use_vids=True,
)

if g.vcount() != n_nodes:
    raise RuntimeError("Número inesperado de vértices no igraph.")
if g.ecount() != n_edges:
    raise RuntimeError("Número inesperado de arestas no igraph.")
if not g.is_simple():
    raise RuntimeError("O grafo dirigido deveria ser simples após a agregação.")

print(f"Vértices: {fmt_int(g.vcount())}")
print(f"Arestas:  {fmt_int(g.ecount())}")
print(f"Tempo:    {elapsed(start)}")

# A tabela original de arestas já não é necessária.
del source, target, graph_edges, vertices, edges
gc.collect()


# ============================================================
# 5. PageRank ponderado no grafo dirigido
# ============================================================

section("5. PageRank ponderado")

start = time.time()

# Em PageRank, os pesos representam intensidade da conexão:
# arestas com mais sinapses recebem maior peso na transição.
pagerank_weighted = np.asarray(
    g.pagerank(
        directed=True,
        damping=0.85,
        weights="weight",
    ),
    dtype=np.float64,
)

if not np.isfinite(pagerank_weighted).all():
    raise RuntimeError("PageRank produziu valores não finitos.")

if not np.isclose(pagerank_weighted.sum(), 1.0, rtol=1e-7, atol=1e-10):
    raise RuntimeError(
        "A soma do PageRank ponderado não ficou aproximadamente igual a 1."
    )

node["pagerank_weighted"] = pagerank_weighted

print(f"Soma:    {pagerank_weighted.sum():.12f}")
print(f"Máximo:  {pagerank_weighted.max():.12g}")
print(f"Tempo:   {elapsed(start)}")


# ============================================================
# 6. Projeção não dirigida e coreness (k-core)
# ============================================================

section("6. Coreness na projeção não dirigida")

start = time.time()

# Mesma convenção usada na F1 para a projeção:
# A -> B ou B -> A passa a ser A -- B; reciprocidade é colapsada.
u = g.as_undirected(
    mode="collapse",
    combine_edges="sum",
)

if not u.is_simple():
    raise RuntimeError("A projeção não dirigida deveria ser simples.")

coreness = np.asarray(u.coreness(), dtype=np.int64)

if np.any(coreness < 0):
    raise RuntimeError("Foram encontrados valores negativos de coreness.")

node["coreness"] = coreness

coreness_dist = discrete_distribution(coreness, "coreness")
coreness_dist.to_csv(CORENESS_DIST_PATH, index=False)

print(f"Arestas na projeção: {fmt_int(u.ecount())}")
print(f"k-core máximo:       {fmt_int(coreness.max())}")
print(f"Tempo:                {elapsed(start)}")

# Não precisamos mais manter os dois grafos em memória.
del g, u
gc.collect()


# ============================================================
# 7. Resumo estatístico
# ============================================================

section("7. Resumo estatístico")

summary_metrics = [
    ("in_degree", node["in_degree"].to_numpy()),
    ("out_degree", node["out_degree"].to_numpy()),
    ("total_degree", node["total_degree"].to_numpy()),
    ("in_strength", node["in_strength"].to_numpy()),
    ("out_strength", node["out_strength"].to_numpy()),
    ("total_strength", node["total_strength"].to_numpy()),
    ("undirected_degree", node["undirected_degree"].to_numpy()),
    ("local_clustering", node["local_clustering"].to_numpy()),
    ("pagerank_weighted", node["pagerank_weighted"].to_numpy()),
    ("coreness", node["coreness"].to_numpy()),
    ("edge_weight", weights),
]

summary_rows = [metric_summary(values, name) for name, values in summary_metrics]
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(STRUCTURAL_SUMMARY_PATH, index=False)

print(
    summary_df[
        ["metric", "mean", "median", "p90", "p95", "p99", "max"]
    ].to_string(index=False)
)
print(f"Tabela salva em: {STRUCTURAL_SUMMARY_PATH}")


# ============================================================
# 8. Rankings de hubs
# ============================================================

section("8. Ranking dos principais hubs")

hub_metrics = [
    "in_degree",
    "out_degree",
    "total_degree",
    "in_strength",
    "out_strength",
    "total_strength",
    "pagerank_weighted",
    "coreness",
]

hub_tables = []

for metric in hub_metrics:
    top = (
        node[["root_id", metric]]
        .nlargest(25, metric)
        .reset_index(drop=True)
    )
    top.insert(0, "metric", metric)
    top.insert(1, "rank", np.arange(1, len(top) + 1, dtype=np.int64))
    top = top.rename(columns={metric: "value"})
    hub_tables.append(top)

hubs = pd.concat(hub_tables, ignore_index=True)
hubs.to_csv(TOP_HUBS_PATH, index=False)

print(f"Tabela salva em: {TOP_HUBS_PATH}")


# ============================================================
# 9. Salva tabela consolidada por neurônio
# ============================================================

section("9. Salvando tabela estrutural por neurônio")

feature_columns = [
    "root_id",
    "in_degree",
    "out_degree",
    "total_degree",
    "in_strength",
    "out_strength",
    "total_strength",
    "undirected_degree",
    "local_clustering",
    "pagerank_weighted",
    "coreness",
    "isolated",
]

node_structural = node[feature_columns].copy()

if node_structural["root_id"].duplicated().any():
    raise RuntimeError("Há root_id duplicado na tabela estrutural final.")

if len(node_structural) != n_nodes:
    raise RuntimeError("A tabela estrutural final perdeu neurônios.")

numeric_cols = [
    c for c in feature_columns
    if c not in {"root_id", "isolated"}
]

if not np.isfinite(node_structural[numeric_cols].to_numpy(dtype=np.float64)).all():
    raise RuntimeError("A tabela estrutural final contém NaN ou infinito.")

node_structural.to_csv(
    NODE_STRUCTURAL_PATH,
    index=False,
    compression="gzip",
)

print(f"Linhas: {fmt_int(len(node_structural))}")
print(f"Salvo em: {NODE_STRUCTURAL_PATH}")


# ============================================================
# 10. Figuras CCDF e coreness
# ============================================================

section("10. Gerando figuras")

save_ccdf_figure(
    [
        (node["in_degree"].to_numpy(), "Grau de entrada"),
        (node["out_degree"].to_numpy(), "Grau de saída"),
        (node["total_degree"].to_numpy(), "Grau total"),
    ],
    title="CCDF dos graus",
    x_title="Grau",
    output_path=DEGREE_CCDF_FIGURE,
)
print(f"Salvo em: {DEGREE_CCDF_FIGURE}")

save_ccdf_figure(
    [
        (node["in_strength"].to_numpy(), "Força de entrada"),
        (node["out_strength"].to_numpy(), "Força de saída"),
        (node["total_strength"].to_numpy(), "Força total"),
    ],
    title="CCDF das forças",
    x_title="Força (número de sinapses)",
    output_path=STRENGTH_CCDF_FIGURE,
)
print(f"Salvo em: {STRENGTH_CCDF_FIGURE}")

save_ccdf_figure(
    [(weights, "Peso da aresta")],
    title="CCDF dos pesos das arestas",
    x_title="Peso da aresta (número de sinapses)",
    output_path=EDGE_WEIGHT_CCDF_FIGURE,
)
print(f"Salvo em: {EDGE_WEIGHT_CCDF_FIGURE}")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=coreness_dist["coreness"],
        y=coreness_dist["count"],
        mode="markers",
        name="Nós",
        marker={"size": 6},
        hovertemplate=(
            "Coreness: %{x}<br>"
            "Número de nós: %{y}"
            "<extra></extra>"
        ),
    )
)
fig.update_layout(
    title="Distribuição de coreness (k-core)",
    xaxis_title="Coreness",
    yaxis_title="Número de nós",
    yaxis_type="log",
    template="simple_white",
    width=1000,
    height=700,
    showlegend=False,
)
fig.write_image(CORENESS_FIGURE, scale=2)
print(f"Salvo em: {CORENESS_FIGURE}")


# ============================================================
# 11. Resumo persistente
# ============================================================

section("11. Salvando resumo")

summary = {
    "graph": "FlyWire FAFB v783",
    "inputs": {
        "node_metrics": "reports/f1/tables/node_metrics.csv.gz",
        "edges": "data/processed/flywire_fafb_v783_edges.csv.gz",
    },
    "graph_size": {
        "nodes": int(n_nodes),
        "directed_edges": int(n_edges),
        "isolated_nodes": int(node["isolated"].sum()),
    },
    "edge_weights": {
        "min": int(weights.min()),
        "max": int(weights.max()),
        "mean": float(weights.mean()),
        "median": float(np.median(weights)),
        "total_synapses": int(weights.sum()),
        "fraction_equal_5": fraction_weight_5,
        "fraction_le_10": fraction_weight_le_10,
    },
    "pagerank": {
        "directed": True,
        "weighted": True,
        "damping": 0.85,
        "sum": float(pagerank_weighted.sum()),
        "max": float(pagerank_weighted.max()),
    },
    "coreness": {
        "representation": "simple undirected projection",
        "max": int(coreness.max()),
    },
    "outputs": {
        "node_structural_features": "reports/f2/tables/node_structural_features.csv.gz",
        "structural_summary": "reports/f2/tables/structural_summary.csv",
        "top_hubs": "reports/f2/tables/top_hubs.csv",
    },
}

with SUMMARY_JSON_PATH.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

summary_text = f"""
FlyWire FAFB v783 — métricas estruturais F2

Nós: {fmt_int(n_nodes)}
Arestas dirigidas: {fmt_int(n_edges)}
Isolados: {fmt_int(node['isolated'].sum())}

Pesos das arestas
Mínimo: {fmt_int(weights.min())}
Máximo: {fmt_int(weights.max())}
Média: {fmt_float(weights.mean(), 4)}
Mediana: {fmt_float(np.median(weights), 4)}
Sinapses totais: {fmt_int(weights.sum())}
Peso exatamente 5: {100 * fraction_weight_5:.2f}%
Peso <= 10: {100 * fraction_weight_le_10:.2f}%

PageRank ponderado
Direcionado: sim
Damping: 0,85
Soma: {pagerank_weighted.sum():.12f}
Máximo: {pagerank_weighted.max():.12g}

Coreness
Representação: projeção simples não dirigida
k-core máximo: {fmt_int(coreness.max())}
""".strip()

SUMMARY_TXT_PATH.write_text(summary_text, encoding="utf-8")

print(f"Salvo em: {SUMMARY_JSON_PATH}")
print(f"Salvo em: {SUMMARY_TXT_PATH}")

print()
print("Concluído.")
