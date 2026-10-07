from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = ROOT / "reports" / "f1"
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"

DEGREE_PATH = TABLE_DIR / "degree_distribution.csv"
NODE_METRICS_PATH = TABLE_DIR / "node_metrics.csv.gz"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# 1. Distribuição dos graus

print("Carregando a distribuição dos graus...")

degree = pd.read_csv(DEGREE_PATH)

fig = go.Figure()

for column, label in [
    ("in_probability", "Grau de entrada"),
    ("out_probability", "Grau de saída"),
    ("total_probability", "Grau total"),
]:
    subset = degree[
        (degree["degree"] > 0)
        & (degree[column] > 0)
    ]

    fig.add_trace(
        go.Scatter(
            x=subset["degree"],
            y=subset[column],
            mode="markers",
            name=label,
            marker={"size": 5},
        )
    )

fig.update_layout(
    title="Distribuição dos graus",
    xaxis_title="Grau",
    yaxis_title="Probabilidade",
    xaxis_type="log",
    yaxis_type="log",
    template="simple_white",
    width=900,
    height=650,
)

degree_png = FIGURE_DIR / "degree_distribution.png"

fig.write_image(
    degree_png,
    scale=2,
)

print(f"Figura salva em: {degree_png}")


# 2. Distribuição da clusterização local

print("Carregando as métricas dos nós...")

nodes = pd.read_csv(
    NODE_METRICS_PATH,
    usecols=["local_clustering"],
)

fig = go.Figure()

fig.add_trace(
    go.Histogram(
        x=nodes["local_clustering"],
        nbinsx=50,
    )
)

fig.update_layout(
    title="Distribuição dos coeficientes de clusterização local",
    xaxis_title="Coeficiente de clusterização local",
    yaxis_title="Número de vértices",
    yaxis_type="log",
    template="simple_white",
    width=900,
    height=650,
    showlegend=False,
)

clustering_png = (
    FIGURE_DIR
    / "clustering_distribution.png"
)

fig.write_image(
    clustering_png,
    scale=2,
)

print(f"Figura salva em: {clustering_png}")

print()
print("Concluído.")