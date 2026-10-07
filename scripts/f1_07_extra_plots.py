from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


# Caminhos

ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = ROOT / "reports" / "f1"
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"

COMPONENTS_PATH = TABLE_DIR / "component_sizes.csv"
OUTPUT_PATH = FIGURE_DIR / "component_size_distribution.png"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def fmt_int(x):
    return f"{int(x):,}".replace(",", ".")



# Carrega os tamanhos das componentes

print("1. Carregando os tamanhos das componentes")

components = pd.read_csv(COMPONENTS_PATH)

required_columns = {"type", "size"}

if not required_columns.issubset(components.columns):
    raise RuntimeError(
        f"Colunas esperadas: {sorted(required_columns)}. "
        f"Colunas encontradas: {components.columns.tolist()}."
    )


# Componentes fracamente conexas

print()
print("2. Preparando a distribuição dos tamanhos")

weak_sizes = (
    components.loc[
        components["type"] == "weak",
        "size",
    ]
    .astype("int64")
)

if weak_sizes.empty:
    raise RuntimeError(
        "Nenhuma componente fracamente conexa foi encontrada em component_sizes.csv."
    )

distribution = (
    weak_sizes
    .value_counts()
    .sort_index()
    .rename_axis("component_size")
    .reset_index(name="component_count")
)

print(f"Componentes fracas:      {fmt_int(len(weak_sizes))}")
print(f"Menor componente:        {fmt_int(weak_sizes.min())}")
print(f"Maior componente:        {fmt_int(weak_sizes.max())}")
print(f"Tamanhos diferentes:     {fmt_int(len(distribution))}")


# Gráfico

print()
print("3. Gerando o gráfico")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=distribution["component_size"],
        y=distribution["component_count"],
        mode="markers",
        name="Componentes fracas",
        hovertemplate=(
            "Tamanho da componente: %{x}<br>"
            "Número de componentes: %{y}"
            "<extra></extra>"
        ),
    )
)

fig.update_layout(
    title="Distribuição dos tamanhos das componentes fracamente conexas",
    xaxis_title="Tamanho da componente (número de vértices)",
    yaxis_title="Número de componentes",
    template="plotly_white",
    width=1400,
    height=900,
)

fig.update_xaxes(type="log")
fig.update_yaxes(type="log")

fig.write_image(
    OUTPUT_PATH,
    scale=2,
)

print(f"Figura salva em: {OUTPUT_PATH}")
print()
print("Concluído.")
