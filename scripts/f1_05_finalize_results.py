from pathlib import Path
import json

import pandas as pd


# Caminhos

ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = ROOT / "reports" / "f1"
TABLE_DIR = REPORT_DIR / "tables"

GRAPH_BUILD = REPORT_DIR / "graph_build.json"
NETWORK_SUMMARY = REPORT_DIR / "network_summary.json"
DISTANCE_EXACT = REPORT_DIR / "distance_exact.json"

FINAL_JSON = REPORT_DIR / "f1_final_results.json"
FINAL_TXT = REPORT_DIR / "f1_final_results.txt"
FINAL_CSV = TABLE_DIR / "f1_metrics.csv"

TABLE_DIR.mkdir(parents=True, exist_ok=True)


# Funções auxiliares

def load_json(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def section(title):
    print()
    print(title)


def fmt_int(x):
    return f"{int(x):,}".replace(",", ".")


def fmt_float(x, casas=6):
    return f"{x:.{casas}f}".replace(".", ",")


def fmt_percent(x, casas=6):
    return f"{100 * x:.{casas}f}".replace(".", ",") + "%"


# Carrega os resultados anteriores

section("1. Carregando as análises anteriores")

graph = load_json(GRAPH_BUILD)
network = load_json(NETWORK_SUMMARY)
distance = load_json(DISTANCE_EXACT)

print(f"Carregado: {GRAPH_BUILD.name}")
print(f"Carregado: {NETWORK_SUMMARY.name}")
print(f"Carregado: {DISTANCE_EXACT.name}")


# Verificações de consistência

section("2. Verificações de consistência")

assert graph["vertices"] == network["representation"]["vertices"]
assert graph["edges"] == network["representation"]["directed_edges"]

assert (
    distance["representation"]
    == "largest strongly connected component "
       "of the directed unweighted graph"
)

print("Tamanho do grafo: OK")
print("Representação das distâncias: OK")


# Extrai os valores

n_vertices = graph["vertices"]
n_edges = graph["edges"]
isolated = graph["isolated_vertices"]
total_synapses = graph["total_synapses"]

mean_in = graph["mean_in_degree"]
mean_out = graph["mean_out_degree"]
mean_total = graph["mean_total_degree"]
density = graph["directed_density"]

reciprocity = network["reciprocity"]

weak_count = network["components"]["weak_count"]
largest_weak = network["components"]["largest_weak"]
largest_weak_fraction = network["components"]["largest_weak_fraction"]

strong_count = network["components"]["strong_count"]
largest_strong = network["components"]["largest_strong"]
largest_strong_fraction = network["components"]["largest_strong_fraction"]

undirected_edges = network["undirected_projection"]["edges"]

mean_local_clustering = (
    network["clustering"]["mean_local_clustering"]
)

global_transitivity = (
    network["clustering"]["global_transitivity"]
)

clustering_eligible = (
    network["clustering"]["eligible_vertices_degree_ge_2"]
)

average_distance = distance["average_distance"]
diameter = distance["diameter"]

distance_vertices = distance["vertices"]
distance_edges = distance["edges"]


# Resultados finais estruturados

section("3. Montando os resultados finais da F1")

# As chaves e descrições do JSON foram mantidas para compatibilidade com os outros scripts.
final = {
    "dataset": "FlyWire FAFB v783",

    "canonical_graph": {
        "directed": True,
        "weighted": True,
        "vertices": n_vertices,
        "edges": n_edges,
        "isolated_vertices": isolated,
        "total_synapses": total_synapses,
        "minimum_edge_weight": graph["min_weight"],
        "maximum_edge_weight": graph["max_weight"],
        "edge_weight_definition": (
            "sum of syn_count for each directed "
            "pre_root_id -> post_root_id pair"
        ),
    },

    "degree_and_density": {
        "mean_in_degree": mean_in,
        "mean_out_degree": mean_out,
        "mean_total_degree": mean_total,
        "directed_density": density,
        "reciprocity": reciprocity,
    },

    "components": {
        "weak_component_count": weak_count,
        "largest_weak_component": largest_weak,
        "largest_weak_fraction": largest_weak_fraction,
        "strong_component_count": strong_count,
        "largest_strong_component": largest_strong,
        "largest_strong_fraction": largest_strong_fraction,
    },

    "clustering": {
        "representation": (
            "simple unweighted undirected projection "
            "of the canonical directed graph"
        ),
        "undirected_edges": undirected_edges,
        "eligible_vertices_degree_ge_2": clustering_eligible,
        "mean_local_clustering": mean_local_clustering,
        "global_transitivity": global_transitivity,
    },

    "distance": {
        "representation": (
            "largest strongly connected component "
            "of the directed unweighted graph"
        ),
        "vertices": distance_vertices,
        "edges": distance_edges,
        "definition": (
            "directed unweighted shortest-path hop count"
        ),
        "average_distance": average_distance,
        "diameter": diameter,
    },
}


with FINAL_JSON.open("w", encoding="utf-8") as f:
    json.dump(
        final,
        f,
        indent=2,
        ensure_ascii=False,
    )


# Tabela compacta de métricas

# A tabela CSV também mantém os nomes originais das métricas e escopos.
rows = [
    {
        "metric": "Vertices",
        "value": n_vertices,
        "scope": "Canonical graph",
    },
    {
        "metric": "Directed edges",
        "value": n_edges,
        "scope": "Canonical graph",
    },
    {
        "metric": "Total synapses",
        "value": total_synapses,
        "scope": "Canonical graph",
    },
    {
        "metric": "Isolated vertices",
        "value": isolated,
        "scope": "Canonical graph",
    },
    {
        "metric": "Mean in-degree",
        "value": mean_in,
        "scope": "Canonical graph",
    },
    {
        "metric": "Mean out-degree",
        "value": mean_out,
        "scope": "Canonical graph",
    },
    {
        "metric": "Mean total degree",
        "value": mean_total,
        "scope": "Canonical graph",
    },
    {
        "metric": "Directed density",
        "value": density,
        "scope": "Canonical graph",
    },
    {
        "metric": "Reciprocity",
        "value": reciprocity,
        "scope": "Canonical graph",
    },
    {
        "metric": "Weak components",
        "value": weak_count,
        "scope": "Canonical directed graph",
    },
    {
        "metric": "Largest weak component",
        "value": largest_weak,
        "scope": "Canonical directed graph",
    },
    {
        "metric": "Largest weak component fraction",
        "value": largest_weak_fraction,
        "scope": "Canonical directed graph",
    },
    {
        "metric": "Strong components",
        "value": strong_count,
        "scope": "Canonical directed graph",
    },
    {
        "metric": "Largest strong component",
        "value": largest_strong,
        "scope": "Canonical directed graph",
    },
    {
        "metric": "Largest strong component fraction",
        "value": largest_strong_fraction,
        "scope": "Canonical directed graph",
    },
    {
        "metric": "Mean local clustering",
        "value": mean_local_clustering,
        "scope": "Simple undirected projection",
    },
    {
        "metric": "Global transitivity",
        "value": global_transitivity,
        "scope": "Simple undirected projection",
    },
    {
        "metric": "Average distance",
        "value": average_distance,
        "scope": "Largest directed SCC",
    },
    {
        "metric": "Diameter",
        "value": diameter,
        "scope": "Largest directed SCC",
    },
]

metrics_df = pd.DataFrame(rows)

metrics_df.to_csv(
    FINAL_CSV,
    index=False,
)


# Resumo final em texto

summary = f"""
FlyWire FAFB v783 - resultados finais da F1

Grafo
Direcionado: sim
Ponderado: sim

Vértices: {fmt_int(n_vertices)}
Arestas direcionadas: {fmt_int(n_edges)}
Total de sinapses: {fmt_int(total_synapses)}
Vértices isolados: {fmt_int(isolated)}

Graus
Grau médio de entrada: {fmt_float(mean_in, 6)}
Grau médio de saída: {fmt_float(mean_out, 6)}
Grau médio total: {fmt_float(mean_total, 6)}

Densidade e reciprocidade
Densidade direcionada: {fmt_float(density, 12)}
Reciprocidade: {fmt_float(reciprocity, 8)}

Componentes
Componentes fracas: {fmt_int(weak_count)}
Maior componente fraca: {fmt_int(largest_weak)}
Fração na maior componente fraca: {fmt_percent(largest_weak_fraction, 6)}

Componentes fortes: {fmt_int(strong_count)}
Maior componente forte: {fmt_int(largest_strong)}
Fração na maior componente forte: {fmt_percent(largest_strong_fraction, 6)}

Clusterização
Projeção simples, não ponderada e não direcionada.

Vértices com grau >= 2: {fmt_int(clustering_eligible)}
Coeficiente de clusterização local médio: {fmt_float(mean_local_clustering, 8)}
Transitividade global: {fmt_float(global_transitivity, 8)}

Distâncias
Maior componente fortemente conexa do grafo
direcionado e não ponderado.

Vértices: {fmt_int(distance_vertices)}
Arestas: {fmt_int(distance_edges)}

Distância média exata: {fmt_float(average_distance, 10)}
Diâmetro exato: {diameter}
""".strip()


FINAL_TXT.write_text(
    summary,
    encoding="utf-8",
)


# Exibe os resultados

section("4. Resultados finais")

print(summary)

section("5. Arquivos")

print(f"JSON: {FINAL_JSON}")
print(f"CSV:  {FINAL_CSV}")
print(f"TXT:  {FINAL_TXT}")