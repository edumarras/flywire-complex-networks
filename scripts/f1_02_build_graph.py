from pathlib import Path
import gzip
import html
import json
import time

import duckdb


# Caminhos
ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "f1"

CONNECTIONS_PATH = RAW_DIR / "connections_princeton.csv.gz"
NEURONS_PATH = RAW_DIR / "neurons.csv.gz"

GRAPHML_PATH = PROCESSED_DIR / "flywire_fafb_v783.graphml"
EDGES_PATH = PROCESSED_DIR / "flywire_fafb_v783_edges.csv.gz"

BUILD_REPORT_JSON = REPORT_DIR / "graph_build.json"
BUILD_REPORT_TXT = REPORT_DIR / "graph_build.txt"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# Verifica se os arquivos de entrada existem
for path in [CONNECTIONS_PATH, NEURONS_PATH]:
    if not path.exists():
        raise FileNotFoundError(path)


def format_int(x):
    return f"{int(x):,}".replace(",", ".")


def format_float(x, casas):
    return f"{x:.{casas}f}".replace(".", ",")


def section(title):
    print()
    print(title)


con = duckdb.connect()

connections = f"""
read_csv_auto(
    '{CONNECTIONS_PATH.as_posix()}',
    header=true,
    sample_size=-1
)
"""

neurons = f"""
read_csv_auto(
    '{NEURONS_PATH.as_posix()}',
    header=true,
    sample_size=-1
)
"""


# Agrega as conexões repetidas entre o mesmo par de neurônios
section("1. Construção da lista de arestas")

con.execute(
    f"""
    CREATE OR REPLACE TEMP TABLE aggregated_edges AS
    SELECT
        pre_root_id,
        post_root_id,
        SUM(syn_count)::BIGINT AS weight
    FROM {connections}
    GROUP BY
        pre_root_id,
        post_root_id
    """
)

n_edges = con.execute(
    """
    SELECT COUNT(*)
    FROM aggregated_edges
    """
).fetchone()[0]

min_weight, max_weight, total_weight = con.execute(
    """
    SELECT
        MIN(weight),
        MAX(weight),
        SUM(weight)
    FROM aggregated_edges
    """
).fetchone()

bad_edges = con.execute(
    """
    SELECT COUNT(*)
    FROM aggregated_edges
    WHERE weight < 5
    """
).fetchone()[0]

self_loops = con.execute(
    """
    SELECT COUNT(*)
    FROM aggregated_edges
    WHERE pre_root_id = post_root_id
    """
).fetchone()[0]

print(f"Arestas:                     {format_int(n_edges)}")
print(f"Peso mínimo:                 {min_weight}")
print(f"Peso máximo:                 {max_weight}")
print(f"Total de sinapses:           {format_int(total_weight)}")
print(f"Arestas com peso < 5:        {format_int(bad_edges)}")
print(f"Autolaços:                   {format_int(self_loops)}")

if bad_edges != 0:
    raise RuntimeError(
        "Foram encontradas arestas com peso agregado menor que 5."
    )

if self_loops != 0:
    raise RuntimeError(
        "Foram encontrados autolaços."
    )


# O conjunto de nós é definido pelos root_id presentes em neurons.csv.gz
section("2. Conjunto de nós")

con.execute(
    f"""
    CREATE OR REPLACE TEMP TABLE graph_nodes AS
    SELECT DISTINCT root_id
    FROM {neurons}
    ORDER BY root_id
    """
)

n_nodes = con.execute(
    """
    SELECT COUNT(*)
    FROM graph_nodes
    """
).fetchone()[0]

isolated_nodes = con.execute(
    """
    WITH connected_nodes AS (
        SELECT pre_root_id AS root_id
        FROM aggregated_edges

        UNION

        SELECT post_root_id AS root_id
        FROM aggregated_edges
    )

    SELECT COUNT(*)
    FROM graph_nodes n
    LEFT JOIN connected_nodes c USING (root_id)
    WHERE c.root_id IS NULL
    """
).fetchone()[0]

edge_nodes_missing_from_neurons = con.execute(
    """
    WITH connected_nodes AS (
        SELECT pre_root_id AS root_id
        FROM aggregated_edges

        UNION

        SELECT post_root_id AS root_id
        FROM aggregated_edges
    )

    SELECT COUNT(*)
    FROM connected_nodes c
    LEFT JOIN graph_nodes n USING (root_id)
    WHERE n.root_id IS NULL
    """
).fetchone()[0]

print(f"Vértices:                            {format_int(n_nodes)}")
print(f"Vértices isolados:                   {format_int(isolated_nodes)}")
print(
    f"Vértices das arestas fora de neurons: "
    f"{format_int(edge_nodes_missing_from_neurons)}"
)

if edge_nodes_missing_from_neurons != 0:
    raise RuntimeError(
        "Alguns extremos das arestas não aparecem em neurons.csv.gz."
    )


# Salva a lista de arestas agregada
section("3. Salvando a lista de arestas")

con.execute(
    f"""
    COPY (
        SELECT
            pre_root_id,
            post_root_id,
            weight
        FROM aggregated_edges
        ORDER BY
            pre_root_id,
            post_root_id
    )
    TO '{EDGES_PATH.as_posix()}'
    (
        HEADER,
        DELIMITER ',',
        COMPRESSION GZIP
    )
    """
)

print(f"Arquivo salvo em: {EDGES_PATH}")


# Exporta o grafo em GraphML
section("4. Exportação para GraphML")

start = time.time()

with GRAPHML_PATH.open(
    "w",
    encoding="utf-8",
    newline="\n",
    buffering=1024 * 1024,
) as f:

    # Cabeçalho do GraphML
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write(
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns"\n'
    )
    f.write(
        '         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
    )
    f.write(
        '         xsi:schemaLocation="'
        'http://graphml.graphdrawing.org/xmlns '
        'http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">\n'
    )

    # Peso das arestas
    f.write(
        '  <key id="weight" for="edge" '
        'attr.name="weight" attr.type="long"/>\n'
    )

    f.write(
        '  <graph id="flywire_fafb_v783" edgedefault="directed">\n'
    )

    # Nós
    node_cursor = con.execute(
        """
        SELECT root_id
        FROM graph_nodes
        ORDER BY root_id
        """
    )

    node_count_written = 0

    while True:
        rows = node_cursor.fetchmany(100_000)

        if not rows:
            break

        for (root_id,) in rows:
            # O root_id é usado como identificador do nó no GraphML.
            f.write(f'    <node id="{root_id}"/>\n')

        node_count_written += len(rows)

        print(
            f"\rNós escritos: "
            f"{format_int(node_count_written)}/{format_int(n_nodes)}",
            end="",
            flush=True,
        )

    print()

    # Arestas
    edge_cursor = con.execute(
        """
        SELECT
            pre_root_id,
            post_root_id,
            weight
        FROM aggregated_edges
        ORDER BY
            pre_root_id,
            post_root_id
        """
    )

    edge_count_written = 0

    while True:
        rows = edge_cursor.fetchmany(100_000)

        if not rows:
            break

        for pre, post, weight in rows:
            f.write(
                f'    <edge '
                f'source="{pre}" '
                f'target="{post}">'
                f'<data key="weight">{weight}</data>'
                f'</edge>\n'
            )

        edge_count_written += len(rows)

        print(
            f"\rArestas escritas: "
            f"{format_int(edge_count_written)}/{format_int(n_edges)}",
            end="",
            flush=True,
        )

    print()

    f.write("  </graph>\n")
    f.write("</graphml>\n")


elapsed = time.time() - start

print(f"Arquivo salvo em: {GRAPHML_PATH}")
print(f"Tempo de exportação: {format_float(elapsed / 60, 2)} min")


# Informações finais do grafo
section("5. Grafo construído")

mean_out_degree = n_edges / n_nodes
mean_in_degree = n_edges / n_nodes
mean_total_degree = 2 * n_edges / n_nodes

density = n_edges / (n_nodes * (n_nodes - 1))

print(f"Vértices:             {format_int(n_nodes)}")
print(f"Arestas:              {format_int(n_edges)}")
print(f"Vértices isolados:    {format_int(isolated_nodes)}")
print("Direcionado:          sim")
print("Ponderado:            sim")
print(f"Grau médio de entrada:{format_float(mean_in_degree, 6):>14}")
print(f"Grau médio de saída:  {format_float(mean_out_degree, 6):>14}")
print(f"Grau médio total:     {format_float(mean_total_degree, 6):>14}")
print(f"Densidade:            {format_float(density, 12):>14}")


# Metadados em JSON. As chaves foram mantidas para não quebrar outros scripts.
results = {
    "graph": "FlyWire FAFB v783",
    "directed": True,
    "weighted": True,
    "node_universe": "all root_id values from neurons.csv.gz",
    "edge_definition": (
        "directed pair pre_root_id -> post_root_id; "
        "weight = sum(syn_count)"
    ),
    "minimum_aggregated_weight": 5,

    "vertices": n_nodes,
    "edges": n_edges,
    "isolated_vertices": isolated_nodes,

    "min_weight": min_weight,
    "max_weight": max_weight,
    "total_synapses": total_weight,

    "mean_in_degree": mean_in_degree,
    "mean_out_degree": mean_out_degree,
    "mean_total_degree": mean_total_degree,
    "directed_density": density,

    "graphml": GRAPHML_PATH.relative_to(ROOT).as_posix(),
    "edge_list": EDGES_PATH.relative_to(ROOT).as_posix(),
}

with BUILD_REPORT_JSON.open(
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
FlyWire FAFB v783 - grafo construído

Representação
Direcionado: sim
Ponderado: sim

Nós
Foram usados todos os root_id presentes em neurons.csv.gz.

Arestas
pre_root_id -> post_root_id
peso = soma de syn_count para cada par direcionado
peso agregado mínimo = 5

Grafo
Vértices: {format_int(n_nodes)}
Arestas: {format_int(n_edges)}
Vértices isolados: {format_int(isolated_nodes)}

Pesos
Mínimo: {min_weight}
Máximo: {max_weight}
Total de sinapses: {format_int(total_weight)}

Graus
Grau médio de entrada: {format_float(mean_in_degree, 6)}
Grau médio de saída: {format_float(mean_out_degree, 6)}
Grau médio total: {format_float(mean_total_degree, 6)}

Densidade
Densidade do grafo direcionado: {format_float(density, 12)}

Arquivos
GraphML: {GRAPHML_PATH.relative_to(ROOT).as_posix()}
Lista de arestas: {EDGES_PATH.relative_to(ROOT).as_posix()}
""".strip()

BUILD_REPORT_TXT.write_text(
    summary,
    encoding="utf-8",
)


section("Concluído")

print(f"Relatório: {BUILD_REPORT_TXT}")
print(f"JSON:      {BUILD_REPORT_JSON}")

con.close()
