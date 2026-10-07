from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]

CONNECTIONS = ROOT / "data" / "raw" / "connections_princeton.csv.gz"
NEURONS = ROOT / "data" / "raw" / "neurons.csv.gz"


if not CONNECTIONS.exists():
    raise FileNotFoundError(CONNECTIONS)

if not NEURONS.exists():
    raise FileNotFoundError(NEURONS)


def format_int(value):
    return f"{int(value):,}".replace(",", ".")


con = duckdb.connect()

connections = f"""
read_csv_auto(
    '{CONNECTIONS.as_posix()}',
    header=true,
    sample_size=-1
)
"""

neurons = f"""
read_csv_auto(
    '{NEURONS.as_posix()}',
    header=true,
    sample_size=-1
)
"""


print("1. Estrutura de neurons.csv.gz")
print()

schema = con.execute(
    f"""
    DESCRIBE
    SELECT *
    FROM {neurons}
    """
).fetchall()

for row in schema:
    print(f"{row[0]:30s} {row[1]}")


columns = {row[0] for row in schema}

# O arquivo de neurônios do FlyWire usa root_id como identificador.
if "root_id" not in columns:
    raise ValueError(
        "A coluna esperada 'root_id' não foi encontrada em neurons.csv.gz"
    )


print()
print("2. Conjunto de nós")
print()

n_neuron_rows = con.execute(
    f"""
    SELECT COUNT(*)
    FROM {neurons}
    """
).fetchone()[0]

n_neuron_ids = con.execute(
    f"""
    SELECT COUNT(DISTINCT root_id)
    FROM {neurons}
    """
).fetchone()[0]

n_connection_nodes = con.execute(
    f"""
    SELECT COUNT(DISTINCT root_id)
    FROM (
        SELECT pre_root_id AS root_id
        FROM {connections}

        UNION

        SELECT post_root_id AS root_id
        FROM {connections}
    )
    """
).fetchone()[0]


n_neurons_not_in_edges = con.execute(
    f"""
    WITH edge_nodes AS (
        SELECT pre_root_id AS root_id
        FROM {connections}

        UNION

        SELECT post_root_id AS root_id
        FROM {connections}
    )

    SELECT COUNT(*)
    FROM (
        SELECT DISTINCT root_id
        FROM {neurons}
    ) n
    LEFT JOIN edge_nodes e USING (root_id)
    WHERE e.root_id IS NULL
    """
).fetchone()[0]


n_edge_nodes_not_in_neurons = con.execute(
    f"""
    WITH edge_nodes AS (
        SELECT pre_root_id AS root_id
        FROM {connections}

        UNION

        SELECT post_root_id AS root_id
        FROM {connections}
    ),

    neuron_nodes AS (
        SELECT DISTINCT root_id
        FROM {neurons}
    )

    SELECT COUNT(*)
    FROM edge_nodes e
    LEFT JOIN neuron_nodes n USING (root_id)
    WHERE n.root_id IS NULL
    """
).fetchone()[0]


print(f"Linhas em neurons.csv:                  {format_int(n_neuron_rows)}")
print(f"root_id únicos em neurons.csv:          {format_int(n_neuron_ids)}")
print(f"Nós presentes nas conexões:             {format_int(n_connection_nodes)}")
print(f"Neurônios ausentes das conexões:        {format_int(n_neurons_not_in_edges)}")
print(
    f"Nós das conexões ausentes em neurons:   "
    f"{format_int(n_edge_nodes_not_in_neurons)}"
)


con.close()
