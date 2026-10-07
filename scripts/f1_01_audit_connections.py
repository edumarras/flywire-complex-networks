from pathlib import Path
import json

import duckdb


# Configuração
ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = ROOT / "data" / "raw" / "connections_princeton.csv.gz"

OUTPUT_DIR = ROOT / "reports" / "f1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "connections_audit.json"
OUTPUT_TXT = OUTPUT_DIR / "connections_audit.txt"


def format_int(value):
    if value is None:
        return "Nenhum"
    return f"{int(value):,}".replace(",", ".")


def format_float(value, decimals=4):
    if value is None:
        return "Nenhum"
    return f"{value:.{decimals}f}".replace(".", ",")


def section(title):
    print(f"\n{title}")


def fetch_one(con, query):
    return con.execute(query).fetchone()[0]


# Validação do arquivo de entrada
if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado:\n{INPUT_PATH}"
    )

section("FlyWire F1 - auditoria das conexões")

print(f"Arquivo de entrada: {INPUT_PATH}")
print(f"Tamanho do arquivo: {format_float(INPUT_PATH.stat().st_size / (1024 ** 2), 2)} MB")


# Leitura com DuckDB
con = duckdb.connect()

# O DuckDB lê o arquivo .gz diretamente.
source = f"""
read_csv_auto(
    '{INPUT_PATH.as_posix()}',
    header = true,
    sample_size = -1
)
"""


# Colunas do arquivo
section("1. Colunas")

schema = con.execute(
    f"""
    DESCRIBE
    SELECT *
    FROM {source}
    """
).fetchall()

columns = [row[0] for row in schema]

for row in schema:
    print(f"{row[0]:30s} {row[1]}")

required = {
    "pre_root_id",
    "post_root_id",
    "syn_count",
}

missing = sorted(required - set(columns))

if missing:
    raise ValueError(
        f"Colunas obrigatórias ausentes: {missing}"
    )


# Estatísticas da tabela original
section("2. Tabela original de conexões")

row_count = fetch_one(
    con,
    f"""
    SELECT COUNT(*)
    FROM {source}
    """
)

unique_pre = fetch_one(
    con,
    f"""
    SELECT COUNT(DISTINCT pre_root_id)
    FROM {source}
    """
)

unique_post = fetch_one(
    con,
    f"""
    SELECT COUNT(DISTINCT post_root_id)
    FROM {source}
    """
)

unique_nodes = fetch_one(
    con,
    f"""
    SELECT COUNT(DISTINCT root_id)
    FROM (
        SELECT pre_root_id AS root_id
        FROM {source}

        UNION

        SELECT post_root_id AS root_id
        FROM {source}
    )
    """
)

print(f"Número de linhas:          {format_int(row_count)}")
print(f"pre_root_id únicos:        {format_int(unique_pre)}")
print(f"post_root_id únicos:       {format_int(unique_post)}")
print(f"Nós únicos:                {format_int(unique_nodes)}")


# Estatísticas de syn_count
section("3. syn_count")

syn_stats = con.execute(
    f"""
    SELECT
        MIN(syn_count),
        MAX(syn_count),
        AVG(syn_count),
        MEDIAN(syn_count),
        SUM(syn_count),
        SUM(CASE WHEN syn_count < 5 THEN 1 ELSE 0 END),
        SUM(CASE WHEN syn_count >= 5 THEN 1 ELSE 0 END)
    FROM {source}
    """
).fetchone()

(
    syn_min,
    syn_max,
    syn_mean,
    syn_median,
    syn_total,
    rows_lt5,
    rows_ge5,
) = syn_stats

print(f"Mínimo:                    {syn_min}")
print(f"Máximo:                    {syn_max}")
print(f"Média:                     {format_float(syn_mean)}")
print(f"Mediana:                   {syn_median}")
print(f"Total de sinapses:         {format_int(syn_total)}")
print(f"Linhas com syn_count < 5:  {format_int(rows_lt5)}")
print(f"Linhas com syn_count >= 5: {format_int(rows_ge5)}")


# Pares direcionados
section("4. Pares direcionados pre -> post")

unique_pairs = fetch_one(
    con,
    f"""
    SELECT COUNT(*)
    FROM (
        SELECT DISTINCT
            pre_root_id,
            post_root_id
        FROM {source}
    )
    """
)

duplicate_records = row_count - unique_pairs

pair_stats = con.execute(
    f"""
    SELECT
        SUM(CASE WHEN n > 1 THEN 1 ELSE 0 END),
        MAX(n)
    FROM (
        SELECT
            pre_root_id,
            post_root_id,
            COUNT(*) AS n
        FROM {source}
        GROUP BY
            pre_root_id,
            post_root_id
    )
    """
).fetchone()

repeated_pairs, max_pair_records = pair_stats

print(f"Pares direcionados únicos:       {format_int(unique_pairs)}")
print(f"Linhas duplicadas adicionais:    {format_int(duplicate_records)}")
print(f"Pares que aparecem mais de 1 vez: {format_int(repeated_pairs)}")
print(f"Máximo de linhas para um par:    {format_int(max_pair_records)}")


# Agregação dos registros do mesmo par direcionado
section("5. Depois da agregação por par direcionado")

aggregated_stats = con.execute(
    f"""
    WITH aggregated AS (
        SELECT
            pre_root_id,
            post_root_id,
            SUM(syn_count) AS syn_count
        FROM {source}
        GROUP BY
            pre_root_id,
            post_root_id
    )

    SELECT
        COUNT(*),
        SUM(CASE WHEN syn_count < 5 THEN 1 ELSE 0 END),
        SUM(CASE WHEN syn_count >= 5 THEN 1 ELSE 0 END),
        MIN(syn_count),
        MAX(syn_count),
        AVG(syn_count),
        SUM(syn_count)
    FROM aggregated
    """
).fetchone()

(
    agg_pairs,
    agg_lt5,
    agg_ge5,
    agg_min,
    agg_max,
    agg_mean,
    agg_total_syn,
) = aggregated_stats

print(f"Pares direcionados agregados: {format_int(agg_pairs)}")
print(f"Pares agregados < 5:          {format_int(agg_lt5)}")
print(f"Pares agregados >= 5:         {format_int(agg_ge5)}")
print(f"Peso mínimo agregado:         {agg_min}")
print(f"Peso máximo agregado:         {agg_max}")
print(f"Peso médio agregado:          {format_float(agg_mean)}")
print(f"Total de sinapses:            {format_int(agg_total_syn)}")


# Laços
section("6. Laços")

self_loop_rows = fetch_one(
    con,
    f"""
    SELECT COUNT(*)
    FROM {source}
    WHERE pre_root_id = post_root_id
    """
)

self_loop_pairs = fetch_one(
    con,
    f"""
    SELECT COUNT(*)
    FROM (
        SELECT DISTINCT
            pre_root_id,
            post_root_id
        FROM {source}
        WHERE pre_root_id = post_root_id
    )
    """
)

print(f"Linhas com laço:       {format_int(self_loop_rows)}")
print(f"Pares de laço únicos:  {format_int(self_loop_pairs)}")


# Valores ausentes nas colunas usadas na construção do grafo
section("7. Valores ausentes nas colunas principais")

missing_stats = con.execute(
    f"""
    SELECT
        SUM(CASE WHEN pre_root_id IS NULL THEN 1 ELSE 0 END),
        SUM(CASE WHEN post_root_id IS NULL THEN 1 ELSE 0 END),
        SUM(CASE WHEN syn_count IS NULL THEN 1 ELSE 0 END)
    FROM {source}
    """
).fetchone()

missing_pre, missing_post, missing_syn = missing_stats

print(f"pre_root_id ausente:   {format_int(missing_pre)}")
print(f"post_root_id ausente:  {format_int(missing_post)}")
print(f"syn_count ausente:     {format_int(missing_syn)}")


# Salva também os resultados em JSON. As chaves foram mantidas para não
# quebrar outros scripts que possam usar esse arquivo depois.
results = {
    "input_file": INPUT_PATH.relative_to(ROOT).as_posix(),
    "file_size_mb": INPUT_PATH.stat().st_size / (1024 ** 2),

    "schema": {
        row[0]: row[1]
        for row in schema
    },

    "raw": {
        "rows": row_count,
        "unique_pre_nodes": unique_pre,
        "unique_post_nodes": unique_post,
        "unique_nodes": unique_nodes,
    },

    "syn_count": {
        "min": syn_min,
        "max": syn_max,
        "mean": syn_mean,
        "median": syn_median,
        "total": syn_total,
        "rows_lt_5": rows_lt5,
        "rows_ge_5": rows_ge5,
    },

    "pairs": {
        "unique_directed_pairs": unique_pairs,
        "additional_duplicate_rows": duplicate_records,
        "repeated_pairs": repeated_pairs,
        "max_rows_per_pair": max_pair_records,
    },

    "aggregated": {
        "pairs": agg_pairs,
        "pairs_lt_5": agg_lt5,
        "pairs_ge_5": agg_ge5,
        "min_weight": agg_min,
        "max_weight": agg_max,
        "mean_weight": agg_mean,
        "total_synapses": agg_total_syn,
    },

    "self_loops": {
        "raw_rows": self_loop_rows,
        "unique_pairs": self_loop_pairs,
    },

    "missing": {
        "pre_root_id": missing_pre,
        "post_root_id": missing_post,
        "syn_count": missing_syn,
    },
}

with OUTPUT_JSON.open("w", encoding="utf-8") as f:
    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


# Resumo em texto para consulta rápida
summary = f"""
Auditoria das conexões - FlyWire FAFB v783

Arquivo de entrada:
{INPUT_PATH.relative_to(ROOT).as_posix()}
Tamanho: {format_float(INPUT_PATH.stat().st_size / (1024 ** 2), 2)} MB

Dados brutos:
Número de linhas: {format_int(row_count)}
pre_root_id únicos: {format_int(unique_pre)}
post_root_id únicos: {format_int(unique_post)}
Nós únicos: {format_int(unique_nodes)}

syn_count:
Mínimo: {syn_min}
Máximo: {syn_max}
Média: {format_float(syn_mean)}
Mediana: {syn_median}
Total de sinapses: {format_int(syn_total)}
Linhas com syn_count < 5: {format_int(rows_lt5)}
Linhas com syn_count >= 5: {format_int(rows_ge5)}

Pares direcionados:
Pares únicos: {format_int(unique_pairs)}
Linhas duplicadas adicionais: {format_int(duplicate_records)}
Pares repetidos: {format_int(repeated_pairs)}
Máximo de linhas para um par: {format_int(max_pair_records)}

Depois da agregação:
Pares: {format_int(agg_pairs)}
Pares < 5: {format_int(agg_lt5)}
Pares >= 5: {format_int(agg_ge5)}
Peso mínimo: {agg_min}
Peso máximo: {agg_max}
Peso médio: {format_float(agg_mean)}
Total de sinapses: {format_int(agg_total_syn)}

Laços:
Linhas com laço: {format_int(self_loop_rows)}
Pares de laço únicos: {format_int(self_loop_pairs)}

Valores ausentes:
pre_root_id: {format_int(missing_pre)}
post_root_id: {format_int(missing_post)}
syn_count: {format_int(missing_syn)}
""".strip()

OUTPUT_TXT.write_text(
    summary,
    encoding="utf-8",
)

section("Finalizado")

print(f"Relatório JSON: {OUTPUT_JSON}")
print(f"Relatório em texto: {OUTPUT_TXT}")

con.close()
