from pathlib import Path
import argparse
import json
import math
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "f2"
ROLES_DIR = REPORT_DIR / "modular_roles"
OUT_DIR = REPORT_DIR / "topology_biology"
FIG_DIR = OUT_DIR / "figures"

CLASSIFICATION_PATH = RAW_DIR / "classification.csv.gz"
EDGES_PATH = PROCESSED_DIR / "flywire_fafb_v783_edges.csv.gz"
MODULAR_ROLES_PATH = ROLES_DIR / "node_modular_roles.csv.gz"

SUMMARY_CSV = OUT_DIR / "community_biology_summary.csv"
SUMMARY_JSON = OUT_DIR / "topology_biology_summary.json"
SUMMARY_TXT = OUT_DIR / "topology_biology_summary.txt"

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compara comunidades/papeis topologicos com super_class e calcula "
            "mixing/homofilia por superclasse no conectoma FlyWire."
        )
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["leiden", "infomap"],
        choices=["leiden", "infomap"],
    )
    parser.add_argument(
        "--top-communities",
        type=int,
        default=20,
        help="Numero de maiores comunidades mostradas nos heatmaps. Default: 20.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Nao gera PNGs via Plotly/Kaleido.",
    )
    args = parser.parse_args()
    if args.top_communities < 1:
        parser.error("--top-communities deve ser >= 1")
    return args


def section(title):
    print("\n" + "=" * 82)
    print(title)
    print("=" * 82)


def entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum())


def contingency_table(labels_a, labels_b):
    a = pd.Series(labels_a, name="a")
    b = pd.Series(labels_b, name="b")
    return pd.crosstab(a, b, dropna=False)


def nmi_arithmetic_from_contingency(table):
    n = float(table.to_numpy(dtype=np.float64).sum())
    if n <= 0:
        return np.nan
    counts = table.to_numpy(dtype=np.float64)
    row = counts.sum(axis=1)
    col = counts.sum(axis=0)
    nz = counts > 0
    ii, jj = np.nonzero(nz)
    nij = counts[ii, jj]
    mi = np.sum((nij / n) * np.log((nij * n) / (row[ii] * col[jj])))
    h_row = entropy_from_counts(row)
    h_col = entropy_from_counts(col)
    denom = 0.5 * (h_row + h_col)
    if denom == 0:
        return 1.0
    return float(mi / denom)


def comb2_sum(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum(x * (x - 1.0) / 2.0))


def ari_from_contingency(table):
    counts = table.to_numpy(dtype=np.float64)
    n = float(counts.sum())
    if n < 2:
        return np.nan
    row = counts.sum(axis=1)
    col = counts.sum(axis=0)
    sum_comb = comb2_sum(counts)
    sum_row = comb2_sum(row)
    sum_col = comb2_sum(col)
    total_pairs = n * (n - 1.0) / 2.0
    expected = (sum_row * sum_col) / total_pairs
    max_index = 0.5 * (sum_row + sum_col)
    denom = max_index - expected
    if abs(denom) < 1e-15:
        return 1.0 if abs(sum_comb - expected) < 1e-15 else 0.0
    return float((sum_comb - expected) / denom)


def partition_purity(table):
    counts = table.to_numpy(dtype=np.float64)
    n = counts.sum()
    if n <= 0:
        return np.nan
    return float(counts.max(axis=1).sum() / n)


def community_stats(table):
    rows = []
    n_classes = table.shape[1]
    max_h = math.log(n_classes) if n_classes > 1 else 1.0
    for community, row in table.iterrows():
        counts = row.to_numpy(dtype=np.int64)
        size = int(counts.sum())
        dominant_idx = int(np.argmax(counts))
        dominant_class = str(table.columns[dominant_idx])
        dominant_count = int(counts[dominant_idx])
        h = entropy_from_counts(counts)
        rows.append({
            "community": int(community),
            "size": size,
            "dominant_super_class": dominant_class,
            "dominant_count": dominant_count,
            "purity": dominant_count / size if size else np.nan,
            "superclass_entropy": h,
            "normalized_superclass_entropy": h / max_h if max_h > 0 else 0.0,
            "n_superclasses_present": int(np.count_nonzero(counts)),
        })
    return pd.DataFrame(rows).sort_values(["size", "community"], ascending=[False, True]).reset_index(drop=True)


def superclass_spread_stats(table):
    rows = []
    n_communities = table.shape[0]
    max_h = math.log(n_communities) if n_communities > 1 else 1.0
    for sc in table.columns:
        counts = table[sc].to_numpy(dtype=np.int64)
        total = int(counts.sum())
        if total == 0:
            continue
        largest_idx = int(np.argmax(counts))
        largest_community = int(table.index[largest_idx])
        largest_count = int(counts[largest_idx])
        h = entropy_from_counts(counts)
        rows.append({
            "super_class": str(sc),
            "nodes": total,
            "communities_present": int(np.count_nonzero(counts)),
            "largest_community": largest_community,
            "largest_community_count": largest_count,
            "largest_community_fraction": largest_count / total,
            "community_entropy": h,
            "normalized_community_entropy": h / max_h if max_h > 0 else 0.0,
            "effective_number_of_communities": float(np.exp(h)),
        })
    return pd.DataFrame(rows).sort_values("nodes", ascending=False).reset_index(drop=True)


def mixing_assortativity(matrix):
    m = np.asarray(matrix, dtype=np.float64)
    total = m.sum()
    if total <= 0:
        return {
            "observed_same_fraction": np.nan,
            "expected_same_fraction": np.nan,
            "same_fraction_enrichment": np.nan,
            "assortativity": np.nan,
        }
    e = m / total
    a = e.sum(axis=1)
    b = e.sum(axis=0)
    observed = float(np.trace(e))
    expected = float(np.dot(a, b))
    denom = 1.0 - expected
    r = (observed - expected) / denom if denom > 0 else np.nan
    enrichment = observed / expected if expected > 0 else np.nan
    return {
        "observed_same_fraction": observed,
        "expected_same_fraction": expected,
        "same_fraction_enrichment": enrichment,
        "assortativity": float(r),
    }


def per_superclass_homophily(matrix):
    m = np.asarray(matrix, dtype=np.float64)
    total = m.sum()
    row_total = m.sum(axis=1)
    col_total = m.sum(axis=0)
    target_marginal = col_total / total if total > 0 else np.zeros_like(col_total)
    rows = []
    labels = list(matrix.index)
    for i, sc in enumerate(labels):
        observed = m[i, i] / row_total[i] if row_total[i] > 0 else np.nan
        expected = target_marginal[i] if total > 0 else np.nan
        rows.append({
            "super_class": str(sc),
            "outgoing_mass": float(row_total[i]),
            "same_superclass_mass": float(m[i, i]),
            "observed_same_fraction": float(observed) if np.isfinite(observed) else np.nan,
            "expected_same_fraction": float(expected) if np.isfinite(expected) else np.nan,
            "enrichment": float(observed / expected) if np.isfinite(observed) and expected > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def save_matrix_variants(table, prefix):
    count_path = OUT_DIR / f"{prefix}_counts.csv"
    row_path = OUT_DIR / f"{prefix}_row_fraction.csv"
    col_path = OUT_DIR / f"{prefix}_column_fraction.csv"
    table.to_csv(count_path)
    table.div(table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0).to_csv(row_path)
    table.div(table.sum(axis=0).replace(0, np.nan), axis=1).fillna(0).to_csv(col_path)


def write_heatmap(df, title, output_path, x_title, y_title):
    """
    Gera heatmap com Plotly.

    Se a exportação PNG falhar por ausência/incompatibilidade do Kaleido,
    salva uma versão HTML e permite que o restante da análise continue.
    """
    try:
        import plotly.express as px
    except ImportError:
        print(
            f"[AVISO] Plotly não disponível; figura não gerada: {output_path.name}"
        )
        return

    fig = px.imshow(
        df,
        aspect="auto",
        text_auto=".2f",
        labels={"x": x_title, "y": y_title, "color": "Fração"},
        title=title,
    )
    fig.update_layout(
        width=1100,
        height=max(550, 30 * len(df) + 220),
    )

    try:
        fig.write_image(output_path, scale=2)
    except Exception as exc:
        html_path = output_path.with_suffix(".html")
        fig.write_html(
            html_path,
            include_plotlyjs="cdn",
        )
        print(
            f"[AVISO] Não foi possível exportar {output_path.name} como PNG. "
            f"Figura salva em HTML: {html_path.name}. "
            f"Motivo: {type(exc).__name__}"
        )


def main():
    args = parse_args()
    start_total = time.time()

    for p in [CLASSIFICATION_PATH, EDGES_PATH, MODULAR_ROLES_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {p}")

    section("1. Carregando classificação e papéis modulares")
    cls = pd.read_csv(
        CLASSIFICATION_PATH,
        usecols=["root_id", "super_class"],
        dtype={"root_id": "int64", "super_class": "string"},
    )
    roles = pd.read_csv(MODULAR_ROLES_PATH, low_memory=False)
    roles["root_id"] = roles["root_id"].astype("int64")

    if cls["root_id"].duplicated().any() or roles["root_id"].duplicated().any():
        raise RuntimeError("Há root_id duplicado em classification ou node_modular_roles.")
    if cls["super_class"].isna().any():
        raise RuntimeError("Há super_class ausente em classification.csv.gz.")

    node = roles.merge(cls, on="root_id", how="left", validate="one_to_one")
    if node["super_class"].isna().any():
        raise RuntimeError("Há neurônios de node_modular_roles sem super_class.")

    active = ~node["isolated"].astype(bool)
    print(f"Nós totais:   {len(node):,}")
    print(f"Nós ativos:   {active.sum():,}")
    print(f"Isolados:     {(~active).sum():,}")
    print(f"Superclasses: {node['super_class'].nunique()}")

    summary_rows = []
    algorithm_payload = {}

    section("2. Associação comunidades × super_class")
    for algorithm in args.algorithms:
        comm_col = f"{algorithm}_community"
        role_col = f"{algorithm}_role"
        if comm_col not in node.columns or role_col not in node.columns:
            raise RuntimeError(f"Colunas de {algorithm} ausentes em node_modular_roles.")

        sub = node.loc[active, [comm_col, "super_class", role_col]].copy()
        if (sub[comm_col] < 0).any():
            raise RuntimeError(f"{algorithm}: nó ativo com community < 0.")

        table = pd.crosstab(sub[comm_col], sub["super_class"], dropna=False).sort_index()
        nmi = nmi_arithmetic_from_contingency(table)
        ari = ari_from_contingency(table)
        purity = partition_purity(table)

        cstats = community_stats(table)
        spread = superclass_spread_stats(table)

        save_matrix_variants(table, f"{algorithm}_community_superclass")
        cstats.to_csv(OUT_DIR / f"{algorithm}_community_purity.csv", index=False)
        spread.to_csv(OUT_DIR / f"{algorithm}_superclass_spread.csv", index=False)

        role_table = pd.crosstab(sub[role_col], sub["super_class"], dropna=False)
        save_matrix_variants(role_table, f"{algorithm}_role_superclass")

        role_nmi = nmi_arithmetic_from_contingency(role_table)

        summary_rows.append({
            "algorithm": algorithm,
            "active_nodes": int(len(sub)),
            "communities": int(table.shape[0]),
            "superclasses": int(table.shape[1]),
            "nmi_arithmetic": nmi,
            "adjusted_rand_index": ari,
            "partition_purity": purity,
            "role_superclass_nmi": role_nmi,
            "largest_community_size": int(table.sum(axis=1).max()),
            "median_community_size": float(table.sum(axis=1).median()),
        })

        algorithm_payload[algorithm] = {
            "nmi_arithmetic": nmi,
            "adjusted_rand_index": ari,
            "partition_purity": purity,
            "role_superclass_nmi": role_nmi,
            "communities": int(table.shape[0]),
        }

        print(
            f"{algorithm.upper():7s} | comunidades={table.shape[0]:4d} | "
            f"NMI={nmi:.6f} | ARI={ari:.6f} | pureza={purity:.6f} | "
            f"NMI(role,super_class)={role_nmi:.6f}"
        )

        if not args.no_figures:
            top_ids = table.sum(axis=1).sort_values(ascending=False).head(args.top_communities).index
            top = table.loc[top_ids]
            top_row = top.div(top.sum(axis=1), axis=0)
            write_heatmap(
                top_row,
                f"{algorithm.capitalize()}: composição das {len(top_row)} maiores comunidades",
                FIG_DIR / f"{algorithm}_top_communities_superclass.png",
                "Superclasse",
                "Comunidade",
            )

            role_row = role_table.div(role_table.sum(axis=1), axis=0)
            write_heatmap(
                role_row,
                f"{algorithm.capitalize()}: composição biológica por papel modular",
                FIG_DIR / f"{algorithm}_roles_superclass.png",
                "Superclasse",
                "Papel modular",
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)

    section("3. Mixing e homofilia por super_class")
    edges = pd.read_csv(
        EDGES_PATH,
        usecols=["pre_root_id", "post_root_id", "weight"],
        dtype={"pre_root_id": "int64", "post_root_id": "int64", "weight": "int64"},
    )
    sc_map = cls.set_index("root_id")["super_class"]
    edges["pre_super_class"] = edges["pre_root_id"].map(sc_map)
    edges["post_super_class"] = edges["post_root_id"].map(sc_map)
    if edges[["pre_super_class", "post_super_class"]].isna().any().any():
        raise RuntimeError("Há extremos de aresta sem super_class.")

    class_order = cls["super_class"].value_counts().index.tolist()
    edge_count = pd.crosstab(edges["pre_super_class"], edges["post_super_class"]).reindex(index=class_order, columns=class_order, fill_value=0)
    edge_weight = pd.pivot_table(
        edges,
        index="pre_super_class",
        columns="post_super_class",
        values="weight",
        aggfunc="sum",
        fill_value=0,
    ).reindex(index=class_order, columns=class_order, fill_value=0)

    save_matrix_variants(edge_count, "superclass_edge_mixing")
    save_matrix_variants(edge_weight, "superclass_weight_mixing")

    count_mix = mixing_assortativity(edge_count)
    weight_mix = mixing_assortativity(edge_weight)

    hom_count = per_superclass_homophily(edge_count)
    hom_count.insert(0, "mode", "edge_count")
    hom_weight = per_superclass_homophily(edge_weight)
    hom_weight.insert(0, "mode", "synapse_weight")
    hom = pd.concat([hom_count, hom_weight], ignore_index=True)
    hom.to_csv(OUT_DIR / "superclass_homophily.csv", index=False)

    print(
        "Edge count: "
        f"same={count_mix['observed_same_fraction']:.6f}, "
        f"expected={count_mix['expected_same_fraction']:.6f}, "
        f"enrichment={count_mix['same_fraction_enrichment']:.3f}x, "
        f"r={count_mix['assortativity']:.6f}"
    )
    print(
        "Weight:     "
        f"same={weight_mix['observed_same_fraction']:.6f}, "
        f"expected={weight_mix['expected_same_fraction']:.6f}, "
        f"enrichment={weight_mix['same_fraction_enrichment']:.3f}x, "
        f"r={weight_mix['assortativity']:.6f}"
    )

    if not args.no_figures:
        edge_row = edge_count.div(edge_count.sum(axis=1), axis=0)
        write_heatmap(
            edge_row,
            "Mixing dirigido entre superclasses — fração de arestas de saída",
            FIG_DIR / "superclass_edge_mixing_row_fraction.png",
            "Superclasse pós-sináptica",
            "Superclasse pré-sináptica",
        )
        weight_row = edge_weight.div(edge_weight.sum(axis=1), axis=0)
        write_heatmap(
            weight_row,
            "Mixing dirigido entre superclasses — fração do peso de saída",
            FIG_DIR / "superclass_weight_mixing_row_fraction.png",
            "Superclasse pós-sináptica",
            "Superclasse pré-sináptica",
        )

    section("4. Salvando resumo")
    payload = {
        "methodology": {
            "target_annotation": "super_class",
            "community_association_population": "active non-isolated neurons only",
            "isolates_excluded_from_nmi_ari": int((~active).sum()),
            "nmi_normalization": "arithmetic mean of partition entropies",
            "ari": "standard adjusted Rand index from contingency table",
            "partition_purity": "sum of dominant superclass count per community / active nodes",
            "homophily_network": "canonical directed aggregated edge list",
            "homophily_modes": ["edge_count", "synapse_weight"],
            "biology_used_to_fit_communities": False,
        },
        "nodes": {
            "total": int(len(node)),
            "active": int(active.sum()),
            "isolated": int((~active).sum()),
            "superclasses": int(cls["super_class"].nunique()),
        },
        "community_association": algorithm_payload,
        "superclass_mixing": {
            "edge_count": count_mix,
            "synapse_weight": weight_mix,
        },
        "runtime_seconds": float(time.time() - start_total),
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    lines = [
        "FlyWire FAFB v783 — topologia × biologia",
        "",
        "POPULAÇÃO",
        f"Nós totais: {len(node)}",
        f"Nós ativos usados em NMI/ARI: {active.sum()}",
        f"Isolados excluídos de NMI/ARI: {(~active).sum()}",
        f"Superclasses: {cls['super_class'].nunique()}",
        "",
        "COMUNIDADES × SUPER_CLASS",
    ]
    for row in summary_rows:
        lines.extend([
            row["algorithm"].upper(),
            f"Comunidades: {row['communities']}",
            f"NMI (normalização aritmética): {row['nmi_arithmetic']:.6f}",
            f"Adjusted Rand Index: {row['adjusted_rand_index']:.6f}",
            f"Pureza da partição: {row['partition_purity']:.6f}",
            f"NMI papel modular × super_class: {row['role_superclass_nmi']:.6f}",
            "",
        ])
    lines.extend([
        "HOMOFILIA / MIXING POR SUPER_CLASS",
        "Contagem de arestas:",
        f"Fração observada intra-superclasse: {count_mix['observed_same_fraction']:.6f}",
        f"Fração esperada pelos marginais dirigidos: {count_mix['expected_same_fraction']:.6f}",
        f"Enriquecimento: {count_mix['same_fraction_enrichment']:.4f}x",
        f"Assortatividade categórica dirigida: {count_mix['assortativity']:.6f}",
        "",
        "Peso sináptico:",
        f"Fração observada intra-superclasse: {weight_mix['observed_same_fraction']:.6f}",
        f"Fração esperada pelos marginais dirigidos: {weight_mix['expected_same_fraction']:.6f}",
        f"Enriquecimento: {weight_mix['same_fraction_enrichment']:.4f}x",
        f"Assortatividade categórica dirigida: {weight_mix['assortativity']:.6f}",
        "",
        "Observação: super_class não foi usada para detectar comunidades nem para definir papéis modulares.",
        f"Tempo total: {(time.time() - start_total) / 60:.2f} min",
    ])
    SUMMARY_TXT.write_text("\n".join(lines), encoding="utf-8")

    print(f"Resumo CSV: {SUMMARY_CSV}")
    print(f"Resumo TXT: {SUMMARY_TXT}")
    print(f"Resumo JSON: {SUMMARY_JSON}")
    print(f"Tempo total: {(time.time() - start_total) / 60:.2f} min")


if __name__ == "__main__":
    main()
