from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


# ============================================================
# Caminhos
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

F2_DIR = ROOT / "reports" / "f2"
STRUCTURAL_PATH = F2_DIR / "tables" / "node_structural_features.csv.gz"
MODULAR_PATH = F2_DIR / "modular_roles" / "node_modular_roles.csv.gz"

FEATURE_DIR = F2_DIR / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = FEATURE_DIR / "node_topological_features.csv.gz"
MANIFEST_PATH = FEATURE_DIR / "topological_feature_manifest.csv"
SETS_PATH = FEATURE_DIR / "topological_feature_sets.json"
AUDIT_PATH = FEATURE_DIR / "topological_features_audit.csv"
CORR_PATH = FEATURE_DIR / "topological_high_correlations.csv"
EXCLUDED_PATH = FEATURE_DIR / "topological_excluded_columns.csv"
SUMMARY_TXT_PATH = FEATURE_DIR / "topological_features_summary.txt"
SUMMARY_JSON_PATH = FEATURE_DIR / "topological_features_summary.json"


# ============================================================
# Whitelist congelada
# ============================================================

# Bloco 1: estrutura básica dirigida/ponderada.
DEGREE_STRENGTH = [
    "in_degree",
    "out_degree",
    "undirected_degree",
    "in_strength",
    "out_strength",
]

# Bloco 2: métricas locais/globais adicionais.
CENTRALITY_LOCAL = [
    "local_clustering",
    "pagerank_weighted",
    "coreness",
]

# Bloco 3: posição modular segundo Leiden.
# IDs de comunidade NÃO entram como features.
MODULAR_LEIDEN = [
    "leiden_module_size",
    "leiden_internal_degree_fraction",
    "leiden_internal_strength_fraction",
    "leiden_within_module_degree_z",
    "leiden_within_module_strength_z",
    "leiden_participation_degree",
    "leiden_participation_strength",
]

# Bloco 4: posição modular segundo Infomap.
MODULAR_INFOMAP = [
    "infomap_module_size",
    "infomap_internal_degree_fraction",
    "infomap_internal_strength_fraction",
    "infomap_within_module_degree_z",
    "infomap_within_module_strength_z",
    "infomap_participation_degree",
    "infomap_participation_strength",
]

FEATURES = (
    DEGREE_STRENGTH
    + CENTRALITY_LOCAL
    + MODULAR_LEIDEN
    + MODULAR_INFOMAP
)

FEATURE_SETS = {
    # Progressão alinhada à proposta:
    # grau/força -> centralidade/clustering -> posição modular.
    "network_degree_strength": DEGREE_STRENGTH,
    "network_plus_centrality": (
        DEGREE_STRENGTH
        + CENTRALITY_LOCAL
    ),
    "network_plus_leiden": (
        DEGREE_STRENGTH
        + CENTRALITY_LOCAL
        + MODULAR_LEIDEN
    ),
    "network_full": FEATURES,

    # Conjuntos auxiliares para análises de sensibilidade.
    "network_modular_leiden_only": MODULAR_LEIDEN,
    "network_modular_infomap_only": MODULAR_INFOMAP,
}

# Colunas deliberadamente excluídas do bloco preditivo.
EXCLUDED_REASONS = {
    "total_degree":
        "Redundante por construção: in_degree + out_degree.",
    "total_strength":
        "Redundante por construção: in_strength + out_strength.",
    "isolated":
        "Determinístico a partir dos graus; não acrescenta informação.",
    "leiden_community":
        "ID nominal/arbitrário da comunidade; não deve ser tratado como número.",
    "infomap_community":
        "ID nominal/arbitrário da comunidade; não deve ser tratado como número.",
    "leiden_role":
        "Rótulo categórico R1-R7 derivado de z e participation; redundante.",
    "infomap_role":
        "Rótulo categórico R1-R7 derivado de z e participation; redundante.",
    "leiden_external_degree_fraction":
        "Redundante: 1 - leiden_internal_degree_fraction para nós ativos.",
    "leiden_external_strength_fraction":
        "Redundante: 1 - leiden_internal_strength_fraction para nós ativos.",
    "infomap_external_degree_fraction":
        "Redundante: 1 - infomap_internal_degree_fraction para nós ativos.",
    "infomap_external_strength_fraction":
        "Redundante: 1 - infomap_internal_strength_fraction para nós ativos.",
}

# Raw decompositions are useful for descriptive network analysis, but we do not
# place them in the frozen ML feature table because they create many exact
# linear dependencies with total in/out degree/strength.
for algorithm in ["leiden", "infomap"]:
    for name in [
        "internal_degree",
        "external_degree",
        "internal_strength",
        "external_strength",
        "in_internal_degree",
        "in_external_degree",
        "out_internal_degree",
        "out_external_degree",
        "in_internal_strength",
        "in_external_strength",
        "out_internal_strength",
        "out_external_strength",
    ]:
        EXCLUDED_REASONS[f"{algorithm}_{name}"] = (
            "Mantida no relatório descritivo, mas excluída da whitelist de ML "
            "para evitar decomposições/redundâncias exatas."
        )


# ============================================================
# Metadados das features
# ============================================================

FEATURE_META = {
    "in_degree": (
        "degree_strength",
        "directed_unweighted",
        "Número de vizinhos pré-sinápticos distintos.",
        "log1p_then_scale",
    ),
    "out_degree": (
        "degree_strength",
        "directed_unweighted",
        "Número de vizinhos pós-sinápticos distintos.",
        "log1p_then_scale",
    ),
    "undirected_degree": (
        "degree_strength",
        "undirected_unweighted_projection",
        "Número de vizinhos distintos após colapso da direção.",
        "log1p_then_scale",
    ),
    "in_strength": (
        "degree_strength",
        "directed_weighted",
        "Número total de sinapses recebidas.",
        "log1p_then_scale",
    ),
    "out_strength": (
        "degree_strength",
        "directed_weighted",
        "Número total de sinapses enviadas.",
        "log1p_then_scale",
    ),
    "local_clustering": (
        "centrality_local",
        "undirected_unweighted_projection",
        "Coeficiente de clusterização local; grau < 2 recebe zero.",
        "scale",
    ),
    "pagerank_weighted": (
        "centrality_local",
        "directed_weighted",
        "PageRank ponderado no grafo dirigido.",
        "scale",
    ),
    "coreness": (
        "centrality_local",
        "undirected_unweighted_projection",
        "Índice k-core/coreness.",
        "scale",
    ),
}

for algorithm in ["leiden", "infomap"]:
    representation = (
        "leiden_partition_on_undirected_weighted"
        if algorithm == "leiden"
        else "infomap_partition_on_directed_weighted"
    )

    FEATURE_META.update({
        f"{algorithm}_module_size": (
            f"modular_{algorithm}",
            representation,
            "Número de neurônios no módulo atribuído; isolados recebem zero.",
            "log1p_then_scale",
        ),
        f"{algorithm}_internal_degree_fraction": (
            f"modular_{algorithm}",
            "undirected_unweighted_projection",
            "Fração dos vizinhos que pertencem ao mesmo módulo.",
            "scale",
        ),
        f"{algorithm}_internal_strength_fraction": (
            f"modular_{algorithm}",
            "undirected_weighted_projection",
            "Fração da força conectada ao mesmo módulo.",
            "scale",
        ),
        f"{algorithm}_within_module_degree_z": (
            f"modular_{algorithm}",
            "undirected_unweighted_projection",
            "Z-score do grau interno relativo ao próprio módulo.",
            "scale",
        ),
        f"{algorithm}_within_module_strength_z": (
            f"modular_{algorithm}",
            "undirected_weighted_projection",
            "Z-score da força interna relativa ao próprio módulo.",
            "scale",
        ),
        f"{algorithm}_participation_degree": (
            f"modular_{algorithm}",
            "undirected_unweighted_projection",
            "Coeficiente de participação entre módulos baseado em grau.",
            "scale",
        ),
        f"{algorithm}_participation_strength": (
            f"modular_{algorithm}",
            "undirected_weighted_projection",
            "Coeficiente de participação entre módulos baseado em força.",
            "scale",
        ),
    })


# ============================================================
# Auxiliares
# ============================================================

def section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def fmt_int(x):
    return f"{int(x):,}".replace(",", ".")


def feature_audit(df):
    rows = []

    for col in FEATURES:
        x = df[col]

        rows.append({
            "feature": col,
            "group": FEATURE_META[col][0],
            "dtype": str(x.dtype),
            "missing": int(x.isna().sum()),
            "infinite": int(
                np.isinf(
                    x.to_numpy(dtype=np.float64)
                ).sum()
            ),
            "n_unique": int(x.nunique(dropna=True)),
            "zeros": int((x == 0).sum()),
            "zero_fraction": float((x == 0).mean()),
            "min": float(x.min()),
            "mean": float(x.mean()),
            "median": float(x.median()),
            "std": float(x.std(ddof=1)),
            "p95": float(x.quantile(0.95)),
            "p99": float(x.quantile(0.99)),
            "max": float(x.max()),
        })

    return pd.DataFrame(rows)


def high_correlations(df, threshold=0.95):
    """
    Spearman é usado somente como auditoria descritiva.
    Nenhuma feature é removida automaticamente com base na correlação.
    """
    corr = df[FEATURES].corr(method="spearman")

    rows = []

    for i, a in enumerate(FEATURES):
        for j in range(i + 1, len(FEATURES)):
            b = FEATURES[j]
            value = float(corr.loc[a, b])

            if abs(value) >= threshold:
                rows.append({
                    "feature_a": a,
                    "feature_b": b,
                    "spearman": value,
                    "abs_spearman": abs(value),
                })

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature_a",
                "feature_b",
                "spearman",
                "abs_spearman",
            ]
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "abs_spearman",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# Main
# ============================================================

def main():
    start = time.time()

    section("F2.06 — Preparação das features topológicas")

    for path in [STRUCTURAL_PATH, MODULAR_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo necessário não encontrado: {path}"
            )

    # --------------------------------------------------------
    # 1. Carrega fontes
    # --------------------------------------------------------

    section("1. Carregando tabelas")

    structural = pd.read_csv(
        STRUCTURAL_PATH,
        low_memory=False,
    )

    modular = pd.read_csv(
        MODULAR_PATH,
        low_memory=False,
    )

    print(
        f"Estrutural: {fmt_int(len(structural))} linhas, "
        f"{len(structural.columns)} colunas"
    )
    print(
        f"Modular:    {fmt_int(len(modular))} linhas, "
        f"{len(modular.columns)} colunas"
    )

    for name, df in [
        ("structural", structural),
        ("modular", modular),
    ]:
        if "root_id" not in df.columns:
            raise RuntimeError(
                f"{name} não possui root_id."
            )

        if df["root_id"].duplicated().any():
            raise RuntimeError(
                f"{name} possui root_id duplicado."
            )

    # Universo de nós deve ser exatamente o mesmo.
    structural_ids = structural["root_id"].to_numpy(
        dtype=np.int64
    )
    modular_ids = modular["root_id"].to_numpy(
        dtype=np.int64
    )

    if set(structural_ids) != set(modular_ids):
        only_struct = len(
            set(structural_ids) - set(modular_ids)
        )
        only_mod = len(
            set(modular_ids) - set(structural_ids)
        )

        raise RuntimeError(
            "Os universos de root_id diferem: "
            f"somente estrutural={only_struct}, "
            f"somente modular={only_mod}."
        )

    # --------------------------------------------------------
    # 2. Verifica whitelist
    # --------------------------------------------------------

    section("2. Validando whitelist")

    source_columns = (
        set(structural.columns)
        | set(modular.columns)
    )

    missing_features = [
        f
        for f in FEATURES
        if f not in source_columns
    ]

    if missing_features:
        raise RuntimeError(
            "Features esperadas ausentes: "
            + ", ".join(missing_features)
        )

    # Confirma que os indicadores de isolado concordam.
    if (
        "isolated" in structural.columns
        and "isolated" in modular.columns
    ):
        iso = structural[
            ["root_id", "isolated"]
        ].merge(
            modular[
                ["root_id", "isolated"]
            ],
            on="root_id",
            validate="one_to_one",
            suffixes=("_structural", "_modular"),
        )

        if not np.array_equal(
            iso["isolated_structural"].to_numpy(),
            iso["isolated_modular"].to_numpy(),
        ):
            raise RuntimeError(
                "Indicador de isolado diverge entre F2.01 e F2.04."
            )

    print(
        f"Whitelist de features: {len(FEATURES)}"
    )
    print(
        f"  grau/força:        {len(DEGREE_STRENGTH)}"
    )
    print(
        f"  centralidade/local:{len(CENTRALITY_LOCAL)}"
    )
    print(
        f"  Leiden:            {len(MODULAR_LEIDEN)}"
    )
    print(
        f"  Infomap:           {len(MODULAR_INFOMAP)}"
    )

    # --------------------------------------------------------
    # 3. Consolida por root_id
    # --------------------------------------------------------

    section("3. Consolidando features")

    structural_keep = [
        "root_id"
    ] + [
        f
        for f in FEATURES
        if f in structural.columns
    ]

    modular_keep = [
        "root_id"
    ] + [
        f
        for f in FEATURES
        if f in modular.columns
    ]

    topological = structural[
        structural_keep
    ].merge(
        modular[
            modular_keep
        ],
        on="root_id",
        how="inner",
        validate="one_to_one",
    )

    # Ordem canônica.
    topological = topological[
        ["root_id"] + FEATURES
    ].sort_values(
        "root_id"
    ).reset_index(drop=True)

    if len(topological) != len(structural):
        raise RuntimeError(
            "Merge alterou o número de neurônios."
        )

    if topological["root_id"].duplicated().any():
        raise RuntimeError(
            "Há root_id duplicado após consolidação."
        )

    # --------------------------------------------------------
    # 4. Auditoria numérica
    # --------------------------------------------------------

    section("4. Auditoria numérica")

    values = topological[
        FEATURES
    ].to_numpy(
        dtype=np.float64
    )

    if np.isnan(values).any():
        raise RuntimeError(
            "Há NaN nas features topológicas."
        )

    if np.isinf(values).any():
        raise RuntimeError(
            "Há infinito nas features topológicas."
        )

    audit = feature_audit(topological)

    constants = audit.loc[
        audit["n_unique"] <= 1,
        "feature",
    ].tolist()

    if constants:
        raise RuntimeError(
            "Features constantes encontradas: "
            + ", ".join(constants)
        )

    # Restrições conhecidas.
    bounded_01 = [
        "local_clustering",
        "leiden_internal_degree_fraction",
        "leiden_internal_strength_fraction",
        "leiden_participation_degree",
        "leiden_participation_strength",
        "infomap_internal_degree_fraction",
        "infomap_internal_strength_fraction",
        "infomap_participation_degree",
        "infomap_participation_strength",
    ]

    for col in bounded_01:
        if (
            topological[col].min() < -1e-12
            or topological[col].max() > 1 + 1e-12
        ):
            raise RuntimeError(
                f"{col} deveria estar em [0, 1]."
            )

    nonnegative = (
        DEGREE_STRENGTH
        + [
            "local_clustering",
            "pagerank_weighted",
            "coreness",
            "leiden_module_size",
            "leiden_internal_degree_fraction",
            "leiden_internal_strength_fraction",
            "leiden_participation_degree",
            "leiden_participation_strength",
            "infomap_module_size",
            "infomap_internal_degree_fraction",
            "infomap_internal_strength_fraction",
            "infomap_participation_degree",
            "infomap_participation_strength",
        ]
    )

    for col in nonnegative:
        if topological[col].min() < -1e-12:
            raise RuntimeError(
                f"{col} deveria ser não negativa."
            )

    print(
        f"Nós:               {fmt_int(len(topological))}"
    )
    print(
        f"Features:           {len(FEATURES)}"
    )
    print(
        "NaN:               0"
    )
    print(
        "Infinito:          0"
    )
    print(
        "Features constantes:0"
    )

    # --------------------------------------------------------
    # 5. Correlações altas
    # --------------------------------------------------------

    section("5. Auditoria de redundância")

    correlations = high_correlations(
        topological,
        threshold=0.95,
    )

    print(
        "Pares com |Spearman| >= 0.95: "
        f"{len(correlations)}"
    )

    if len(correlations):
        print(
            correlations.head(15).to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # 6. Manifesto
    # --------------------------------------------------------

    section("6. Gerando manifesto")

    manifest_rows = []

    membership = {}

    for set_name, columns in FEATURE_SETS.items():
        for col in columns:
            membership.setdefault(
                col,
                [],
            ).append(set_name)

    for feature in FEATURES:
        group, representation, description, transform = (
            FEATURE_META[feature]
        )

        source = (
            "node_structural_features.csv.gz"
            if feature in structural.columns
            else "node_modular_roles.csv.gz"
        )

        manifest_rows.append({
            "feature": feature,
            "group": group,
            "source_file": source,
            "network_representation": representation,
            "description": description,
            "recommended_preprocessing": transform,
            "feature_sets": ";".join(
                membership.get(feature, [])
            ),
        })

    manifest = pd.DataFrame(
        manifest_rows
    )

    # Registra todas as colunas deliberadamente excluídas
    # que de fato aparecem nas tabelas de origem.
    excluded_rows = []

    for col, reason in EXCLUDED_REASONS.items():
        source = None

        if col in structural.columns:
            source = "node_structural_features.csv.gz"
        elif col in modular.columns:
            source = "node_modular_roles.csv.gz"

        if source is not None:
            excluded_rows.append({
                "column": col,
                "source_file": source,
                "reason": reason,
            })

    excluded = pd.DataFrame(
        excluded_rows
    ).sort_values(
        "column"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 7. Salva
    # --------------------------------------------------------

    section("7. Salvando tabela de features")

    topological.to_csv(
        OUTPUT_PATH,
        index=False,
        compression="gzip",
    )

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    audit.to_csv(
        AUDIT_PATH,
        index=False,
    )

    correlations.to_csv(
        CORR_PATH,
        index=False,
    )

    excluded.to_csv(
        EXCLUDED_PATH,
        index=False,
    )

    with SETS_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            FEATURE_SETS,
            f,
            indent=2,
            ensure_ascii=False,
        )

    summary = {
        "nodes": int(len(topological)),
        "features": int(len(FEATURES)),
        "feature_groups": {
            "degree_strength": DEGREE_STRENGTH,
            "centrality_local": CENTRALITY_LOCAL,
            "modular_leiden": MODULAR_LEIDEN,
            "modular_infomap": MODULAR_INFOMAP,
        },
        "feature_sets": FEATURE_SETS,
        "excluded_columns": {
            row["column"]: row["reason"]
            for _, row in excluded.iterrows()
        },
        "audit": {
            "missing_total": int(
                topological[
                    FEATURES
                ].isna().sum().sum()
            ),
            "duplicate_root_id": int(
                topological[
                    "root_id"
                ].duplicated().sum()
            ),
            "constant_features": constants,
            "high_spearman_pairs_ge_0_95": int(
                len(correlations)
            ),
        },
        "important_methodological_decisions": [
            (
                "Community IDs are excluded because their numeric values are "
                "arbitrary labels, not ordered quantities."
            ),
            (
                "R1-R7 modular-role labels are excluded because they are "
                "deterministic discretizations of within-module z and "
                "participation coefficient."
            ),
            (
                "Exact total/decomposition redundancies are excluded from the "
                "ML whitelist; raw descriptive versions remain preserved in "
                "the upstream F2 outputs."
            ),
            (
                "No biological annotation and no super_class label is present "
                "in the frozen topological feature table."
            ),
            (
                "Suggested transformations are metadata only. All scaling, "
                "imputation or transformation must be fitted inside the "
                "training split/fold during ML."
            ),
        ],
        "runtime_seconds": float(
            time.time() - start
        ),
        "output": OUTPUT_PATH.relative_to(
            ROOT
        ).as_posix(),
    }

    with SUMMARY_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    text = [
        "FlyWire FAFB v783 — congelamento das features topológicas",
        "",
        f"Nós: {len(topological)}",
        f"Features selecionadas: {len(FEATURES)}",
        "",
        "GRUPOS",
        (
            f"degree_strength ({len(DEGREE_STRENGTH)}): "
            + ", ".join(DEGREE_STRENGTH)
        ),
        (
            f"centrality_local ({len(CENTRALITY_LOCAL)}): "
            + ", ".join(CENTRALITY_LOCAL)
        ),
        (
            f"modular_leiden ({len(MODULAR_LEIDEN)}): "
            + ", ".join(MODULAR_LEIDEN)
        ),
        (
            f"modular_infomap ({len(MODULAR_INFOMAP)}): "
            + ", ".join(MODULAR_INFOMAP)
        ),
        "",
        "AUDITORIA",
        "NaN: 0",
        "Infinito: 0",
        "root_id duplicado: 0",
        "Features constantes: 0",
        (
            "Pares com |Spearman| >= 0.95: "
            f"{len(correlations)}"
        ),
        "",
        "DECISÕES",
        "- total_degree e total_strength excluídos por redundância exata.",
        "- isolated excluído porque é determinístico a partir dos graus.",
        "- IDs Leiden/Infomap excluídos: são rótulos nominais arbitrários.",
        "- R1-R7 excluídos: derivam de z intramodular e participation.",
        (
            "- Conectividades brutas intra/intermódulo permanecem nos "
            "relatórios descritivos, mas não entram na whitelist de ML."
        ),
        (
            "- Nenhum rótulo biológico ou super_class está presente na "
            "tabela de features."
        ),
        (
            "- Transformações/normalização serão ajustadas somente no treino "
            "durante a etapa de IA."
        ),
        "",
        "CONJUNTO PRINCIPAL PARA IA",
        "network_full:",
        *[
            f"  - {feature}"
            for feature in FEATURE_SETS[
                "network_full"
            ]
        ],
        "",
        (
            "Tempo total: "
            f"{(time.time() - start):.2f} s"
        ),
    ]

    SUMMARY_TXT_PATH.write_text(
        "\n".join(text),
        encoding="utf-8",
    )

    print()
    print(f"Tabela:    {OUTPUT_PATH}")
    print(f"Manifesto: {MANIFEST_PATH}")
    print(f"Sets:      {SETS_PATH}")
    print(f"Auditoria: {AUDIT_PATH}")
    print(f"Resumo:    {SUMMARY_TXT_PATH}")

    section("Concluído")
    print(
        f"Tempo total: {time.time() - start:.2f} s"
    )


if __name__ == "__main__":
    main()
