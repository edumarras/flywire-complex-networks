from pathlib import Path
import argparse
import gc
import json
import time

import numpy as np
import pandas as pd


# ============================================================
# Caminhos
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "f2"
TABLE_DIR = REPORT_DIR / "tables"
COMMUNITY_DIR = REPORT_DIR / "communities"
ROLES_DIR = REPORT_DIR / "modular_roles"

EDGES_PATH = PROCESSED_DIR / "flywire_fafb_v783_edges.csv.gz"
NODE_FEATURES_PATH = TABLE_DIR / "node_structural_features.csv.gz"
ASSIGNMENTS_PATH = COMMUNITY_DIR / "community_assignments.csv.gz"

ROLES_PATH = ROLES_DIR / "node_modular_roles.csv.gz"
SUMMARY_CSV_PATH = ROLES_DIR / "modular_roles_summary.csv"
SUMMARY_JSON_PATH = ROLES_DIR / "modular_roles_summary.json"
SUMMARY_TXT_PATH = ROLES_DIR / "modular_roles_summary.txt"

LEIDEN_MODULE_STATS_PATH = ROLES_DIR / "leiden_module_stats.csv"
INFOMAP_MODULE_STATS_PATH = ROLES_DIR / "infomap_module_stats.csv"

LEIDEN_ROLE_COUNTS_PATH = ROLES_DIR / "leiden_role_counts.csv"
INFOMAP_ROLE_COUNTS_PATH = ROLES_DIR / "infomap_role_counts.csv"

ROLES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Funções auxiliares
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calcula papéis modulares por neurônio usando as partições "
            "Leiden e Infomap produzidas por f2_03_communities.py."
        )
    )

    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["leiden", "infomap"],
        choices=["leiden", "infomap"],
        help=(
            "Partições para as quais calcular papéis modulares. "
            "Default: leiden infomap."
        ),
    )

    return parser.parse_args()


def section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def fmt_int(x):
    return f"{int(x):,}".replace(",", ".")


def elapsed(start):
    return f"{time.time() - start:.2f} s"


def safe_ratio(num, den):
    out = np.zeros_like(
        np.asarray(num, dtype=np.float64),
        dtype=np.float64,
    )

    den = np.asarray(den, dtype=np.float64)
    num = np.asarray(num, dtype=np.float64)

    mask = den != 0
    out[mask] = num[mask] / den[mask]

    return out


def within_module_z(values, membership):
    """
    Z-score intramodular:
        z_i = (k_i^s - mean_s) / sd_s

    A média e o desvio-padrão são calculados somente entre os nós
    da mesma comunidade.

    Usa desvio-padrão populacional (ddof=0), em linha com a definição
    clássica de papel modular. Comunidades sem variância recebem z=0.

    membership = -1 representa isolados e recebe z=0.
    """
    values = np.asarray(values, dtype=np.float64)
    membership = np.asarray(membership, dtype=np.int64)

    out = np.zeros(len(values), dtype=np.float64)

    active = membership >= 0

    if not np.any(active):
        return out

    comm = membership[active]
    vals = values[active]

    n_communities = int(comm.max()) + 1

    count = np.bincount(
        comm,
        minlength=n_communities,
    ).astype(np.float64)

    total = np.bincount(
        comm,
        weights=vals,
        minlength=n_communities,
    )

    total_sq = np.bincount(
        comm,
        weights=vals * vals,
        minlength=n_communities,
    )

    mean = np.divide(
        total,
        count,
        out=np.zeros_like(total),
        where=count > 0,
    )

    variance = np.divide(
        total_sq,
        count,
        out=np.zeros_like(total_sq),
        where=count > 0,
    ) - mean * mean

    # Proteção contra pequenos valores negativos por erro numérico.
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)

    denom = std[comm]
    numer = vals - mean[comm]

    z = np.divide(
        numer,
        denom,
        out=np.zeros_like(numer),
        where=denom > 0,
    )

    out[active] = z
    return out


def participation_coefficients(
    u_src,
    u_dst,
    u_weight,
    membership,
    total_degree,
    total_strength,
):
    """
    Coeficiente de participação na projeção simples não dirigida:

        P_i = 1 - sum_s (k_is / k_i)^2

    e sua versão ponderada:

        P_i^w = 1 - sum_s (s_is / s_i)^2

    onde k_is é o número de vizinhos de i na comunidade s e
    s_is é a força de i direcionada à comunidade s.

    Os IDs das comunidades devem ser inteiros consecutivos a partir de 0.
    Isolados possuem membership=-1 e recebem P=0.

    Implementação vetorizada:
    - cria uma contribuição para cada extremidade de cada aresta;
    - codifica (nó, comunidade do vizinho) em uma chave inteira;
    - agrega contagens e pesos por chave;
    - soma os quadrados por nó.
    """
    membership = np.asarray(membership, dtype=np.int64)

    n_nodes = len(membership)

    active_membership = membership[membership >= 0]
    if len(active_membership) == 0:
        return (
            np.zeros(n_nodes, dtype=np.float64),
            np.zeros(n_nodes, dtype=np.float64),
        )

    n_communities = int(active_membership.max()) + 1

    # Cada aresta não dirigida contribui para os dois extremos.
    endpoint_node = np.concatenate(
        [u_src, u_dst]
    ).astype(np.int64, copy=False)

    neighbor_community = np.concatenate(
        [
            membership[u_dst],
            membership[u_src],
        ]
    ).astype(np.int64, copy=False)

    endpoint_weight = np.concatenate(
        [u_weight, u_weight]
    ).astype(np.float64, copy=False)

    if np.any(neighbor_community < 0):
        raise RuntimeError(
            "Uma aresta da projeção possui extremo marcado como isolado."
        )

    # Chave única para o par (nó, comunidade do vizinho).
    keys = (
        endpoint_node * np.int64(n_communities)
        + neighbor_community
    )

    order = np.argsort(keys, kind="mergesort")

    keys_sorted = keys[order]
    weights_sorted = endpoint_weight[order]

    unique_keys, first_idx, counts = np.unique(
        keys_sorted,
        return_index=True,
        return_counts=True,
    )

    strength_by_group = np.add.reduceat(
        weights_sorted,
        first_idx,
    )

    node_of_group = (
        unique_keys // np.int64(n_communities)
    ).astype(np.int64)

    degree_square_sum = np.bincount(
        node_of_group,
        weights=counts.astype(np.float64) ** 2,
        minlength=n_nodes,
    )

    strength_square_sum = np.bincount(
        node_of_group,
        weights=strength_by_group ** 2,
        minlength=n_nodes,
    )

    total_degree = np.asarray(
        total_degree,
        dtype=np.float64,
    )
    total_strength = np.asarray(
        total_strength,
        dtype=np.float64,
    )

    p_degree = np.zeros(
        n_nodes,
        dtype=np.float64,
    )
    p_strength = np.zeros(
        n_nodes,
        dtype=np.float64,
    )

    degree_mask = total_degree > 0
    strength_mask = total_strength > 0

    p_degree[degree_mask] = (
        1.0
        - degree_square_sum[degree_mask]
        / (total_degree[degree_mask] ** 2)
    )

    p_strength[strength_mask] = (
        1.0
        - strength_square_sum[strength_mask]
        / (total_strength[strength_mask] ** 2)
    )

    # Proteção numérica.
    p_degree = np.clip(p_degree, 0.0, 1.0)
    p_strength = np.clip(p_strength, 0.0, 1.0)

    del (
        endpoint_node,
        neighbor_community,
        endpoint_weight,
        keys,
        order,
        keys_sorted,
        weights_sorted,
        unique_keys,
        first_idx,
        counts,
        strength_by_group,
        node_of_group,
        degree_square_sum,
        strength_square_sum,
    )
    gc.collect()

    return p_degree, p_strength


def guimera_amaral_roles(z, participation, isolated):
    """
    Classificação clássica de papéis de Guimerà-Amaral.

    Não hubs: z < 2.5
      R1: ultra-peripheral     P <= 0.05
      R2: peripheral          0.05 < P <= 0.62
      R3: non-hub connector   0.62 < P <= 0.80
      R4: non-hub kinless     P > 0.80

    Hubs: z >= 2.5
      R5: provincial hub      P <= 0.30
      R6: connector hub       0.30 < P <= 0.75
      R7: kinless hub         P > 0.75

    Isolados recebem "isolated".
    """
    z = np.asarray(z, dtype=np.float64)
    participation = np.asarray(
        participation,
        dtype=np.float64,
    )
    isolated = np.asarray(isolated, dtype=bool)

    role = np.full(
        len(z),
        "",
        dtype=object,
    )

    role[isolated] = "isolated"

    nonhub = (~isolated) & (z < 2.5)
    hub = (~isolated) & (z >= 2.5)

    role[
        nonhub & (participation <= 0.05)
    ] = "R1_ultra_peripheral"

    role[
        nonhub
        & (participation > 0.05)
        & (participation <= 0.62)
    ] = "R2_peripheral"

    role[
        nonhub
        & (participation > 0.62)
        & (participation <= 0.80)
    ] = "R3_nonhub_connector"

    role[
        nonhub & (participation > 0.80)
    ] = "R4_nonhub_kinless"

    role[
        hub & (participation <= 0.30)
    ] = "R5_provincial_hub"

    role[
        hub
        & (participation > 0.30)
        & (participation <= 0.75)
    ] = "R6_connector_hub"

    role[
        hub & (participation > 0.75)
    ] = "R7_kinless_hub"

    if np.any(role == ""):
        raise RuntimeError(
            "Alguns nós ativos não receberam papel modular."
        )

    return role


def role_count_table(role, algorithm):
    s = pd.Series(role, name="role")

    out = (
        s.value_counts(dropna=False)
        .rename_axis("role")
        .reset_index(name="count")
    )

    out["fraction_all_nodes"] = (
        out["count"] / len(role)
    )

    active_total = int(
        np.sum(np.asarray(role) != "isolated")
    )

    out["fraction_active_nodes"] = np.where(
        out["role"] == "isolated",
        np.nan,
        out["count"] / active_total,
    )

    out.insert(0, "algorithm", algorithm)

    return out


def build_module_stats(
    algorithm,
    membership,
    module_size,
    internal_degree,
    external_degree,
    internal_strength,
    external_strength,
    within_degree_z,
    within_strength_z,
    participation_degree,
    participation_strength,
    role,
):
    df = pd.DataFrame(
        {
            "community": membership,
            "module_size": module_size,
            "internal_degree": internal_degree,
            "external_degree": external_degree,
            "internal_strength": internal_strength,
            "external_strength": external_strength,
            "within_module_degree_z": within_degree_z,
            "within_module_strength_z": within_strength_z,
            "participation_degree": participation_degree,
            "participation_strength": participation_strength,
            "role": role,
        }
    )

    df = df[df["community"] >= 0].copy()

    module = (
        df.groupby(
            "community",
            sort=True,
            observed=True,
        )
        .agg(
            size=("module_size", "first"),
            internal_degree_sum=(
                "internal_degree",
                "sum",
            ),
            boundary_degree_sum=(
                "external_degree",
                "sum",
            ),
            internal_strength_sum=(
                "internal_strength",
                "sum",
            ),
            boundary_strength_sum=(
                "external_strength",
                "sum",
            ),
            mean_within_module_degree_z=(
                "within_module_degree_z",
                "mean",
            ),
            mean_within_module_strength_z=(
                "within_module_strength_z",
                "mean",
            ),
            mean_participation_degree=(
                "participation_degree",
                "mean",
            ),
            mean_participation_strength=(
                "participation_strength",
                "mean",
            ),
            max_participation_degree=(
                "participation_degree",
                "max",
            ),
        )
        .reset_index()
    )

    # Cada aresta interna aparece no grau/força dos dois extremos.
    module["internal_edges"] = (
        module["internal_degree_sum"] / 2.0
    )
    module["internal_weight"] = (
        module["internal_strength_sum"] / 2.0
    )

    # Uma aresta de fronteira aparece exatamente uma vez para cada
    # módulo incidente; portanto não dividimos por dois por módulo.
    module["boundary_edges"] = (
        module["boundary_degree_sum"]
    )
    module["boundary_weight"] = (
        module["boundary_strength_sum"]
    )

    # Fração da incidência local de arestas/força que permanece no módulo.
    degree_denom = (
        2.0 * module["internal_edges"]
        + module["boundary_edges"]
    )
    strength_denom = (
        2.0 * module["internal_weight"]
        + module["boundary_weight"]
    )

    module["internal_degree_fraction"] = np.divide(
        2.0 * module["internal_edges"],
        degree_denom,
        out=np.zeros(
            len(module),
            dtype=np.float64,
        ),
        where=degree_denom.to_numpy() > 0,
    )

    module["internal_strength_fraction"] = np.divide(
        2.0 * module["internal_weight"],
        strength_denom,
        out=np.zeros(
            len(module),
            dtype=np.float64,
        ),
        where=strength_denom.to_numpy() > 0,
    )

    # Número de hubs/connector hubs por módulo.
    role_counts = (
        df.groupby(
            ["community", "role"],
            observed=True,
        )
        .size()
        .unstack(fill_value=0)
    )

    for role_name in [
        "R5_provincial_hub",
        "R6_connector_hub",
        "R7_kinless_hub",
    ]:
        if role_name in role_counts.columns:
            role_series = role_counts[
                role_name
            ]
        else:
            role_series = pd.Series(
                0,
                index=role_counts.index,
            )

        module[role_name + "_count"] = (
            module["community"]
            .map(role_series)
            .fillna(0)
            .astype(np.int64)
        )

    module.insert(
        0,
        "algorithm",
        algorithm,
    )

    preferred = [
        "algorithm",
        "community",
        "size",
        "internal_edges",
        "boundary_edges",
        "internal_weight",
        "boundary_weight",
        "internal_degree_fraction",
        "internal_strength_fraction",
        "mean_participation_degree",
        "mean_participation_strength",
        "max_participation_degree",
        "R5_provincial_hub_count",
        "R6_connector_hub_count",
        "R7_kinless_hub_count",
    ]

    remaining = [
        c
        for c in module.columns
        if c not in preferred
        and c not in {
            "internal_degree_sum",
            "boundary_degree_sum",
            "internal_strength_sum",
            "boundary_strength_sum",
        }
    ]

    return module[
        preferred + remaining
    ].sort_values(
        ["size", "community"],
        ascending=[False, True],
    ).reset_index(drop=True)


def summarize_algorithm(
    algorithm,
    membership,
    isolated,
    u_src,
    u_dst,
    u_weight,
    internal_degree,
    external_degree,
    internal_strength,
    external_strength,
    within_degree_z,
    within_strength_z,
    participation_degree,
    participation_strength,
    role,
):
    active = ~isolated

    edge_same = (
        membership[u_src] == membership[u_dst]
    )

    internal_edges = int(edge_same.sum())
    external_edges = int((~edge_same).sum())

    internal_weight = float(
        u_weight[edge_same].sum()
    )
    external_weight = float(
        u_weight[~edge_same].sum()
    )

    active_membership = membership[
        membership >= 0
    ]

    n_communities = int(
        len(np.unique(active_membership))
    )

    role_counts = pd.Series(
        role[active]
    ).value_counts()

    return {
        "algorithm": algorithm,
        "active_nodes": int(active.sum()),
        "isolated_nodes": int(isolated.sum()),
        "communities": n_communities,
        "undirected_edges": int(
            len(u_src)
        ),
        "internal_edges": internal_edges,
        "external_edges": external_edges,
        "internal_edge_fraction": float(
            internal_edges / len(u_src)
        ),
        "external_edge_fraction": float(
            external_edges / len(u_src)
        ),
        "undirected_total_weight": float(
            u_weight.sum()
        ),
        "internal_weight": internal_weight,
        "external_weight": external_weight,
        "internal_weight_fraction": float(
            internal_weight / u_weight.sum()
        ),
        "external_weight_fraction": float(
            external_weight / u_weight.sum()
        ),
        "mean_internal_degree_active": float(
            internal_degree[active].mean()
        ),
        "mean_external_degree_active": float(
            external_degree[active].mean()
        ),
        "mean_internal_strength_active": float(
            internal_strength[active].mean()
        ),
        "mean_external_strength_active": float(
            external_strength[active].mean()
        ),
        "mean_participation_degree_active": float(
            participation_degree[active].mean()
        ),
        "median_participation_degree_active": float(
            np.median(
                participation_degree[active]
            )
        ),
        "p90_participation_degree_active": float(
            np.quantile(
                participation_degree[active],
                0.90,
            )
        ),
        "mean_participation_strength_active": float(
            participation_strength[active].mean()
        ),
        "median_participation_strength_active": float(
            np.median(
                participation_strength[active]
            )
        ),
        "p90_participation_strength_active": float(
            np.quantile(
                participation_strength[active],
                0.90,
            )
        ),
        "max_within_module_degree_z": float(
            within_degree_z[active].max()
        ),
        "max_within_module_strength_z": float(
            within_strength_z[active].max()
        ),
        "provincial_hubs_R5": int(
            role_counts.get(
                "R5_provincial_hub",
                0,
            )
        ),
        "connector_hubs_R6": int(
            role_counts.get(
                "R6_connector_hub",
                0,
            )
        ),
        "kinless_hubs_R7": int(
            role_counts.get(
                "R7_kinless_hub",
                0,
            )
        ),
    }


# ============================================================
# Cálculo por partição
# ============================================================

def calculate_partition_roles(
    algorithm,
    membership,
    isolated,
    d_src,
    d_dst,
    d_weight,
    u_src,
    u_dst,
    u_weight,
    undirected_degree,
    undirected_strength,
):
    section(
        f"Papéis modulares — {algorithm.upper()}"
    )

    start = time.time()

    membership = np.asarray(
        membership,
        dtype=np.int64,
    )

    n_nodes = len(membership)

    if np.any(
        (membership < 0) != isolated
    ):
        raise RuntimeError(
            f"{algorithm}: membership=-1 não coincide exatamente "
            "com os isolados."
        )

    # --------------------------------------------------------
    # 1. Direção: entrada/saída interna e externa
    # --------------------------------------------------------

    same_directed = (
        membership[d_src]
        == membership[d_dst]
    )

    if np.any(
        membership[d_src] < 0
    ) or np.any(
        membership[d_dst] < 0
    ):
        raise RuntimeError(
            f"{algorithm}: uma aresta dirigida possui extremo isolado."
        )

    out_internal_degree = np.bincount(
        d_src[same_directed],
        minlength=n_nodes,
    ).astype(np.int64)

    in_internal_degree = np.bincount(
        d_dst[same_directed],
        minlength=n_nodes,
    ).astype(np.int64)

    out_external_degree = np.bincount(
        d_src[~same_directed],
        minlength=n_nodes,
    ).astype(np.int64)

    in_external_degree = np.bincount(
        d_dst[~same_directed],
        minlength=n_nodes,
    ).astype(np.int64)

    out_internal_strength = np.bincount(
        d_src[same_directed],
        weights=d_weight[same_directed],
        minlength=n_nodes,
    )

    in_internal_strength = np.bincount(
        d_dst[same_directed],
        weights=d_weight[same_directed],
        minlength=n_nodes,
    )

    out_external_strength = np.bincount(
        d_src[~same_directed],
        weights=d_weight[~same_directed],
        minlength=n_nodes,
    )

    in_external_strength = np.bincount(
        d_dst[~same_directed],
        weights=d_weight[~same_directed],
        minlength=n_nodes,
    )

    # --------------------------------------------------------
    # 2. Projeção não dirigida: interno/externo
    # --------------------------------------------------------

    same_undirected = (
        membership[u_src]
        == membership[u_dst]
    )

    internal_endpoints = np.concatenate(
        [
            u_src[same_undirected],
            u_dst[same_undirected],
        ]
    )

    external_endpoints = np.concatenate(
        [
            u_src[~same_undirected],
            u_dst[~same_undirected],
        ]
    )

    internal_degree = np.bincount(
        internal_endpoints,
        minlength=n_nodes,
    ).astype(np.int64)

    external_degree = np.bincount(
        external_endpoints,
        minlength=n_nodes,
    ).astype(np.int64)

    internal_weights_twice = np.concatenate(
        [
            u_weight[same_undirected],
            u_weight[same_undirected],
        ]
    )

    external_weights_twice = np.concatenate(
        [
            u_weight[~same_undirected],
            u_weight[~same_undirected],
        ]
    )

    internal_strength = np.bincount(
        internal_endpoints,
        weights=internal_weights_twice,
        minlength=n_nodes,
    )

    external_strength = np.bincount(
        external_endpoints,
        weights=external_weights_twice,
        minlength=n_nodes,
    )

    del (
        internal_endpoints,
        external_endpoints,
        internal_weights_twice,
        external_weights_twice,
    )
    gc.collect()

    # Sanidade: partição interno+externo recompõe a rede.
    if not np.array_equal(
        internal_degree + external_degree,
        undirected_degree,
    ):
        raise RuntimeError(
            f"{algorithm}: grau interno + externo não recompõe "
            "o grau não dirigido."
        )

    if not np.allclose(
        internal_strength + external_strength,
        undirected_strength,
        rtol=0.0,
        atol=1e-9,
    ):
        raise RuntimeError(
            f"{algorithm}: força interna + externa não recompõe "
            "a força da projeção."
        )

    # --------------------------------------------------------
    # 3. Tamanho do módulo por nó
    # --------------------------------------------------------

    active_membership = membership[
        membership >= 0
    ]

    n_communities = int(
        active_membership.max()
    ) + 1

    community_sizes = np.bincount(
        active_membership,
        minlength=n_communities,
    ).astype(np.int64)

    module_size = np.zeros(
        n_nodes,
        dtype=np.int64,
    )

    active = membership >= 0
    module_size[active] = (
        community_sizes[
            membership[active]
        ]
    )

    # --------------------------------------------------------
    # 4. Within-module z
    # --------------------------------------------------------

    within_degree_z = within_module_z(
        internal_degree,
        membership,
    )

    within_strength_z = within_module_z(
        internal_strength,
        membership,
    )

    # --------------------------------------------------------
    # 5. Participation coefficient
    # --------------------------------------------------------

    print(
        "Calculando participation coefficient "
        "(esta é a etapa mais pesada)..."
    )

    participation_degree, participation_strength = (
        participation_coefficients(
            u_src=u_src,
            u_dst=u_dst,
            u_weight=u_weight,
            membership=membership,
            total_degree=undirected_degree,
            total_strength=undirected_strength,
        )
    )

    # --------------------------------------------------------
    # 6. Papel clássico R1-R7
    # --------------------------------------------------------

    role = guimera_amaral_roles(
        z=within_degree_z,
        participation=participation_degree,
        isolated=isolated,
    )

    # --------------------------------------------------------
    # 7. Frações internas
    # --------------------------------------------------------

    internal_degree_fraction = safe_ratio(
        internal_degree,
        undirected_degree,
    )

    external_degree_fraction = safe_ratio(
        external_degree,
        undirected_degree,
    )

    internal_strength_fraction = safe_ratio(
        internal_strength,
        undirected_strength,
    )

    external_strength_fraction = safe_ratio(
        external_strength,
        undirected_strength,
    )

    # --------------------------------------------------------
    # 8. Tabela de saída desta partição
    # --------------------------------------------------------

    prefix = algorithm

    out = pd.DataFrame(
        {
            f"{prefix}_community": membership,
            f"{prefix}_module_size": module_size,

            f"{prefix}_internal_degree":
                internal_degree,
            f"{prefix}_external_degree":
                external_degree,

            f"{prefix}_internal_degree_fraction":
                internal_degree_fraction,
            f"{prefix}_external_degree_fraction":
                external_degree_fraction,

            f"{prefix}_internal_strength":
                internal_strength,
            f"{prefix}_external_strength":
                external_strength,

            f"{prefix}_internal_strength_fraction":
                internal_strength_fraction,
            f"{prefix}_external_strength_fraction":
                external_strength_fraction,

            f"{prefix}_within_module_degree_z":
                within_degree_z,
            f"{prefix}_within_module_strength_z":
                within_strength_z,

            f"{prefix}_participation_degree":
                participation_degree,
            f"{prefix}_participation_strength":
                participation_strength,

            f"{prefix}_role":
                role,

            # Versões direcionais — preservam informação de entrada/saída.
            f"{prefix}_in_internal_degree":
                in_internal_degree,
            f"{prefix}_in_external_degree":
                in_external_degree,
            f"{prefix}_out_internal_degree":
                out_internal_degree,
            f"{prefix}_out_external_degree":
                out_external_degree,

            f"{prefix}_in_internal_strength":
                in_internal_strength,
            f"{prefix}_in_external_strength":
                in_external_strength,
            f"{prefix}_out_internal_strength":
                out_internal_strength,
            f"{prefix}_out_external_strength":
                out_external_strength,
        }
    )

    module_stats = build_module_stats(
        algorithm=algorithm,
        membership=membership,
        module_size=module_size,
        internal_degree=internal_degree,
        external_degree=external_degree,
        internal_strength=internal_strength,
        external_strength=external_strength,
        within_degree_z=within_degree_z,
        within_strength_z=within_strength_z,
        participation_degree=participation_degree,
        participation_strength=participation_strength,
        role=role,
    )

    role_counts = role_count_table(
        role=role,
        algorithm=algorithm,
    )

    summary = summarize_algorithm(
        algorithm=algorithm,
        membership=membership,
        isolated=isolated,
        u_src=u_src,
        u_dst=u_dst,
        u_weight=u_weight,
        internal_degree=internal_degree,
        external_degree=external_degree,
        internal_strength=internal_strength,
        external_strength=external_strength,
        within_degree_z=within_degree_z,
        within_strength_z=within_strength_z,
        participation_degree=participation_degree,
        participation_strength=participation_strength,
        role=role,
    )

    print(
        f"Comunidades:               "
        f"{fmt_int(summary['communities'])}"
    )
    print(
        f"Arestas internas:          "
        f"{fmt_int(summary['internal_edges'])} "
        f"({100 * summary['internal_edge_fraction']:.2f}%)"
    )
    print(
        f"Peso interno:              "
        f"{summary['internal_weight']:,.0f} "
        f"({100 * summary['internal_weight_fraction']:.2f}%)"
    )
    print(
        f"Participation médio:       "
        f"{summary['mean_participation_degree_active']:.6f}"
    )
    print(
        f"Participation ponderado:   "
        f"{summary['mean_participation_strength_active']:.6f}"
    )
    print(
        f"Connector hubs (R6):       "
        f"{fmt_int(summary['connector_hubs_R6'])}"
    )
    print(
        f"Tempo:                      "
        f"{elapsed(start)}"
    )

    return out, module_stats, role_counts, summary


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    total_start = time.time()

    section("F2.04 — Papéis modulares")

    print(
        "Algoritmos: "
        + ", ".join(args.algorithms)
    )

    for path in [
        EDGES_PATH,
        NODE_FEATURES_PATH,
        ASSIGNMENTS_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {path}"
            )

    # --------------------------------------------------------
    # 1. Carrega universo de nós e assignments
    # --------------------------------------------------------

    section("1. Carregando nós e comunidades")

    start = time.time()

    nodes = pd.read_csv(
        NODE_FEATURES_PATH,
        usecols=[
            "root_id",
            "undirected_degree",
            "isolated",
        ],
        dtype={
            "root_id": "int64",
            "undirected_degree": "int64",
            "isolated": "bool",
        },
    )

    assignments = pd.read_csv(
        ASSIGNMENTS_PATH,
        dtype={
            "root_id": "int64",
            "isolated": "bool",
            "leiden_community": "int64",
            "infomap_community": "int64",
        },
    )

    if nodes["root_id"].duplicated().any():
        raise RuntimeError(
            "node_structural_features possui root_id duplicado."
        )

    if assignments["root_id"].duplicated().any():
        raise RuntimeError(
            "community_assignments possui root_id duplicado."
        )

    base = nodes.merge(
        assignments[
            [
                "root_id",
                "isolated",
                "leiden_community",
                "infomap_community",
            ]
        ],
        on="root_id",
        how="left",
        validate="one_to_one",
        suffixes=(
            "_structural",
            "_community",
        ),
    )

    if base[
        [
            "leiden_community",
            "infomap_community",
        ]
    ].isna().any().any():
        raise RuntimeError(
            "Há nós sem assignment de comunidade."
        )

    if not np.array_equal(
        base["isolated_structural"].to_numpy(),
        base["isolated_community"].to_numpy(),
    ):
        raise RuntimeError(
            "Indicador de isolado diverge entre F2.01 e F2.03."
        )

    base = base.rename(
        columns={
            "isolated_structural": "isolated",
        }
    ).drop(
        columns=["isolated_community"]
    )

    n_nodes = len(base)

    root_ids = base[
        "root_id"
    ].to_numpy(dtype=np.int64)

    isolated = base[
        "isolated"
    ].to_numpy(dtype=bool)

    undirected_degree_expected = base[
        "undirected_degree"
    ].to_numpy(dtype=np.int64)

    print(
        f"Nós:       {fmt_int(n_nodes)}"
    )
    print(
        f"Ativos:    {fmt_int((~isolated).sum())}"
    )
    print(
        f"Isolados:  {fmt_int(isolated.sum())}"
    )
    print(
        f"Tempo:     {elapsed(start)}"
    )

    # --------------------------------------------------------
    # 2. Carrega grafo dirigido
    # --------------------------------------------------------

    section("2. Carregando arestas dirigidas")

    start = time.time()

    edges = pd.read_csv(
        EDGES_PATH,
        usecols=[
            "pre_root_id",
            "post_root_id",
            "weight",
        ],
        dtype={
            "pre_root_id": "int64",
            "post_root_id": "int64",
            "weight": "int64",
        },
    )

    node_index = pd.Index(root_ids)

    d_src = node_index.get_indexer(
        edges["pre_root_id"].to_numpy(
            dtype=np.int64
        )
    ).astype(np.int64)

    d_dst = node_index.get_indexer(
        edges["post_root_id"].to_numpy(
            dtype=np.int64
        )
    ).astype(np.int64)

    d_weight = edges[
        "weight"
    ].to_numpy(dtype=np.float64)

    if np.any(d_src < 0) or np.any(d_dst < 0):
        raise RuntimeError(
            "Há extremos de aresta ausentes do universo de nós."
        )

    if np.any(d_src == d_dst):
        raise RuntimeError(
            "Foram encontrados self-loops no grafo processado."
        )

    print(
        f"Arestas dirigidas: {fmt_int(len(d_src))}"
    )
    print(
        f"Peso total:        "
        f"{d_weight.sum():,.0f}"
    )
    print(
        f"Tempo:             {elapsed(start)}"
    )

    del edges
    gc.collect()

    # --------------------------------------------------------
    # 3. Projeção simples não dirigida ponderada
    # --------------------------------------------------------

    section(
        "3. Construindo projeção não dirigida ponderada"
    )

    start = time.time()

    pair_u = np.minimum(
        d_src,
        d_dst,
    )
    pair_v = np.maximum(
        d_src,
        d_dst,
    )

    undirected = pd.DataFrame(
        {
            "u": pair_u,
            "v": pair_v,
            "weight": d_weight,
        }
    )

    undirected = (
        undirected.groupby(
            ["u", "v"],
            sort=False,
            as_index=False,
            observed=True,
        )
        .agg(
            weight=("weight", "sum")
        )
    )

    u_src = undirected[
        "u"
    ].to_numpy(dtype=np.int64)

    u_dst = undirected[
        "v"
    ].to_numpy(dtype=np.int64)

    u_weight = undirected[
        "weight"
    ].to_numpy(dtype=np.float64)

    del undirected, pair_u, pair_v
    gc.collect()

    undirected_degree = np.bincount(
        np.concatenate(
            [u_src, u_dst]
        ),
        minlength=n_nodes,
    ).astype(np.int64)

    if not np.array_equal(
        undirected_degree,
        undirected_degree_expected,
    ):
        raise RuntimeError(
            "A projeção reconstruída não reproduz undirected_degree "
            "de node_structural_features."
        )

    undirected_strength = np.bincount(
        np.concatenate(
            [u_src, u_dst]
        ),
        weights=np.concatenate(
            [u_weight, u_weight]
        ),
        minlength=n_nodes,
    )

    if not np.array_equal(
        undirected_degree == 0,
        isolated,
    ):
        raise RuntimeError(
            "Isolados não coincidem com grau não dirigido zero."
        )

    print(
        f"Arestas não dirigidas: {fmt_int(len(u_src))}"
    )
    print(
        f"Peso total único:       "
        f"{u_weight.sum():,.0f}"
    )
    print(
        f"Tempo:                  {elapsed(start)}"
    )

    # --------------------------------------------------------
    # 4. Calcula métricas por partição
    # --------------------------------------------------------

    results = pd.DataFrame(
        {
            "root_id": root_ids,
            "isolated": isolated,
        }
    )

    summaries = []

    for algorithm in args.algorithms:
        membership = base[
            f"{algorithm}_community"
        ].to_numpy(dtype=np.int64)

        (
            algorithm_df,
            module_stats,
            role_counts,
            summary,
        ) = calculate_partition_roles(
            algorithm=algorithm,
            membership=membership,
            isolated=isolated,
            d_src=d_src,
            d_dst=d_dst,
            d_weight=d_weight,
            u_src=u_src,
            u_dst=u_dst,
            u_weight=u_weight,
            undirected_degree=undirected_degree,
            undirected_strength=undirected_strength,
        )

        results = pd.concat(
            [
                results.reset_index(drop=True),
                algorithm_df.reset_index(drop=True),
            ],
            axis=1,
        )

        summaries.append(summary)

        if algorithm == "leiden":
            module_stats.to_csv(
                LEIDEN_MODULE_STATS_PATH,
                index=False,
            )
            role_counts.to_csv(
                LEIDEN_ROLE_COUNTS_PATH,
                index=False,
            )

        elif algorithm == "infomap":
            module_stats.to_csv(
                INFOMAP_MODULE_STATS_PATH,
                index=False,
            )
            role_counts.to_csv(
                INFOMAP_ROLE_COUNTS_PATH,
                index=False,
            )

        del (
            algorithm_df,
            module_stats,
            role_counts,
            membership,
        )
        gc.collect()

    # --------------------------------------------------------
    # 5. Auditoria final da tabela por nó
    # --------------------------------------------------------

    section("5. Auditoria da tabela final")

    if len(results) != n_nodes:
        raise RuntimeError(
            "Número de linhas da tabela final mudou."
        )

    if results["root_id"].duplicated().any():
        raise RuntimeError(
            "Há root_id duplicado na tabela final."
        )

    numeric = results.select_dtypes(
        include=[np.number]
    )

    numeric_values = numeric.to_numpy(
        dtype=np.float64,
        copy=False,
    )

    if not np.isfinite(
        numeric_values
    ).all():
        raise RuntimeError(
            "Há NaN ou infinito em colunas numéricas."
        )

    # Participation deve estar em [0,1].
    for algorithm in args.algorithms:
        for suffix in [
            "participation_degree",
            "participation_strength",
        ]:
            col = (
                f"{algorithm}_{suffix}"
            )

            if (
                results[col].min() < -1e-12
                or results[col].max() > 1 + 1e-12
            ):
                raise RuntimeError(
                    f"{col} saiu do intervalo [0,1]."
                )

        # Isolados: métricas quantitativas devem ser zero.
        isolated_rows = results[
            results["isolated"]
        ]

        quantitative_cols = [
            c
            for c in results.columns
            if c.startswith(
                f"{algorithm}_"
            )
            and c not in {
                f"{algorithm}_community",
                f"{algorithm}_role",
            }
        ]

        for col in quantitative_cols:
            if not np.allclose(
                isolated_rows[
                    col
                ].to_numpy(
                    dtype=np.float64
                ),
                0.0,
            ):
                raise RuntimeError(
                    f"Isolados possuem valores não zero em {col}."
                )

        if not np.all(
            isolated_rows[
                f"{algorithm}_community"
            ].to_numpy() == -1
        ):
            raise RuntimeError(
                f"Isolados não possuem {algorithm}_community=-1."
            )

        if not np.all(
            isolated_rows[
                f"{algorithm}_role"
            ].to_numpy() == "isolated"
        ):
            raise RuntimeError(
                f"Isolados não possuem papel 'isolated' em {algorithm}."
            )

    print(
        f"Linhas:            {fmt_int(len(results))}"
    )
    print(
        f"Colunas:           {fmt_int(len(results.columns))}"
    )
    print(
        f"root_id duplicado: {results['root_id'].duplicated().sum()}"
    )
    print(
        "NaN/infinito:      0"
    )

    # --------------------------------------------------------
    # 6. Salva tabela e resumos
    # --------------------------------------------------------

    section("6. Salvando resultados")

    results.to_csv(
        ROLES_PATH,
        index=False,
        compression="gzip",
    )

    summary_df = pd.DataFrame(
        summaries
    )

    summary_df.to_csv(
        SUMMARY_CSV_PATH,
        index=False,
    )

    summary_json = {
        "methodology": {
            "network_for_modular_roles": (
                "simple weighted undirected projection"
            ),
            "directed_internal_external_metrics": (
                "also calculated from original directed graph"
            ),
            "within_module_z": (
                "population z-score within each detected community"
            ),
            "participation_coefficient": (
                "Guimera-Amaral; degree and weighted-strength versions"
            ),
            "role_thresholds": {
                "hub_threshold_z": 2.5,
                "R1_max_P": 0.05,
                "R2_max_P": 0.62,
                "R3_max_P": 0.80,
                "R5_max_P": 0.30,
                "R6_max_P": 0.75,
            },
            "biology_used": False,
            "target_labels_used": False,
        },
        "graph": {
            "nodes": int(n_nodes),
            "active_nodes": int(
                (~isolated).sum()
            ),
            "isolated_nodes": int(
                isolated.sum()
            ),
            "directed_edges": int(
                len(d_src)
            ),
            "undirected_edges": int(
                len(u_src)
            ),
            "directed_total_weight": float(
                d_weight.sum()
            ),
            "undirected_total_weight": float(
                u_weight.sum()
            ),
        },
        "algorithms": {
            row["algorithm"]: {
                key: (
                    int(value)
                    if isinstance(
                        value,
                        (np.integer,)
                    )
                    else float(value)
                    if isinstance(
                        value,
                        (np.floating,)
                    )
                    else value
                )
                for key, value in row.items()
                if key != "algorithm"
            }
            for row in summaries
        },
        "output": {
            "node_modular_roles": (
                ROLES_PATH.relative_to(
                    ROOT
                ).as_posix()
            ),
            "leiden_module_stats": (
                LEIDEN_MODULE_STATS_PATH.relative_to(
                    ROOT
                ).as_posix()
            ),
            "infomap_module_stats": (
                INFOMAP_MODULE_STATS_PATH.relative_to(
                    ROOT
                ).as_posix()
            ),
        },
        "runtime_seconds": float(
            time.time() - total_start
        ),
    }

    with SUMMARY_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary_json,
            f,
            indent=2,
            ensure_ascii=False,
        )

    lines = [
        "FlyWire FAFB v783 — papéis modulares",
        "",
        "METODOLOGIA",
        (
            "Papéis modulares calculados na projeção simples, "
            "não dirigida e ponderada."
        ),
        (
            "Também foram preservadas métricas direcionais de "
            "entrada/saída internas e externas."
        ),
        (
            "Within-module z-score calculado dentro de cada "
            "comunidade com desvio-padrão populacional."
        ),
        (
            "Participation coefficient calculado em versões por "
            "grau e por força."
        ),
        (
            "Papéis R1-R7 seguem os limiares clássicos de "
            "Guimerà-Amaral."
        ),
        (
            "Nenhuma variável biológica ou super_class foi usada."
        ),
        "",
        "GRAFO",
        f"Nós: {n_nodes}",
        f"Nós ativos: {(~isolated).sum()}",
        f"Isolados: {isolated.sum()}",
        f"Arestas dirigidas: {len(d_src)}",
        f"Arestas não dirigidas: {len(u_src)}",
        "",
    ]

    for summary in summaries:
        algorithm = summary[
            "algorithm"
        ].upper()

        lines.extend(
            [
                algorithm,
                f"Comunidades: {summary['communities']}",
                (
                    "Fração de arestas internas: "
                    f"{summary['internal_edge_fraction']:.6f}"
                ),
                (
                    "Fração do peso interna: "
                    f"{summary['internal_weight_fraction']:.6f}"
                ),
                (
                    "Participation degree médio: "
                    f"{summary['mean_participation_degree_active']:.6f}"
                ),
                (
                    "Participation degree mediano: "
                    f"{summary['median_participation_degree_active']:.6f}"
                ),
                (
                    "Participation strength médio: "
                    f"{summary['mean_participation_strength_active']:.6f}"
                ),
                (
                    "Participation strength mediano: "
                    f"{summary['median_participation_strength_active']:.6f}"
                ),
                (
                    "Maior z intramodular (grau): "
                    f"{summary['max_within_module_degree_z']:.6f}"
                ),
                (
                    "Maior z intramodular (força): "
                    f"{summary['max_within_module_strength_z']:.6f}"
                ),
                (
                    "R5 provincial hubs: "
                    f"{summary['provincial_hubs_R5']}"
                ),
                (
                    "R6 connector hubs: "
                    f"{summary['connector_hubs_R6']}"
                ),
                (
                    "R7 kinless hubs: "
                    f"{summary['kinless_hubs_R7']}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            (
                "Tempo total: "
                f"{(time.time() - total_start) / 60:.2f} min"
            ),
        ]
    )

    SUMMARY_TXT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        f"Tabela por nó:       {ROLES_PATH}"
    )
    print(
        f"Resumo:               {SUMMARY_TXT_PATH}"
    )

    if "leiden" in args.algorithms:
        print(
            f"Stats Leiden:         "
            f"{LEIDEN_MODULE_STATS_PATH}"
        )
        print(
            f"Papéis Leiden:        "
            f"{LEIDEN_ROLE_COUNTS_PATH}"
        )

    if "infomap" in args.algorithms:
        print(
            f"Stats Infomap:        "
            f"{INFOMAP_MODULE_STATS_PATH}"
        )
        print(
            f"Papéis Infomap:       "
            f"{INFOMAP_ROLE_COUNTS_PATH}"
        )

    section("Concluído")

    print(
        f"Tempo total: "
        f"{(time.time() - total_start) / 60:.2f} min"
    )


if __name__ == "__main__":
    main()
