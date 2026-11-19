from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd


# ============================================================
# Caminhos
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_DIR = ROOT / "reports" / "f2" / "features"
SPLIT_DIR = ROOT / "reports" / "f2" / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

BIO_PATH = FEATURE_DIR / "node_biological_features.csv.gz"
TOPO_PATH = FEATURE_DIR / "node_topological_features.csv.gz"
TARGET_PATH = FEATURE_DIR / "node_targets.csv.gz"

MAIN_SPLIT_PATH = SPLIT_DIR / "main_split.csv.gz"
MAIN_COUNTS_PATH = SPLIT_DIR / "main_split_counts.csv"

LEFT_TO_RIGHT_PATH = SPLIT_DIR / "hemisphere_left_to_right.csv.gz"
RIGHT_TO_LEFT_PATH = SPLIT_DIR / "hemisphere_right_to_left.csv.gz"
HEMISPHERE_COUNTS_PATH = SPLIT_DIR / "hemisphere_split_counts.csv"

SUMMARY_TXT_PATH = SPLIT_DIR / "splits_summary.txt"
SUMMARY_JSON_PATH = SPLIT_DIR / "splits_summary.json"


# ============================================================
# Configuração experimental
# ============================================================

SEED = 42

MAIN_FRACTIONS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

# Para o teste hemisférico:
# - todo o hemisfério de destino é TESTE;
# - o hemisfério de origem é dividido internamente em treino/validação.
HEMISPHERE_SOURCE_FRACTIONS = {
    "train": 0.85,
    "val": 0.15,
}

HEMISPHERE_SIDES = {
    "left_to_right": {
        "source": "left",
        "target": "right",
    },
    "right_to_left": {
        "source": "right",
        "target": "left",
    },
}


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


def largest_remainder_allocation(n, fractions, labels):
    """
    Converte proporções em contagens inteiras somando exatamente n.

    Usa o método do maior resto:
      1) floor(n * p)
      2) distribui as unidades restantes pelas maiores partes fracionárias.

    Em empates, a ordem de `labels` decide de forma determinística.
    """
    raw = [n * fractions[label] for label in labels]
    base = [math.floor(x) for x in raw]

    remaining = n - sum(base)

    order = sorted(
        range(len(labels)),
        key=lambda i: (
            -(raw[i] - base[i]),
            i,
        ),
    )

    for i in order[:remaining]:
        base[i] += 1

    allocation = {
        label: int(count)
        for label, count in zip(labels, base)
    }

    if sum(allocation.values()) != n:
        raise RuntimeError(
            "Falha na alocação inteira das proporções."
        )

    return allocation


def stratified_split(
    df,
    label_col,
    fractions,
    split_labels,
    seed,
):
    """
    Split estratificado determinístico sem depender de scikit-learn.

    Cada classe é embaralhada separadamente com RNG fixo, depois dividida
    pelas proporções solicitadas. Todas as linhas recebem exatamente um split.
    """
    rng = np.random.default_rng(seed)

    assignments = np.empty(
        len(df),
        dtype=object,
    )

    label_array = df[label_col].to_numpy()

    for label in sorted(df[label_col].unique()):
        idx = np.flatnonzero(label_array == label)

        shuffled = rng.permutation(idx)

        allocation = largest_remainder_allocation(
            n=len(idx),
            fractions=fractions,
            labels=split_labels,
        )

        cursor = 0

        for split_name in split_labels:
            count = allocation[split_name]

            selected = shuffled[
                cursor : cursor + count
            ]

            assignments[selected] = split_name
            cursor += count

        if cursor != len(idx):
            raise RuntimeError(
                f"Falha ao dividir classe {label}."
            )

    if pd.isna(assignments).any():
        raise RuntimeError(
            "Algumas linhas não receberam split."
        )

    return assignments


def summarize_split(df, split_col, design):
    counts = (
        df.groupby(
            ["super_class", split_col],
            observed=True,
        )
        .size()
        .rename("count")
        .reset_index()
    )

    counts.insert(0, "design", design)

    total_by_class = (
        counts.groupby(
            ["design", "super_class"],
            observed=True,
        )["count"]
        .transform("sum")
    )

    counts["fraction_within_superclass"] = (
        counts["count"] / total_by_class
    )

    return counts


def assert_all_classes_present(
    df,
    split_col,
    required_splits,
):
    classes = set(df["super_class"].unique())

    for split_name in required_splits:
        present = set(
            df.loc[
                df[split_col] == split_name,
                "super_class",
            ].unique()
        )

        missing = sorted(classes - present)

        if missing:
            raise RuntimeError(
                f"Split {split_name} não contém classes: {missing}"
            )


def class_count_matrix(df, split_col):
    return (
        pd.crosstab(
            df["super_class"],
            df[split_col],
            dropna=False,
        )
        .sort_index()
    )


def build_hemisphere_split(
    base,
    source_side,
    target_side,
    seed,
    design_name,
):
    """
    Robustez hemisférica:
      - source_side -> train/val estratificados 85/15;
      - target_side -> test integral;
      - center/unknown -> excluded.

    `side` é excluída das features durante a modelagem deste desenho.
    """
    result = base[
        ["root_id", "super_class", "side"]
    ].copy()

    result["split"] = "excluded"

    source_mask = (
        result["side"] == source_side
    )
    target_mask = (
        result["side"] == target_side
    )

    source_df = result.loc[
        source_mask,
        ["root_id", "super_class", "side"],
    ].copy()

    source_assignments = stratified_split(
        df=source_df,
        label_col="super_class",
        fractions=HEMISPHERE_SOURCE_FRACTIONS,
        split_labels=["train", "val"],
        seed=seed,
    )

    result.loc[
        source_mask,
        "split",
    ] = source_assignments

    result.loc[
        target_mask,
        "split",
    ] = "test"

    # Auditoria: todos os lados de origem/destino devem conter as 10 classes.
    included = result[
        result["split"] != "excluded"
    ].copy()

    assert_all_classes_present(
        included,
        split_col="split",
        required_splits=[
            "train",
            "val",
            "test",
        ],
    )

    # Garantias estruturais do desenho.
    if not (
        result.loc[
            result["split"].isin(
                ["train", "val"]
            ),
            "side",
        ]
        == source_side
    ).all():
        raise RuntimeError(
            f"{design_name}: train/val contém lado diferente de {source_side}."
        )

    if not (
        result.loc[
            result["split"] == "test",
            "side",
        ]
        == target_side
    ).all():
        raise RuntimeError(
            f"{design_name}: test contém lado diferente de {target_side}."
        )

    return result


# ============================================================
# Main
# ============================================================

def main():
    start = time.time()

    section("F2.08 — Preparação dos splits")

    for path in [
        BIO_PATH,
        TOPO_PATH,
        TARGET_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo necessário não encontrado: {path}"
            )

    # --------------------------------------------------------
    # 1. Carrega e valida universo
    # --------------------------------------------------------

    section("1. Carregando universo de neurônios")

    bio = pd.read_csv(
        BIO_PATH,
        usecols=["root_id", "side"],
        low_memory=False,
    )

    topo = pd.read_csv(
        TOPO_PATH,
        usecols=["root_id"],
        low_memory=False,
    )

    target = pd.read_csv(
        TARGET_PATH,
        usecols=["root_id", "super_class"],
        low_memory=False,
    )

    for name, df in [
        ("BIO", bio),
        ("REDE", topo),
        ("TARGET", target),
    ]:
        if df["root_id"].duplicated().any():
            raise RuntimeError(
                f"{name} possui root_id duplicado."
            )

    bio_ids = set(
        bio["root_id"].astype("int64")
    )
    topo_ids = set(
        topo["root_id"].astype("int64")
    )
    target_ids = set(
        target["root_id"].astype("int64")
    )

    if not (
        bio_ids
        == topo_ids
        == target_ids
    ):
        raise RuntimeError(
            "BIO, REDE e TARGET não possuem exatamente o mesmo universo de root_id."
        )

    base = (
        target.merge(
            bio,
            on="root_id",
            how="inner",
            validate="one_to_one",
        )
        .sort_values("root_id")
        .reset_index(drop=True)
    )

    n = len(base)
    classes = sorted(
        base["super_class"].unique()
    )

    print(f"Nós:      {fmt_int(n)}")
    print(f"Classes:  {len(classes)}")
    print(
        "Side:     "
        + ", ".join(
            f"{str(k)}={fmt_int(v)}"
            for k, v in base["side"]
            .fillna("unknown")
            .value_counts()
            .to_dict()
            .items()
        )
    )

    if len(classes) != 10:
        raise RuntimeError(
            f"Esperadas 10 classes, encontradas {len(classes)}."
        )

    # --------------------------------------------------------
    # 2. Split principal 70/15/15 estratificado
    # --------------------------------------------------------

    section("2. Split principal estratificado 70/15/15")

    main = base[
        ["root_id", "super_class"]
    ].copy()

    main["split"] = stratified_split(
        df=main,
        label_col="super_class",
        fractions=MAIN_FRACTIONS,
        split_labels=[
            "train",
            "val",
            "test",
        ],
        seed=SEED,
    )

    assert_all_classes_present(
        main,
        split_col="split",
        required_splits=[
            "train",
            "val",
            "test",
        ],
    )

    if main["root_id"].duplicated().any():
        raise RuntimeError(
            "main_split possui root_id duplicado."
        )

    main_counts = summarize_split(
        main,
        split_col="split",
        design="main_stratified_70_15_15",
    )

    main_matrix = class_count_matrix(
        main,
        split_col="split",
    )

    print(main_matrix.to_string())

    split_totals = (
        main["split"]
        .value_counts()
        .reindex(
            ["train", "val", "test"]
        )
    )

    print()
    for split_name, count in split_totals.items():
        print(
            f"{split_name:>5}: "
            f"{fmt_int(count)} "
            f"({100 * count / n:.4f}%)"
        )

    # --------------------------------------------------------
    # 3. Robustez hemisférica
    # --------------------------------------------------------

    section("3. Split hemisférico — left -> right")

    left_to_right = build_hemisphere_split(
        base=base,
        source_side="left",
        target_side="right",
        seed=SEED + 100,
        design_name="left_to_right",
    )

    print(
        class_count_matrix(
            left_to_right[
                left_to_right["split"] != "excluded"
            ],
            "split",
        ).to_string()
    )

    section("4. Split hemisférico — right -> left")

    right_to_left = build_hemisphere_split(
        base=base,
        source_side="right",
        target_side="left",
        seed=SEED + 200,
        design_name="right_to_left",
    )

    print(
        class_count_matrix(
            right_to_left[
                right_to_left["split"] != "excluded"
            ],
            "split",
        ).to_string()
    )

    # Contagens dos desenhos hemisféricos.
    hemi_counts = pd.concat(
        [
            summarize_split(
                left_to_right[
                    left_to_right["split"] != "excluded"
                ],
                split_col="split",
                design="hemisphere_left_to_right",
            ),
            summarize_split(
                right_to_left[
                    right_to_left["split"] != "excluded"
                ],
                split_col="split",
                design="hemisphere_right_to_left",
            ),
        ],
        ignore_index=True,
    )

    # Acrescenta excluídos por desenho.
    excluded_rows = []

    for design, df in [
        (
            "hemisphere_left_to_right",
            left_to_right,
        ),
        (
            "hemisphere_right_to_left",
            right_to_left,
        ),
    ]:
        exc = df[
            df["split"] == "excluded"
        ]

        for superclass, count in (
            exc["super_class"]
            .value_counts()
            .items()
        ):
            excluded_rows.append({
                "design": design,
                "super_class": superclass,
                "split": "excluded",
                "count": int(count),
                "fraction_within_superclass": np.nan,
            })

    if excluded_rows:
        hemi_counts = pd.concat(
            [
                hemi_counts,
                pd.DataFrame(excluded_rows),
            ],
            ignore_index=True,
        )

    # --------------------------------------------------------
    # 5. Auditorias finais
    # --------------------------------------------------------

    section("5. Auditorias finais")

    if len(main) != n:
        raise RuntimeError(
            "main_split perdeu linhas."
        )

    if len(left_to_right) != n:
        raise RuntimeError(
            "left_to_right perdeu linhas."
        )

    if len(right_to_left) != n:
        raise RuntimeError(
            "right_to_left perdeu linhas."
        )

    if set(main["root_id"]) != target_ids:
        raise RuntimeError(
            "main_split não contém exatamente o universo alvo."
        )

    # No split principal, nenhuma linha é excluída.
    if set(main["split"].unique()) != {
        "train",
        "val",
        "test",
    }:
        raise RuntimeError(
            "main_split contém labels inesperados."
        )

    # Nos splits hemisféricos, somente center/missing podem ser excluded.
    for design_name, df in [
        ("left_to_right", left_to_right),
        ("right_to_left", right_to_left),
    ]:
        excluded = df[
            df["split"] == "excluded"
        ]

        invalid = excluded[
            excluded["side"].isin(
                ["left", "right"]
            )
        ]

        if len(invalid):
            raise RuntimeError(
                f"{design_name}: há left/right indevidamente excluídos."
            )

    print("BIO / REDE / TARGET: mesmo universo")
    print("Split principal:      100% das linhas atribuídas")
    print("Todas as classes:     presentes em train/val/test")
    print("Hemisférico:          center/unknown excluídos somente na robustez")
    print("Side como feature:    excluída nos testes hemisféricos")

    # --------------------------------------------------------
    # 6. Salva
    # --------------------------------------------------------

    section("6. Salvando splits")

    main.to_csv(
        MAIN_SPLIT_PATH,
        index=False,
        compression="gzip",
    )

    main_counts.to_csv(
        MAIN_COUNTS_PATH,
        index=False,
    )

    left_to_right.to_csv(
        LEFT_TO_RIGHT_PATH,
        index=False,
        compression="gzip",
    )

    right_to_left.to_csv(
        RIGHT_TO_LEFT_PATH,
        index=False,
        compression="gzip",
    )

    hemi_counts.to_csv(
        HEMISPHERE_COUNTS_PATH,
        index=False,
    )

    main_totals = {
        split_name: int(count)
        for split_name, count in split_totals.items()
    }

    def hemi_totals(df):
        return {
            split_name: int(count)
            for split_name, count in (
                df["split"]
                .value_counts()
                .reindex(
                    [
                        "train",
                        "val",
                        "test",
                        "excluded",
                    ],
                    fill_value=0,
                )
                .items()
            )
        }

    summary = {
        "seed": SEED,
        "universe": {
            "nodes": int(n),
            "superclasses": int(len(classes)),
            "same_root_id_universe_bio_network_target": True,
        },
        "main_split": {
            "design": "single stratified random split",
            "fractions": MAIN_FRACTIONS,
            "totals": main_totals,
            "all_classes_present_in_all_splits": True,
            "side_is_allowed_as_bio_feature": True,
            "path": MAIN_SPLIT_PATH.relative_to(
                ROOT
            ).as_posix(),
        },
        "hemisphere_robustness": {
            "source_train_val_fractions": HEMISPHERE_SOURCE_FRACTIONS,
            "center_and_missing_side_are_excluded": True,
            "side_feature_must_be_removed": True,
            "left_to_right": {
                "source": "left",
                "target": "right",
                "totals": hemi_totals(
                    left_to_right
                ),
                "path": LEFT_TO_RIGHT_PATH.relative_to(
                    ROOT
                ).as_posix(),
            },
            "right_to_left": {
                "source": "right",
                "target": "left",
                "totals": hemi_totals(
                    right_to_left
                ),
                "path": RIGHT_TO_LEFT_PATH.relative_to(
                    ROOT
                ).as_posix(),
            },
        },
        "methodological_rules_for_ml": [
            (
                "The same frozen main split must be used for BIO, NETWORK, "
                "and BIO+NETWORK comparisons."
            ),
            (
                "All imputation, scaling, log transforms, encoding, and any "
                "other fitted preprocessing must be learned from training data only."
            ),
            (
                "Validation data are used for model/hyperparameter selection; "
                "the test set is reserved for the final evaluation."
            ),
            (
                "For hemisphere robustness, side is excluded from the feature set "
                "because it defines the domain split itself."
            ),
            (
                "Center and missing-side neurons remain in the main experiment but are "
                "excluded from hemisphere-transfer robustness."
            ),
        ],
        "runtime_seconds": float(
            time.time() - start
        ),
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

    lines = [
        "FlyWire FAFB v783 — splits experimentais",
        "",
        f"Seed: {SEED}",
        f"Universo: {n} neurônios",
        f"Superclasses: {len(classes)}",
        "",
        "SPLIT PRINCIPAL",
        "Estratificado por super_class, 70/15/15.",
        f"Train: {main_totals['train']} ({100 * main_totals['train'] / n:.4f}%)",
        f"Val:   {main_totals['val']} ({100 * main_totals['val'] / n:.4f}%)",
        f"Test:  {main_totals['test']} ({100 * main_totals['test'] / n:.4f}%)",
        "Todas as 10 classes aparecem nos três splits.",
        "BIO, REDE e BIO+REDE usam a mesma partição.",
        "",
        "ROBUSTEZ HEMISFÉRICA",
        (
            "left -> right: left dividido 85/15 em train/val; "
            "right inteiro como test."
        ),
        (
            "right -> left: right dividido 85/15 em train/val; "
            "left inteiro como test."
        ),
        "center e side ausente são excluídos somente desse teste de robustez.",
        "side como feature: excluída nos experimentos hemisféricos.",
        "",
        "REGRAS DE PRÉ-PROCESSAMENTO",
        "- imputação ajustada somente no treino;",
        "- scaling ajustado somente no treino;",
        "- log-transform/encoding definidos pelo pipeline e aplicados sem olhar o teste;",
        "- validação pode selecionar configuração;",
        "- test só é usado na avaliação final.",
        "",
        f"Tempo total: {time.time() - start:.2f} s",
    ]

    SUMMARY_TXT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Main:           {MAIN_SPLIT_PATH}")
    print(f"Main counts:    {MAIN_COUNTS_PATH}")
    print(f"Left -> right:  {LEFT_TO_RIGHT_PATH}")
    print(f"Right -> left:  {RIGHT_TO_LEFT_PATH}")
    print(f"Hemi counts:    {HEMISPHERE_COUNTS_PATH}")
    print(f"Resumo:         {SUMMARY_TXT_PATH}")

    section("Concluído")
    print(
        f"Tempo total: {time.time() - start:.2f} s"
    )


if __name__ == "__main__":
    main()
