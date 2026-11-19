from pathlib import Path
import json
import time

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
F2_DIR = ROOT / "reports" / "f2"
FEATURE_DIR = F2_DIR / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

TOPO_PATH = FEATURE_DIR / "node_topological_features.csv.gz"

BIO_PATH = FEATURE_DIR / "node_biological_features.csv.gz"
TARGET_PATH = FEATURE_DIR / "node_targets.csv.gz"
MANIFEST_PATH = FEATURE_DIR / "biological_feature_manifest.csv"
SETS_PATH = FEATURE_DIR / "biological_feature_sets.json"
AUDIT_PATH = FEATURE_DIR / "biological_features_audit.csv"
MISSING_BY_CLASS_PATH = FEATURE_DIR / "biological_missingness_by_superclass.csv"
NT_TOPO_AUDIT_PATH = FEATURE_DIR / "nt_availability_topology_audit.csv"
FORBIDDEN_PATH = FEATURE_DIR / "biological_forbidden_columns.csv"
LEAKAGE_PATH = FEATURE_DIR / "biological_leakage_proxy_audit.csv"
SUMMARY_TXT_PATH = FEATURE_DIR / "biological_features_summary.txt"
SUMMARY_JSON_PATH = FEATURE_DIR / "biological_features_summary.json"

MORPH_FEATURES = ["length_nm", "area_nm", "size_nm"]
NT_PROB_FEATURES = [
    "da_avg",
    "ser_avg",
    "gaba_avg",
    "glut_avg",
    "ach_avg",
    "oct_avg",
]
SIDE_FEATURES = ["side"]
AVAILABILITY_FEATURES = ["nt_available"]

BIO_STRICT = SIDE_FEATURES + MORPH_FEATURES + NT_PROB_FEATURES
BIO_PLUS_AVAILABILITY = BIO_STRICT + AVAILABILITY_FEATURES

FEATURE_SETS = {
    "bio_morph_side": SIDE_FEATURES + MORPH_FEATURES,
    "bio_nt_probabilities": NT_PROB_FEATURES,
    "bio_strict": BIO_STRICT,
    "bio_plus_availability": BIO_PLUS_AVAILABILITY,
    "nt_availability_only": AVAILABILITY_FEATURES,
}

# Explicitly forbidden from the predictive BIO block.
FORBIDDEN_REASONS = {
    "flow": "Taxonomic/functional grouping directly upstream of super_class.",
    "class": "Taxonomic label closely related to super_class.",
    "sub_class": "Taxonomic label closely related to super_class.",
    "hemilineage": "Biological annotation close to taxonomy; outside the declared BIO baseline.",
    "nerve": "Annotation strongly tied to specific superclasses; leakage risk.",
    "group": "Coarse neuron grouping with very high association to super_class; outside declared BIO baseline.",
    "primary_type": "Fine cell-type taxonomy; direct leakage risk.",
    "additional_type(s)": "Fine cell-type taxonomy; direct leakage risk.",
    "nt_type": "Final categorical neurotransmitter call excluded in favor of the six probabilistic scores.",
    "nt_type_score": "Deterministically reveals availability of the final nt_type call; observability proxy.",
}

FEATURE_META = {
    "side": {
        "group": "lateralality",
        "source": "classification",
        "kind": "categorical",
        "description": "Neuron lateralality (left/right/center).",
        "missing_policy": "encode missing as an explicit unknown category inside the ML pipeline",
        "recommended_preprocessing": "constant_unknown_then_one_hot",
    },
    "length_nm": {
        "group": "morphology",
        "source": "cell_stats",
        "kind": "numeric",
        "description": "Skeleton/path length in nanometers.",
        "missing_policy": "impute using training data only; no missingness indicator in BIO-strict",
        "recommended_preprocessing": "train_impute_then_log1p_then_scale",
    },
    "area_nm": {
        "group": "morphology",
        "source": "cell_stats",
        "kind": "numeric",
        "description": "Cell surface-area proxy from FlyWire cell statistics.",
        "missing_policy": "impute using training data only; no missingness indicator in BIO-strict",
        "recommended_preprocessing": "train_impute_then_log1p_then_scale",
    },
    "size_nm": {
        "group": "morphology",
        "source": "cell_stats",
        "kind": "numeric",
        "description": "Cell size/volume proxy from FlyWire cell statistics.",
        "missing_policy": "impute using training data only; no missingness indicator in BIO-strict",
        "recommended_preprocessing": "train_impute_then_log1p_then_scale",
    },
    "nt_available": {
        "group": "neurotransmitter_availability",
        "source": "neurons",
        "kind": "binary",
        "description": "1 when the six neurotransmitter probability scores contain information (sum > 0), else 0.",
        "missing_policy": "none",
        "recommended_preprocessing": "passthrough",
    },
}

for col in NT_PROB_FEATURES:
    FEATURE_META[col] = {
        "group": "neurotransmitter_probabilities",
        "source": "neurons",
        "kind": "numeric",
        "description": f"FlyWire neurotransmitter probability/score: {col}.",
        "missing_policy": (
            "all-zero source vectors are converted to NaN as a six-dimensional block; "
            "later impute by resampling complete valid NT vectors from the TRAINING fold only"
        ),
        "recommended_preprocessing": "training_fold_joint_vector_imputation_then_scale_optional",
    }


def section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def fmt_int(x):
    return f"{int(x):,}".replace(",", ".")


def pick_raw(stem):
    for suffix in [".csv.gz", ".csv"]:
        path = RAW_DIR / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Não encontrado em {RAW_DIR}: {stem}.csv(.gz)")


def weighted_purity(df, feature, target="super_class"):
    tmp = df[[feature, target]].dropna()
    if len(tmp) == 0:
        return np.nan, 0
    counts = pd.crosstab(tmp[feature], tmp[target])
    purity = float(counts.max(axis=1).sum() / counts.to_numpy().sum())
    return purity, len(tmp)


def feature_audit(df):
    rows = []
    for col in BIO_PLUS_AVAILABILITY:
        s = df[col]
        row = {
            "feature": col,
            "group": FEATURE_META[col]["group"],
            "kind": FEATURE_META[col]["kind"],
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_fraction": float(s.isna().mean()),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if FEATURE_META[col]["kind"] in {"numeric", "binary"}:
            x = pd.to_numeric(s, errors="coerce")
            row.update({
                "min": float(x.min()) if x.notna().any() else np.nan,
                "mean": float(x.mean()) if x.notna().any() else np.nan,
                "median": float(x.median()) if x.notna().any() else np.nan,
                "p95": float(x.quantile(0.95)) if x.notna().any() else np.nan,
                "max": float(x.max()) if x.notna().any() else np.nan,
            })
        else:
            row.update({"min": np.nan, "mean": np.nan, "median": np.nan, "p95": np.nan, "max": np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def build_missingness_by_superclass(master):
    rows = []
    for sc, g in master.groupby("super_class", sort=True, observed=True):
        n = len(g)
        rows.append({
            "super_class": sc,
            "n": n,
            "nt_type_missing_n": int(g["nt_type"].isna().sum()),
            "nt_type_missing_fraction": float(g["nt_type"].isna().mean()),
            "nt_probability_unavailable_n": int((g["nt_available"] == 0).sum()),
            "nt_probability_unavailable_fraction": float((g["nt_available"] == 0).mean()),
            "morphology_missing_n": int(g[MORPH_FEATURES].isna().any(axis=1).sum()),
            "morphology_missing_fraction": float(g[MORPH_FEATURES].isna().any(axis=1).mean()),
            "side_missing_n": int(g["side"].isna().sum()),
            "side_missing_fraction": float(g["side"].isna().mean()),
        })
    return pd.DataFrame(rows)


def build_nt_topology_audit(master):
    rows = []
    for sc, g in master.groupby("super_class", sort=True, observed=True):
        row = {"super_class": sc, "n": len(g)}
        for label, mask in [
            ("prob_unavailable", g["nt_available"] == 0),
            ("prob_available", g["nt_available"] == 1),
            ("type_missing", g["nt_type"].isna()),
            ("type_present", g["nt_type"].notna()),
        ]:
            sub = g.loc[mask, "out_degree"]
            row[f"{label}_n"] = int(len(sub))
            row[f"{label}_out_degree_median"] = float(sub.median()) if len(sub) else np.nan
            row[f"{label}_out_degree_mean"] = float(sub.mean()) if len(sub) else np.nan
            row[f"{label}_out_degree_zero_fraction"] = float((sub == 0).mean()) if len(sub) else np.nan
        rows.append(row)

    all_row = {"super_class": "__ALL__", "n": len(master)}
    for label, mask in [
        ("prob_unavailable", master["nt_available"] == 0),
        ("prob_available", master["nt_available"] == 1),
        ("type_missing", master["nt_type"].isna()),
        ("type_present", master["nt_type"].notna()),
    ]:
        sub = master.loc[mask, "out_degree"]
        all_row[f"{label}_n"] = int(len(sub))
        all_row[f"{label}_out_degree_median"] = float(sub.median()) if len(sub) else np.nan
        all_row[f"{label}_out_degree_mean"] = float(sub.mean()) if len(sub) else np.nan
        all_row[f"{label}_out_degree_zero_fraction"] = float((sub == 0).mean()) if len(sub) else np.nan
    rows.append(all_row)
    return pd.DataFrame(rows)


def main():
    start = time.time()

    section("F2.07 — Preparação das features biológicas")

    classification_path = pick_raw("classification")
    neurons_path = pick_raw("neurons")
    cell_stats_path = pick_raw("cell_stats")
    types_path = None
    for candidate in [RAW_DIR / "consolidated_cell_types.csv.gz", RAW_DIR / "consolidated_cell_types.csv"]:
        if candidate.exists():
            types_path = candidate
            break

    if not TOPO_PATH.exists():
        raise FileNotFoundError(
            f"Não encontrado: {TOPO_PATH}\n"
            "Dependência gerada por scripts/f2_06_build_topological_features.py."
        )

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    section("1. Carregando fontes")

    topo = pd.read_csv(TOPO_PATH, usecols=["root_id", "out_degree"], dtype={"root_id": "int64"})
    cls = pd.read_csv(classification_path, low_memory=False)
    neurons = pd.read_csv(neurons_path, low_memory=False)
    morph = pd.read_csv(cell_stats_path, low_memory=False)
    types = pd.read_csv(types_path, low_memory=False) if types_path is not None else None

    for name, df in [("topological", topo), ("classification", cls), ("neurons", neurons), ("cell_stats", morph)]:
        if "root_id" not in df.columns:
            raise RuntimeError(f"{name} não possui root_id.")
        if df["root_id"].duplicated().any():
            raise RuntimeError(f"{name} possui root_id duplicado.")

    required_cls = {"root_id", "super_class", "side"}
    required_neurons = {"root_id", "nt_type", "nt_type_score", "group", *NT_PROB_FEATURES}
    required_morph = {"root_id", *MORPH_FEATURES}

    if not required_cls.issubset(cls.columns):
        raise RuntimeError(f"classification sem colunas: {sorted(required_cls - set(cls.columns))}")
    if not required_neurons.issubset(neurons.columns):
        raise RuntimeError(f"neurons sem colunas: {sorted(required_neurons - set(neurons.columns))}")
    if not required_morph.issubset(morph.columns):
        raise RuntimeError(f"cell_stats sem colunas: {sorted(required_morph - set(morph.columns))}")

    print(f"Universo topológico: {fmt_int(len(topo))}")
    print(f"classification:      {fmt_int(len(cls))}")
    print(f"neurons:             {fmt_int(len(neurons))}")
    print(f"cell_stats:          {fmt_int(len(morph))}")

    # ------------------------------------------------------------------
    # 2. Master join on the frozen network universe
    # ------------------------------------------------------------------
    section("2. Construindo universo BIO")

    master = topo.merge(
        cls,
        on="root_id",
        how="left",
        validate="one_to_one",
    ).merge(
        neurons,
        on="root_id",
        how="left",
        validate="one_to_one",
    ).merge(
        morph,
        on="root_id",
        how="left",
        validate="one_to_one",
    )

    if len(master) != len(topo):
        raise RuntimeError("Merge BIO alterou o número de linhas do universo topológico.")
    if master["super_class"].isna().any():
        raise RuntimeError("Há neurônios sem super_class no universo de análise.")

    # ------------------------------------------------------------------
    # 3. Neurotransmitter availability audit
    # ------------------------------------------------------------------
    section("3. Auditando disponibilidade de neurotransmissor")

    raw_prob_sum = master[NT_PROB_FEATURES].sum(axis=1, min_count=1)
    if raw_prob_sum.isna().any():
        raise RuntimeError("Há NaN nas seis colunas de probabilidade NT na fonte neurons.csv.")

    # In this FlyWire export the regimes are clean: exactly zero or approx 1.
    master["nt_available"] = (raw_prob_sum > 0).astype(np.int8)
    master["nt_type_available"] = master["nt_type"].notna().astype(np.int8)

    n_prob_unavailable = int((master["nt_available"] == 0).sum())
    n_type_missing = int((master["nt_type_available"] == 0).sum())
    n_type_missing_prob_valid = int(
        ((master["nt_type_available"] == 0) & (master["nt_available"] == 1)).sum()
    )

    valid_sums = raw_prob_sum[master["nt_available"] == 1]
    if len(valid_sums) and (valid_sums.min() < 0.90 or valid_sums.max() > 1.10):
        raise RuntimeError(
            "Probabilidades NT disponíveis não somam aproximadamente 1; revise a definição de nt_available."
        )

    # Crucial: all-zero vectors are NOT treated as biological zero probability.
    # Convert the whole six-dimensional vector to NaN so the later ML pipeline
    # must impute it from TRAINING data only.
    unavailable_mask = master["nt_available"] == 0
    master.loc[unavailable_mask, NT_PROB_FEATURES] = np.nan

    print(f"nt_type ausente:                  {fmt_int(n_type_missing)}")
    print(f"Vetor probabilístico indisponível:{fmt_int(n_prob_unavailable)}")
    print(f"Sem nt_type, mas vetor válido:    {fmt_int(n_type_missing_prob_valid)}")
    print(f"Soma NT válida — min/max:         {valid_sums.min():.3f} / {valid_sums.max():.3f}")

    # nt_type_score audit: perfect proxy for FINAL CALL availability.
    call_missing = master["nt_type_available"] == 0
    score_zero_when_call_missing = bool((master.loc[call_missing, "nt_type_score"] == 0).all())
    score_positive_when_call_present = bool((master.loc[~call_missing, "nt_type_score"] > 0).all())

    # ------------------------------------------------------------------
    # 4. Freeze feature/target tables
    # ------------------------------------------------------------------
    section("4. Congelando tabelas")

    bio = master[["root_id"] + BIO_PLUS_AVAILABILITY].copy()
    target = master[["root_id", "super_class"]].copy()

    if bio["root_id"].duplicated().any() or target["root_id"].duplicated().any():
        raise RuntimeError("root_id duplicado nas tabelas de features/target.")

    # Side stays raw; missing values are intentionally preserved for pipeline handling.
    # Morphology missing values are also preserved. NT unavailable vectors are intentional NaN blocks.
    expected_nt_missing = int(bio[NT_PROB_FEATURES[0]].isna().sum())
    for col in NT_PROB_FEATURES:
        if int(bio[col].isna().sum()) != expected_nt_missing:
            raise RuntimeError("As seis probabilidades NT não possuem a mesma máscara de missingness.")

    if expected_nt_missing != n_prob_unavailable:
        raise RuntimeError("Missingness NT na tabela BIO não coincide com nt_available=0.")

    # Forbidden columns must not appear in the frozen feature table.
    forbidden_present = sorted(set(FORBIDDEN_REASONS) & set(bio.columns))
    if forbidden_present:
        raise RuntimeError("Colunas proibidas entraram na tabela BIO: " + ", ".join(forbidden_present))

    bio.to_csv(BIO_PATH, index=False, compression="gzip")
    target.to_csv(TARGET_PATH, index=False, compression="gzip")

    print(f"BIO:    {fmt_int(len(bio))} linhas, {len(bio.columns)-1} features máximas")
    print(f"Target: {fmt_int(len(target))} linhas, {target['super_class'].nunique()} classes")

    # ------------------------------------------------------------------
    # 5. Audits
    # ------------------------------------------------------------------
    section("5. Auditorias")

    audit = feature_audit(bio)
    audit.to_csv(AUDIT_PATH, index=False)

    missing_by_class = build_missingness_by_superclass(master)
    missing_by_class.to_csv(MISSING_BY_CLASS_PATH, index=False)

    nt_topology = build_nt_topology_audit(master)
    nt_topology.to_csv(NT_TOPO_AUDIT_PATH, index=False)

    forbidden_rows = []
    source_map = {
        "flow": "classification",
        "class": "classification",
        "sub_class": "classification",
        "hemilineage": "classification",
        "nerve": "classification",
        "group": "neurons",
        "nt_type": "neurons",
        "nt_type_score": "neurons",
        "primary_type": "consolidated_cell_types",
        "additional_type(s)": "consolidated_cell_types",
    }

    for col, reason in FORBIDDEN_REASONS.items():
        forbidden_rows.append({
            "column": col,
            "source": source_map[col],
            "reason": reason,
            "present_in_frozen_bio_table": col in bio.columns,
        })

    pd.DataFrame(forbidden_rows).to_csv(FORBIDDEN_PATH, index=False)

    # Quantify association of forbidden categorical proxies with the target.
    leakage_frames = []
    leakage_source = master.copy()

    if types is not None:
        leakage_source = leakage_source.merge(
            types,
            on="root_id",
            how="left",
            validate="one_to_one",
        )

    categorical_forbidden = [
        "flow", "class", "sub_class", "hemilineage", "nerve", "group", "nt_type"
    ]
    if types is not None:
        categorical_forbidden += ["primary_type", "additional_type(s)"]

    for col in categorical_forbidden:
        if col not in leakage_source.columns:
            continue
        purity, labeled_n = weighted_purity(leakage_source, col)
        leakage_frames.append({
            "column": col,
            "source": source_map[col],
            "n_unique": int(leakage_source[col].nunique(dropna=True)),
            "coverage_n": int(leakage_source[col].notna().sum()),
            "coverage_fraction": float(leakage_source[col].notna().mean()),
            "weighted_purity_to_super_class": purity,
            "reason_excluded": FORBIDDEN_REASONS[col],
        })

    pd.DataFrame(leakage_frames).sort_values(
        "weighted_purity_to_super_class", ascending=False
    ).to_csv(LEAKAGE_PATH, index=False)

    # ------------------------------------------------------------------
    # 6. Manifest and feature sets
    # ------------------------------------------------------------------
    section("6. Manifesto e feature sets")

    set_membership = {}
    for set_name, cols in FEATURE_SETS.items():
        for col in cols:
            set_membership.setdefault(col, []).append(set_name)

    manifest_rows = []
    for feature in BIO_PLUS_AVAILABILITY:
        meta = FEATURE_META[feature]
        manifest_rows.append({
            "feature": feature,
            "group": meta["group"],
            "source": meta["source"],
            "kind": meta["kind"],
            "description": meta["description"],
            "missing_policy": meta["missing_policy"],
            "recommended_preprocessing": meta["recommended_preprocessing"],
            "feature_sets": ";".join(set_membership.get(feature, [])),
        })

    pd.DataFrame(manifest_rows).to_csv(MANIFEST_PATH, index=False)

    with SETS_PATH.open("w", encoding="utf-8") as f:
        json.dump(FEATURE_SETS, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    section("7. Resumo")

    overall_nt = nt_topology.loc[nt_topology["super_class"] == "__ALL__"].iloc[0]
    group_purity = np.nan
    leakage_df = pd.DataFrame(leakage_frames)
    if "group" in leakage_df["column"].values:
        group_purity = float(
            leakage_df.loc[leakage_df["column"] == "group", "weighted_purity_to_super_class"].iloc[0]
        )

    summary = {
        "nodes": int(len(bio)),
        "target_classes": int(target["super_class"].nunique()),
        "feature_sets": FEATURE_SETS,
        "source_missingness": {
            "side_missing": int(master["side"].isna().sum()),
            "morphology_rows_missing_any": int(master[MORPH_FEATURES].isna().any(axis=1).sum()),
            "nt_type_missing": n_type_missing,
            "nt_probability_vector_unavailable": n_prob_unavailable,
            "nt_type_missing_but_probability_vector_valid": n_type_missing_prob_valid,
        },
        "neurotransmitter_definition": {
            "nt_available": "1 iff sum(da_avg, ser_avg, gaba_avg, glut_avg, ach_avg, oct_avg) > 0",
            "valid_probability_sum_min": float(valid_sums.min()),
            "valid_probability_sum_max": float(valid_sums.max()),
            "all_zero_vectors_converted_to_nan_block": True,
            "nt_type_score_zero_for_every_missing_nt_type": score_zero_when_call_missing,
            "nt_type_score_positive_for_every_present_nt_type": score_positive_when_call_present,
        },
        "observability_topology_audit": {
            "prob_unavailable_out_degree_median": float(overall_nt["prob_unavailable_out_degree_median"]),
            "prob_available_out_degree_median": float(overall_nt["prob_available_out_degree_median"]),
            "prob_unavailable_out_degree_zero_fraction": float(overall_nt["prob_unavailable_out_degree_zero_fraction"]),
            "prob_available_out_degree_zero_fraction": float(overall_nt["prob_available_out_degree_zero_fraction"]),
        },
        "forbidden_proxy_example": {
            "group_weighted_purity_to_super_class": group_purity,
        },
        "important_methodological_decisions": [
            "super_class is stored in a separate target file and never inside the biological feature table.",
            "The frozen BIO table is whitelist-based, not blacklist-selected at model time.",
            "flow/class/sub_class/hemilineage/nerve/group/cell-type annotations are excluded to prevent taxonomy leakage.",
            "nt_type_score is excluded because it explicitly encodes availability of the final nt_type call.",
            "The six all-zero NT vectors are treated as missing information, not as six biological zero probabilities.",
            "BIO-strict excludes nt_available, but this only removes the explicit NT-availability indicator; morphology/topology may still carry observability signal.",
            "NT vector imputation is performed inside the ML pipeline using training folds only.",
            "All compared models use the same root_ids and the same splits.",
        ],
        "runtime_seconds": float(time.time() - start),
        "outputs": {
            "biological_features": BIO_PATH.relative_to(ROOT).as_posix(),
            "targets": TARGET_PATH.relative_to(ROOT).as_posix(),
            "feature_sets": SETS_PATH.relative_to(ROOT).as_posix(),
        },
    }

    with SUMMARY_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = [
        "FlyWire FAFB v783 — preparação das features biológicas",
        "",
        f"Nós: {len(bio)}",
        f"Classes-alvo: {target['super_class'].nunique()}",
        "",
        "FEATURE SETS",
        "bio_strict:",
        *[f"  - {x}" for x in BIO_STRICT],
        "",
        "bio_plus_availability:",
        *[f"  - {x}" for x in BIO_PLUS_AVAILABILITY],
        "",
        "nt_availability_only:",
        "  - nt_available",
        "",
        "MISSINGNESS / DISPONIBILIDADE",
        f"side ausente: {master['side'].isna().sum()}",
        f"morfologia ausente (qualquer coluna): {master[MORPH_FEATURES].isna().any(axis=1).sum()}",
        f"nt_type ausente: {n_type_missing}",
        f"vetor probabilístico NT indisponível (soma=0): {n_prob_unavailable}",
        f"sem nt_type, mas com vetor probabilístico válido: {n_type_missing_prob_valid}",
        f"soma dos vetores NT válidos: {valid_sums.min():.3f} a {valid_sums.max():.3f}",
        "",
        "OBSERVABILIDADE × TOPOLOGIA (TODOS OS NÓS)",
        f"out_degree mediano, NT prob indisponível: {overall_nt['prob_unavailable_out_degree_median']:.3f}",
        f"out_degree mediano, NT prob disponível: {overall_nt['prob_available_out_degree_median']:.3f}",
        f"fração out_degree=0, NT prob indisponível: {overall_nt['prob_unavailable_out_degree_zero_fraction']:.6f}",
        f"fração out_degree=0, NT prob disponível: {overall_nt['prob_available_out_degree_zero_fraction']:.6f}",
        "",
        "ANTI-LEAKAGE",
        "- super_class fica em node_targets.csv.gz, separado das features.",
        "- seleção por whitelist explícita.",
        "- flow, class, sub_class, hemilineage, nerve, group e cell types não entram.",
        "- nt_type não entra; usamos as seis probabilidades.",
        "- nt_type_score não entra; ele revela disponibilidade da chamada categórica.",
        f"- pureza ponderada de group em relação a super_class: {group_purity:.6f}",
        "",
        "TRATAMENTO DO NT",
        "- vetor [0,0,0,0,0,0] é convertido para NaN nas seis colunas.",
        "- os 9.752 casos sem nt_type mas com vetor válido permanecem com suas probabilidades.",
        "- nenhuma imputação é feita aqui.",
        "- imputação do bloco NT: ajustada somente no conjunto de treino da dobra/split.",
        "- BIO-strict não inclui nt_available.",
        "- BIO+availability inclui nt_available explicitamente.",
        "",
        "LIMITAÇÃO METODOLÓGICA",
        "Remover nt_available elimina o indicador explícito de disponibilidade, mas não garante",
        "remoção completa do artefato de observabilidade, pois morfologia e topologia também podem",
        "estar associadas ao grau de reconstrução/proofreading.",
        "",
        f"Tempo total: {time.time() - start:.2f} s",
    ]

    SUMMARY_TXT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines[-23:]))
    print()
    print(f"BIO:       {BIO_PATH}")
    print(f"Targets:   {TARGET_PATH}")
    print(f"Manifesto: {MANIFEST_PATH}")
    print(f"Sets:      {SETS_PATH}")
    print(f"Resumo:    {SUMMARY_TXT_PATH}")

    section("Concluído")
    print(f"Tempo total: {time.time() - start:.2f} s")


if __name__ == "__main__":
    main()
