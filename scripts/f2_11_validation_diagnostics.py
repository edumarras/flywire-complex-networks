from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

try:
    import plotly.express as px
except ImportError:
    px = None

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "reports" / "f2" / "models"
LOG_DIR = MODEL_ROOT / "logistic_validation"
RF_DIR = MODEL_ROOT / "random_forest_validation"
OUT_DIR = MODEL_ROOT / "validation_diagnostics"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

LOG_PER_CLASS = LOG_DIR / "validation_per_class_metrics.csv"
LOG_SELECTED = LOG_DIR / "selected_logistic_configs.csv"
RF_PER_CLASS = RF_DIR / "validation_per_class_metrics.csv"
RF_SELECTED = RF_DIR / "selected_random_forest_configs.csv"

ALL_COMPARISON_PATH = OUT_DIR / "per_class_selected_comparison.csv"
KEY_DELTAS_PATH = OUT_DIR / "per_class_key_deltas.csv"
RARITY_PATH = OUT_DIR / "per_class_rarity_summary.csv"
MACRO_CHECK_PATH = OUT_DIR / "macro_f1_consistency_check.csv"
SUMMARY_TXT_PATH = OUT_DIR / "validation_diagnostics_summary.txt"
SUMMARY_JSON_PATH = OUT_DIR / "validation_diagnostics_summary.json"

KEY_EXPERIMENTS = [
    "bio_strict",
    "bio_plus_availability",
    "network_degree_strength",
    "network_plus_centrality",
    "network_plus_leiden",
    "network_full",
    "bio_strict_plus_network_full",
    "bio_plus_availability_plus_network_full",
]

KEY_RF_FOR_REPORT = [
    "bio_strict",
    "bio_plus_availability",
    "network_full",
    "bio_strict_plus_network_full",
    "bio_plus_availability_plus_network_full",
]

EXPERIMENT_LABELS = {
    "bio_strict": "BIO-strict",
    "bio_plus_availability": "BIO+availability",
    "network_degree_strength": "REDE grau/força",
    "network_plus_centrality": "REDE + centralidades",
    "network_plus_leiden": "REDE + Leiden",
    "network_full": "REDE completa",
    "bio_strict_plus_network_full": "BIO-strict + REDE",
    "bio_plus_availability_plus_network_full": "BIO+availability + REDE",
}

RARE_SUPPORT_THRESHOLD = 100
VERY_RARE_SUPPORT_THRESHOLD = 25


def section(title):
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def require(path):
    if not path.exists():
        raise FileNotFoundError(f"Arquivo necessário não encontrado: {path}")


def safe_export(fig, stem):
    png = FIG_DIR / f"{stem}.png"
    html = FIG_DIR / f"{stem}.html"
    try:
        fig.write_image(str(png), scale=2)
        return str(png.relative_to(ROOT))
    except Exception:
        fig.write_html(str(html), include_plotlyjs="cdn")
        return str(html.relative_to(ROOT))


def normalize_per_class(df, family):
    required = {"experiment", "super_class", "precision", "recall", "f1", "support"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{family}: colunas ausentes: {sorted(missing)}")
    out = df.copy()
    out["model_family"] = family
    if "C" not in out.columns:
        out["C"] = np.nan
    if "config_id" not in out.columns:
        out["config_id"] = np.nan
    return out[[
        "model_family", "experiment", "C", "config_id", "super_class",
        "precision", "recall", "f1", "support"
    ]]


def macro_check(per_class, selected, family):
    rows = []
    selected_idx = selected.set_index("experiment")
    for experiment, group in per_class[per_class["experiment"].isin(KEY_EXPERIMENTS)].groupby("experiment"):
        if experiment not in selected_idx.index:
            continue
        m1 = float(group["f1"].mean())
        m2 = float(selected_idx.loc[experiment, "macro_f1"])
        rows.append({
            "model_family": family,
            "experiment": experiment,
            "macro_f1_from_per_class": m1,
            "macro_f1_reported": m2,
            "absolute_difference": abs(m1 - m2),
        })
    return pd.DataFrame(rows)


def get_f1(df, experiment):
    part = df[df["experiment"] == experiment][["super_class", "f1", "support"]].copy()
    if part["super_class"].duplicated().any():
        raise RuntimeError(f"{experiment}: super_class duplicada")
    return part.rename(columns={"f1": experiment, "support": f"{experiment}_support"})


def main():
    start = time.time()
    section("F2.11 — Diagnóstico por classe em validação")
    print("Escopo: conjunto de validação.")

    for path in [LOG_PER_CLASS, LOG_SELECTED, RF_PER_CLASS, RF_SELECTED]:
        require(path)

    log_pc = normalize_per_class(pd.read_csv(LOG_PER_CLASS), "logistic")
    rf_pc = normalize_per_class(pd.read_csv(RF_PER_CLASS), "random_forest")
    log_selected = pd.read_csv(LOG_SELECTED)
    rf_selected = pd.read_csv(RF_SELECTED)

    classes = sorted(rf_pc["super_class"].unique())
    if len(classes) != 10:
        raise RuntimeError(f"Esperadas 10 classes; encontradas {len(classes)}")

    section("1. Consistência do macro-F1")
    macro_df = pd.concat([
        macro_check(log_pc, log_selected, "logistic"),
        macro_check(rf_pc, rf_selected, "random_forest"),
    ], ignore_index=True)
    macro_df.to_csv(MACRO_CHECK_PATH, index=False)
    max_diff = float(macro_df["absolute_difference"].max())
    print(f"Maior diferença: {max_diff:.12f}")
    if max_diff > 1e-9:
        raise RuntimeError("Macro-F1 per-class não bate com o resultado selecionado.")

    section("2. Comparação geral por classe")
    all_comp = pd.concat([
        log_pc[log_pc["experiment"].isin(KEY_EXPERIMENTS)],
        rf_pc[rf_pc["experiment"].isin(KEY_EXPERIMENTS)],
    ], ignore_index=True)
    all_comp["experiment_label"] = all_comp["experiment"].map(EXPERIMENT_LABELS)
    all_comp.to_csv(ALL_COMPARISON_PATH, index=False)

    section("3. Deltas centrais no Random Forest")
    base = None
    for experiment in KEY_RF_FOR_REPORT:
        part = get_f1(rf_pc, experiment)
        base = part if base is None else base.merge(part, on="super_class", validate="one_to_one")

    support_cols = [c for c in base.columns if c.endswith("_support")]
    ref_support = base[support_cols[0]].astype(int)
    for col in support_cols[1:]:
        if not np.array_equal(ref_support.to_numpy(), base[col].astype(int).to_numpy()):
            raise RuntimeError("Support difere entre experimentos.")

    d = pd.DataFrame({"super_class": base["super_class"], "support": ref_support})
    for exp in KEY_RF_FOR_REPORT:
        d[EXPERIMENT_LABELS[exp]] = base[exp]

    d["gain_network_over_bio_strict"] = base["bio_strict_plus_network_full"] - base["bio_strict"]
    d["gain_network_over_bio_plus_availability"] = (
        base["bio_plus_availability_plus_network_full"] - base["bio_plus_availability"]
    )
    d["gain_bio_over_network"] = base["bio_strict_plus_network_full"] - base["network_full"]
    d["availability_gain_bio_only"] = base["bio_plus_availability"] - base["bio_strict"]
    d["availability_gain_with_network"] = (
        base["bio_plus_availability_plus_network_full"] - base["bio_strict_plus_network_full"]
    )
    d["rare_class"] = d["support"] < RARE_SUPPORT_THRESHOLD
    d["very_rare_class"] = d["support"] < VERY_RARE_SUPPORT_THRESHOLD
    d = d.sort_values(["support", "super_class"]).reset_index(drop=True)
    d.to_csv(KEY_DELTAS_PATH, index=False)

    rarity = d[[
        "super_class", "support", "BIO-strict", "REDE completa",
        "BIO-strict + REDE", "BIO+availability + REDE",
        "gain_network_over_bio_strict", "rare_class", "very_rare_class"
    ]].copy()
    rarity["support_band"] = np.select(
        [rarity["support"] < 25, rarity["support"] < 100, rarity["support"] < 1000],
        ["<25", "25–99", "100–999"], default=">=1000"
    )
    rarity.to_csv(RARITY_PATH, index=False)
    print(rarity.to_string(index=False))

    gain_col = "gain_network_over_bio_strict"
    combined_col = "BIO-strict + REDE"
    improved = int((d[gain_col] > 0).sum())
    largest = d.sort_values(gain_col, ascending=False).iloc[0]
    smallest = d.sort_values(gain_col).iloc[0]
    best = d.sort_values(combined_col, ascending=False).iloc[0]
    worst = d.sort_values(combined_col).iloc[0]

    section("4. Destaques")
    print(f"Classes com ganho de rede sobre BIO-strict: {improved}/10")
    print(f"Maior ganho: {largest['super_class']} ({largest[gain_col]:+.4f})")
    print(f"Menor ganho: {smallest['super_class']} ({smallest[gain_col]:+.4f})")
    print(f"Melhor F1 combinado: {best['super_class']} ({best[combined_col]:.4f})")
    print(f"Pior F1 combinado: {worst['super_class']} ({worst[combined_col]:.4f}, support={int(worst['support'])})")

    figure_outputs = {}
    if px is not None:
        heat = (
            rf_pc[rf_pc["experiment"].isin(KEY_RF_FOR_REPORT)]
            .pivot(index="super_class", columns="experiment", values="f1")
            .reindex(columns=KEY_RF_FOR_REPORT)
        )
        heat.columns = [EXPERIMENT_LABELS[c] for c in heat.columns]
        fig = px.imshow(
            heat, text_auto=".2f", aspect="auto", zmin=0, zmax=1,
            labels={"x": "Bloco de features", "y": "Superclasse", "color": "F1"},
            title="Random Forest — F1 por superclasse na validação"
        )
        fig.update_layout(template="plotly_white")
        figure_outputs["rf_heatmap"] = safe_export(fig, "rf_per_class_f1_heatmap")

        gain_plot = d.sort_values(gain_col)
        fig = px.bar(
            gain_plot, x=gain_col, y="super_class", orientation="h",
            hover_data=["support", "BIO-strict", "BIO-strict + REDE"],
            labels={gain_col: "Δ F1: combinado − BIO-strict", "super_class": "Superclasse"},
            title="Ganho da topologia sobre BIO-strict por superclasse"
        )
        fig.add_vline(x=0, line_dash="dash")
        fig.update_layout(template="plotly_white")
        figure_outputs["network_gain"] = safe_export(fig, "rf_gain_network_over_bio_by_class")

        fig = px.scatter(
            d, x="support", y=combined_col, text="super_class", log_x=True,
            hover_data=[gain_col, "rare_class"],
            labels={"support": "Support na validação (log)", combined_col: "F1 combinado"},
            title="F1 combinado versus tamanho da classe"
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(template="plotly_white")
        figure_outputs["support_vs_f1"] = safe_export(fig, "rf_support_vs_f1_combined")

    rare_classes = rarity.loc[rarity["rare_class"], "super_class"].tolist()
    very_rare = rarity.loc[rarity["very_rare_class"], "super_class"].tolist()

    summary = {
        "stage": "validation_per_class_diagnostics_only",
        "test_set_accessed": False,
        "macro_f1_consistency_max_abs_difference": max_diff,
        "classes_improved_bio_to_bio_plus_network": improved,
        "largest_gain_class": str(largest["super_class"]),
        "largest_gain": float(largest[gain_col]),
        "smallest_gain_class": str(smallest["super_class"]),
        "smallest_gain": float(smallest[gain_col]),
        "best_combined_class": str(best["super_class"]),
        "best_combined_f1": float(best[combined_col]),
        "worst_combined_class": str(worst["super_class"]),
        "worst_combined_f1": float(worst[combined_col]),
        "worst_combined_support": int(worst["support"]),
        "rare_classes_support_lt_100": rare_classes,
        "very_rare_classes_support_lt_25": very_rare,
        "figures": figure_outputs,
        "runtime_seconds": float(time.time() - start),
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "FlyWire FAFB v783 — F2.11 Diagnóstico por classe em validação",
        "",
        "Escopo: conjunto de validação.",
        "",
        f"Macro-F1: maior inconsistência = {max_diff:.12f}",
        f"Classes com ganho BIO-strict -> BIO-strict+REDE: {improved}/10",
        f"Maior ganho: {largest['super_class']} ({largest[gain_col]:+.6f})",
        f"Menor ganho: {smallest['super_class']} ({smallest[gain_col]:+.6f})",
        f"Melhor F1 combinado: {best['super_class']} ({best[combined_col]:.6f})",
        f"Pior F1 combinado: {worst['super_class']} ({worst[combined_col]:.6f}, support={int(worst['support'])})",
        "",
        "Classes com support <100: " + (", ".join(rare_classes) if rare_classes else "nenhuma"),
        "Classes com support <25: " + (", ".join(very_rare) if very_rare else "nenhuma"),
        "",
        f"Tempo total: {time.time() - start:.2f} s",
    ]
    SUMMARY_TXT_PATH.write_text("\n".join(lines), encoding="utf-8")

    section("5. Arquivos salvos")
    print(ALL_COMPARISON_PATH)
    print(KEY_DELTAS_PATH)
    print(RARITY_PATH)
    print(MACRO_CHECK_PATH)
    print(SUMMARY_TXT_PATH)
    section("Concluído")
    print(f"Tempo total: {time.time() - start:.2f} s")


if __name__ == "__main__":
    main()
