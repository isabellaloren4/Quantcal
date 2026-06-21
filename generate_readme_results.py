#!/usr/bin/env python3
"""
Gera os resultados suplementares (não incluídos no artigo) em Markdown,
a partir do CSV de medianas de MAE por dataset.

Saídas:
  - results_summary.md : tabela compacta (MAE médio + rank médio) p/ colar no README
  - RESULTS.md         : tabela completa por dataset (106 x 40), linkada pelo README

Uso:
  python generate_readme_results.py medianas_methods_all_exp_part4.csv
"""
import sys
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else "medianas_methods_all_exp_part4.csv"

df = pd.read_csv(CSV)

# 13 metodos do artigo (ordem do diagrama de diferenca critica)
methods = [
    "QuantCal(knn,3)", "EMQ(bcts(rf))", "DyS(rf)", "MS(rf)", "PACC(rf)",
    "EMQ(rf)", "FM(rf)", "ACC(rf)", "MLQ(rf)", "HDy(rf)",
    "KDEyML(rf)", "CC(rf)", "CC(bcts(rf))",
]
missing = [m for m in methods if m not in df.columns]
if missing:
    raise SystemExit(f"Colunas ausentes no CSV: {missing}")
df = df[["dataset"] + methods]
M = df[methods]

# --- agregados ---
median_mae = M.median()                        # mediana entre os datasets (igual ao artigo)
ranks = M.rank(axis=1, method="average")       # rank 1 = menor MAE no dataset
avg_rank = ranks.mean()

summary = (
    pd.DataFrame({"Median MAE": median_mae, "Avg. rank": avg_rank})
    .sort_values("Avg. rank")
    .reset_index()
    .rename(columns={"index": "Method"})
)

# --- 1) resumo compacto p/ README ---
lines = []
lines.append("### Summary across all datasets\n")
lines.append("Methods ranked over **{} binary datasets**. *Avg. rank* is the "
             "mean across datasets of the per-dataset rank of each method's "
             "median MAE (rank 1 = best). *Median MAE* is the median of those "
             "per-dataset values across datasets. Lower is better in both.\n"
             .format(len(df)))
lines.append("| Method | Median MAE | Avg. rank |")
lines.append("|:---|---:|---:|")
for _, r in summary.iterrows():
    lines.append(f"| {r['Method']} | {r['Median MAE']:.4f} | {r['Avg. rank']:.2f} |")
lines.append("")
lines.append("> Full per-dataset results: see [`RESULTS.md`](RESULTS.md).")
with open("results_summary.md", "w") as f:
    f.write("\n".join(lines))

# --- 2) tabela completa por dataset ---
full = df.copy()
for c in methods:
    full[c] = full[c].map(lambda x: f"{x:.4f}")
md_full = ["# Per-dataset results (median MAE)\n",
           f"{len(df)} binary datasets x {len(methods)} methods. "
           "Median MAE over the experiment runs; lower is better.\n",
           full.to_markdown(index=False)]
with open("RESULTS.md", "w") as f:
    f.write("\n".join(md_full))

print("OK -> results_summary.md, RESULTS.md")
print(summary.round(4).to_string(index=False))


# --- 3) tabela por dataset com vencedor (so os 13 metodos) ---
pd_tbl = df.copy()
pd_tbl["best method"] = M.idxmin(axis=1)
pd_tbl["best MAE"] = M.min(axis=1)
for c in methods + ["best MAE"]:
    pd_tbl[c] = pd_tbl[c].map(lambda x: f"{x:.4f}")
with open("RESULTS_per_dataset.md", "w") as f:
    f.write("# Per-dataset results (median MAE) \u2014 13 methods\n\n")
    f.write(f"One row per dataset ({len(df)} total). `best method` is the "
            "lowest-MAE method on that dataset; lower is better.\n\n")
    f.write(pd_tbl.to_markdown(index=False))
print("OK -> RESULTS_per_dataset.md")
