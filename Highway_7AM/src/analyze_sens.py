"""Análise da sensibilidade: B e C com paciência 15 (máx. 200 épocas) contra o protocolo original (paciência 5).

Lê results/runs_test.csv (A, B', B), results/runs_test_C.csv e results/runs_sens_BC.csv.
Usa só as seeds presentes nos dois modelos da sensibilidade. Comparações pareadas por seed
(Shapiro nas diferenças -> t pareado ou Wilcoxon), bilaterais, descritivas (n pequeno, fora do plano).
Saída: results/analise_sens/*.csv e figura.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

R = Path("results"); OUT = R / "analise_sens"; OUT.mkdir(exist_ok=True)
base = pd.concat([pd.read_csv(R / "runs_test.csv"), pd.read_csv(R / "runs_test_C.csv")])
sens = pd.read_csv(R / "runs_sens_BC.csv")
d = pd.concat([base, sens])
seeds_ok = [s for s in sorted(sens.seed.unique())
            if {"B_p15", "C_p15"} <= set(sens[sens.seed == s].model) and
            all(len(sens[(sens.seed == s) & (sens.model == m)]) == 8 for m in ("B_p15", "C_p15"))]
d = d[d.seed.isin(seeds_ok)]
per = d.groupby(["model", "seed"]).agg(auc=("auc", "mean"), f1=("f1", "mean"), epocas=("epochs", "first")).reset_index()
wide = {m: per[per.model == m].set_index("seed") for m in ["B", "C", "B_p15", "C_p15"]}
print(f"Seeds completas nos dois modelos da sensibilidade: {seeds_ok} (n = {len(seeds_ok)})\n")

res = per.groupby("model").agg(auc=("auc", "mean"), auc_dp=("auc", "std"), f1=("f1", "mean"), epocas_med=("epocas", "median"), epocas_max=("epocas", "max")).loc[["B", "B_p15", "C", "C_p15"]]
print("=== Média por modelo nas mesmas seeds (AUC e F1 = média dos 8 ataques) ===")
print(res.round(4).to_string()); res.round(5).to_csv(OUT / "resumo_sens.csv")


def paired(a, b, metric):
    x = (wide[a][metric] - wide[b][metric]).dropna().values
    sh = stats.shapiro(x).pvalue if len(x) >= 3 else np.nan
    if sh > 0.05:
        p = stats.ttest_1samp(x, 0).pvalue; t = "t pareado"
    else:
        p = stats.wilcoxon(x).pvalue; t = "Wilcoxon"
    se = x.std(ddof=1) / np.sqrt(len(x)); tc = stats.t.ppf(0.975, len(x) - 1)
    return dict(comparacao=f"{a} - {b}", metrica=metric, n=len(x), media_d=x.mean(), ic95_inf=x.mean() - tc * se, ic95_sup=x.mean() + tc * se,
                dz=x.mean() / x.std(ddof=1) if x.std(ddof=1) > 0 else np.nan, teste=t, p=p)


rows = []
for metric in ("auc", "f1"):
    for a, b in [("B_p15", "B"), ("C_p15", "C"), ("B", "C"), ("B_p15", "C_p15")]:
        rows.append(paired(a, b, metric))
comp = pd.DataFrame(rows)
print("\n=== Comparações pareadas (bilaterais; n pequeno, descritivas) ===")
print(comp.round(4).to_string(index=False)); comp.round(6).to_csv(OUT / "comparacoes_sens.csv", index=False)

# efeito do treino extra por seed (épocas ganhas x ganho de AUC)
for m in ("B", "C"):
    j = wide[f"{m}_p15"].join(wide[m], lsuffix="_p15", rsuffix="_orig")
    j["ganho_auc"] = j.auc_p15 - j.auc_orig; j["epocas_a_mais"] = j.epocas_p15 - j.epocas_orig
    print(f"\n=== {m}: por seed (paciência 15 menos paciência 5) ===")
    print(j[["epocas_orig", "epocas_p15", "epocas_a_mais", "auc_orig", "auc_p15", "ganho_auc"]].round(4).to_string())
    j.round(5).to_csv(OUT / f"por_seed_{m}.csv")

# AUC por ataque, B_p15 - C_p15
pa = d[d.model.isin(["B_p15", "C_p15", "B", "C"])].groupby(["model", "attack"]).auc.mean().unstack(0)[["B", "B_p15", "C", "C_p15"]]
print("\n=== AUC média por ataque ===\n" + pa.round(3).to_string()); pa.round(4).to_csv(OUT / "auc_por_ataque_sens.csv")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for k, m in enumerate(("B", "C")):
    for s in seeds_ok:
        ax[k].plot([0, 1], [wide[m].loc[s, "auc"], wide[f"{m}_p15"].loc[s, "auc"]], "-o", color="#7f8c9a", lw=1, ms=3)
    ax[k].set_xticks([0, 1]); ax[k].set_xticklabels(["paciência 5", "paciência 15"]); ax[k].set_title(f"Modelo {m}: AUC média por seed"); ax[k].set_ylabel("AUC-ROC (8 ataques)")
fig.tight_layout(); fig.savefig(OUT / "fig_sensibilidade.png", dpi=160)
print(f"\nSalvo em {OUT}/")
