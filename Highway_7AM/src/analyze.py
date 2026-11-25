"""Análise estatística do experimento (results/runs_test.csv).

PLANO DE ANÁLISE (declarado antes de analisar a tabela completa)
  Unidade: a seed (30 por modelo); os três modelos usam as MESMAS seeds, então as comparações
  são PAREADAS: d = métrica do primeiro modelo - métrica do segundo, na mesma seed.
  Comparações:  B - A   (principal)          H1 direcional: B é melhor
                B - B'  (QP4, secundária)    H1 direcional: B é melhor
                B' - A  (descritiva)         bilateral
  Métricas:     QP1  erro de reconstrução (MSE) nos nós normais do baseline/Test   (menor é melhor)
                QP2  AUC-ROC e F1, médias dos 8 ataques por seed, e cada ataque em separado
                     (maior é melhor)
  Teste:        Shapiro-Wilk nas 30 diferenças -> se não rejeita a normalidade (p > 0,05),
                t de Student pareado; senão, Wilcoxon de postos sinalizados.
  Múltiplos testes: correção de Holm dentro de cada família (as 8 comparações por ataque, para
                cada comparação e métrica). Também é feito o teste de Friedman nos 3 modelos.
  Tamanho do efeito: dz (média de d / desvio de d) e intervalo de confiança de 95% da média de d.
  Latência e memória (QP3) NÃO estão aqui: são medidas em outro script.

Uso (de dentro da pasta do cenário, com o ambiente .venv ativo):
    python src/analyze.py
Saídas em results/analise/ (tabelas .csv e figuras .png).
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = Path("results/runs_test.csv")
OUT = Path("results/analise")
ALPHA = 0.05
MODELS = ["A", "Bp", "B"]
LABEL = {"A": "A", "Bp": "B'", "B": "B"}
# (primeiro, segundo, direção da H1 para métricas "maior é melhor")
COMPARISONS = [("B", "A", "greater"), ("B", "Bp", "greater"), ("Bp", "A", "two-sided")]


def holm(p):
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def paired(x, y, alternative):
    """Teste pareado de x contra y. alternative: 'greater', 'less' ou 'two-sided'."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x - y
    n = len(d)
    sw_p = stats.shapiro(d).pvalue if np.ptp(d) > 0 else float("nan")
    normal = bool(sw_p > ALPHA) if not np.isnan(sw_p) else False
    p_t = stats.ttest_rel(x, y, alternative=alternative).pvalue
    try:
        p_w = stats.wilcoxon(x, y, alternative=alternative).pvalue
    except ValueError:
        p_w = float("nan")
    sd = d.std(ddof=1)
    ci = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=stats.sem(d)) if sd > 0 else (np.nan, np.nan)
    return {
        "n": n, "media_d": d.mean(), "ic95_inf": ci[0], "ic95_sup": ci[1],
        "dz": d.mean() / sd if sd > 0 else float("nan"),
        "seeds_com_d_positivo": int((d > 0).sum()), "shapiro_p": sw_p,
        "teste": "t pareado" if normal else "Wilcoxon", "p": p_t if normal else p_w,
        "p_t": p_t, "p_wilcoxon": p_w,
    }


def flip(alt):
    return {"greater": "less", "less": "greater", "two-sided": "two-sided"}[alt]


def main():
    df = pd.read_csv(RUNS)
    OUT.mkdir(parents=True, exist_ok=True)
    n_seeds = df["seed"].nunique()
    attacks = list(dict.fromkeys(df["attack"]))
    print(f"{len(df)} linhas | {n_seeds} seeds | modelos: {sorted(df['model'].unique())} | {len(attacks)} ataques")
    expected = n_seeds * len(MODELS) * len(attacks)
    if len(df) != expected:
        print(f"ATENÇÃO: eram esperadas {expected} linhas (seeds x modelos x ataques).")

    # ---------- resumo por seed ----------
    seed_df = (df.groupby(["model", "seed"])
                 .agg(auc=("auc", "mean"), f1=("f1", "mean"), fpr=("fpr", "mean"),
                      mse=("mse_normal_baseline", "first"), epocas=("epochs", "first"),
                      treino_s=("train_seconds", "first"), params=("params", "first"))
                 .reset_index())
    wide = {m: seed_df[seed_df.model == m].set_index("seed").sort_index() for m in MODELS}
    seeds = wide["A"].index
    for m in MODELS:
        assert (wide[m].index == seeds).all(), "as seeds dos modelos não coincidem"

    # ---------- tabela de resumo ----------
    rows = []
    for m in MODELS:
        w = wide[m]
        rows.append({
            "modelo": LABEL[m], "parametros": int(w["params"].iloc[0]),
            "auc_media": w["auc"].mean(), "auc_dp": w["auc"].std(ddof=1),
            "f1_medio": w["f1"].mean(), "f1_dp": w["f1"].std(ddof=1),
            "falsos_alarmes_medio": w["fpr"].mean(),
            "mse_normal": w["mse"].mean(), "mse_dp": w["mse"].std(ddof=1),
            "epocas_mediana": w["epocas"].median(), "epocas_max": int(w["epocas"].max()),
            "treino_s_medio": w["treino_s"].mean(),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "resumo_modelos.csv", index=False)
    print("\n=== Resumo por modelo (média das 30 seeds; AUC e F1 = média dos 8 ataques) ===")
    print(summary.round(4).to_string(index=False))

    # ---------- comparações globais ----------
    glob = []
    for metric, label, better in (("mse", "QP1 erro de reconstrução (MSE, nós normais)", "less"),
                                  ("auc", "QP2 AUC-ROC (média dos ataques)", "greater"),
                                  ("f1", "QP2 F1 (média dos ataques)", "greater")):
        fr = stats.friedmanchisquare(*[wide[m][metric].to_numpy() for m in MODELS])
        for a, b, alt in COMPARISONS:
            alternative = alt if better == "greater" else flip(alt)
            r = paired(wide[a][metric], wide[b][metric], alternative)
            r.update({"metrica": label, "comparacao": f"{LABEL[a]} - {LABEL[b]}",
                      "h1": {"greater": "primeiro maior", "less": "primeiro menor",
                             "two-sided": "bilateral"}[alternative],
                      "friedman_p": fr.pvalue})
            glob.append(r)
    glob = pd.DataFrame(glob)
    cols = ["metrica", "comparacao", "h1", "media_d", "ic95_inf", "ic95_sup", "dz",
            "seeds_com_d_positivo", "shapiro_p", "teste", "p", "friedman_p"]
    glob[cols].to_csv(OUT / "comparacoes_globais.csv", index=False)
    print("\n=== Comparações globais pareadas (d = primeiro - segundo, mesma seed) ===")
    print(glob[cols].round(4).to_string(index=False))

    # ---------- por ataque ----------
    per = []
    for metric, better in (("auc", "greater"), ("f1", "greater")):
        for a, b, alt in COMPARISONS:
            fam = []
            for atk in attacks:
                xa = df[(df.model == a) & (df.attack == atk)].set_index("seed").sort_index()[metric]
                xb = df[(df.model == b) & (df.attack == atk)].set_index("seed").sort_index()[metric]
                r = paired(xa.to_numpy(), xb.to_numpy(), alt)
                r.update({"metrica": metric, "comparacao": f"{LABEL[a]} - {LABEL[b]}", "ataque": atk,
                          "media_primeiro": xa.mean(), "media_segundo": xb.mean()})
                fam.append(r)
            adj = holm([r["p"] for r in fam])
            for r, pa in zip(fam, adj):
                r["p_holm"] = pa
                per.append(r)
    per = pd.DataFrame(per)
    pcols = ["metrica", "comparacao", "ataque", "media_primeiro", "media_segundo", "media_d",
             "ic95_inf", "ic95_sup", "dz", "teste", "p", "p_holm"]
    per[pcols].to_csv(OUT / "comparacoes_por_ataque.csv", index=False)

    means = (df.groupby(["attack", "model"])[["auc", "f1"]].mean().unstack("model"))
    means.columns = [f"{m}_{LABEL[mod]}" for m, mod in means.columns]
    means.round(4).to_csv(OUT / "media_por_ataque.csv")
    print("\n=== AUC-ROC média por ataque e modelo (30 seeds) ===")
    print(means[[f"auc_{LABEL[m]}" for m in MODELS]].round(3).to_string())

    for comp in ("B - A", "B - B'"):
        sub = per[(per.metrica == "auc") & (per.comparacao == comp)]
        print(f"\n=== AUC por ataque, {comp} (H1: primeiro maior; p com correção de Holm) ===")
        print(sub[["ataque", "media_d", "ic95_inf", "ic95_sup", "dz", "teste", "p_holm"]]
              .round(4).to_string(index=False))

    # ---------- figuras ----------
    fig, ax = plt.subplots(figsize=(5, 4))
    data = [wide[m]["auc"].to_numpy() for m in MODELS]
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(MODELS) + 1))
    ax.set_xticklabels([LABEL[m] for m in MODELS])
    for i, v in enumerate(data, 1):
        ax.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-0.08, 0.08, len(v)), v, s=10, alpha=0.5)
    ax.set_ylabel("AUC-ROC (média dos 8 ataques)")
    ax.set_title("Distribuição por modelo (30 seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_auc_media_por_modelo.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.27
    x = np.arange(len(attacks))
    for i, m in enumerate(MODELS):
        g = df[df.model == m].groupby("attack")["auc"]
        mu = g.mean().reindex(attacks)
        sd = g.std(ddof=1).reindex(attacks)
        ax.bar(x + (i - 1) * width, mu, width, yerr=sd, capsize=2, label=LABEL[m])
    ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=30, ha="right")
    ax.set_ylabel("AUC-ROC (média ± dp das 30 seeds)")
    ax.set_ylim(0.4, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_auc_por_ataque.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(wide["B"]["auc"].to_numpy() - wide["A"]["auc"].to_numpy(), bins=12, alpha=0.6, label="B - A")
    ax.hist(wide["B"]["auc"].to_numpy() - wide["Bp"]["auc"].to_numpy(), bins=12, alpha=0.6, label="B - B'")
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("diferença de AUC-ROC (média dos ataques), por seed")
    ax.set_ylabel("número de seeds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_diferencas_pareadas.png", dpi=150)
    plt.close(fig)
    print(f"\nTabelas e figuras salvas em {OUT}/")


if __name__ == "__main__":
    main()
