"""Análise do desenho 2 x 2 (A, B', C, B), com o grupo C acrescentado DEPOIS dos 3 primeiros.

PLANO DE ANÁLISE DO ADENDO (declarado antes de ver os resultados do C)
  Dados: results/runs_test.csv (A, B', B; protocolo congelado) + results/runs_test_C.csv (C).
  Unidade: a seed (as mesmas 30 seeds em todos os modelos), comparações PAREADAS, d = primeiro - segundo.

  As três comparações originais (B - A principal, B - B' secundária, B' - A descritiva) ficam como
  estavam em analyze.py; aqui elas só são repetidas, para referência, sem nova correção.

  Comparações NOVAS (todas PÓS-HOC e BILATERAIS, porque o C foi incluído depois de ver os resultados):
    C - A          efeito do espaço sem tempo
    B - C          efeito do tempo com espaço
    interação      (B - C) - (B' - A): o espaço ajuda mais (ou menos) quando há tempo?
                   Equivale a B - C - B' + A; interação ~ 0 significa efeitos aditivos.
  Correção de Holm entre essas três, em cada métrica (MSE, AUC-ROC, F1) e, por ataque, em cada
  comparação nos 8 ataques. Teste: Shapiro-Wilk nas 30 diferenças -> t pareado ou Wilcoxon (analyze.paired).
  Friedman nos quatro modelos. Tamanho de efeito dz e IC 95% da média de d. Nível 0,05.

Uso (de dentro da pasta do cenário, com o .venv ativo, depois de rodar run_c.py):
    python src/analyze_2x2.py
Saídas em results/analise_2x2/.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyze import holm, paired  # noqa: E402

RUNS = Path("results/runs_test.csv")
RUNS_C = Path("results/runs_test_C.csv")
OUT = Path("results/analise_2x2")
MODELS = ["A", "Bp", "C", "B"]
LABEL = {"A": "A", "Bp": "B'", "C": "C", "B": "B"}
METRICS = (("mse", "QP1 MSE nos nós normais (menor é melhor)"),
           ("auc", "QP2 AUC-ROC (média dos ataques)"),
           ("f1", "QP2 F1 (média dos ataques)"))


def effects(get):
    """get(modelo) -> vetor por seed. Retorna {nome: (vetor_x, vetor_y)} das comparações."""
    a, bp, c, b = (get(m) for m in ("A", "Bp", "C", "B"))
    return {
        "C - A (espaço sem tempo)": (c, a, "novo"),
        "B - C (tempo com espaço)": (b, c, "novo"),
        "interação (B - C) - (B' - A)": (b - c, bp - a, "novo"),
        "B' - A (tempo sem espaço)": (bp, a, "original"),
        "B - B' (espaço com tempo)": (b, bp, "original"),
        "B - A (efeito total)": (b, a, "original"),
    }


def main():
    df = pd.concat([pd.read_csv(RUNS), pd.read_csv(RUNS_C)], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    attacks = list(dict.fromkeys(df["attack"]))
    n_seeds = df["seed"].nunique()
    print(f"{len(df)} linhas | {n_seeds} seeds | modelos: {sorted(df['model'].unique())} | {len(attacks)} ataques")
    expected = n_seeds * len(MODELS) * len(attacks)
    if len(df) != expected:
        print(f"ATENÇÃO: eram esperadas {expected} linhas (seeds x modelos x ataques).")

    seed_df = (df.groupby(["model", "seed"])
                 .agg(auc=("auc", "mean"), f1=("f1", "mean"), fpr=("fpr", "mean"),
                      mse=("mse_normal_baseline", "first"), epocas=("epochs", "first"),
                      treino_s=("train_seconds", "first"), params=("params", "first"))
                 .reset_index())
    wide = {m: seed_df[seed_df.model == m].set_index("seed").sort_index() for m in MODELS}
    seeds = wide["A"].index
    for m in MODELS:
        assert (wide[m].index == seeds).all(), "as seeds dos modelos não coincidem"

    # ---------- resumo por modelo ----------
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
    summary.to_csv(OUT / "resumo_4_modelos.csv", index=False)
    print("\n=== Resumo por modelo (média das seeds; AUC e F1 = média dos 8 ataques) ===")
    print(summary.round(4).to_string(index=False))
    full = df[df.epochs >= 100].groupby("model").size() // len(attacks)
    print("\nExecuções que atingiram o máximo de 100 épocas:", dict(full) if len(full) else "nenhuma")

    # ---------- comparações globais ----------
    glob = []
    for metric, label in METRICS:
        fr = stats.friedmanchisquare(*[wide[m][metric].to_numpy() for m in MODELS])
        eff = effects(lambda m: wide[m][metric].to_numpy())
        fam = []
        for name, (x, y, kind) in eff.items():
            r = paired(x, y, "two-sided")
            r.update({"metrica": label, "comparacao": name, "origem": kind, "friedman_p": fr.pvalue})
            fam.append(r)
        new = [r for r in fam if r["origem"] == "novo"]
        for r, pa in zip(new, holm([r["p"] for r in new])):
            r["p_holm"] = pa
        glob.extend(fam)
    glob = pd.DataFrame(glob)
    cols = ["metrica", "comparacao", "origem", "media_d", "ic95_inf", "ic95_sup", "dz",
            "seeds_com_d_positivo", "shapiro_p", "teste", "p", "p_holm", "friedman_p"]
    glob[cols].to_csv(OUT / "comparacoes_globais_2x2.csv", index=False)
    print("\n=== Comparações globais pareadas, bilaterais (p_holm só entre as 3 novas) ===")
    print(glob[cols].round(4).to_string(index=False))

    # ---------- por ataque ----------
    per = []
    for metric in ("auc", "f1"):
        names = list(effects(lambda m: np.zeros(1)).keys())
        for name in names:
            fam = []
            for atk in attacks:
                def get(m, atk=atk):
                    return (df[(df.model == m) & (df.attack == atk)]
                            .set_index("seed").sort_index()[metric].to_numpy())
                x, y, kind = effects(get)[name]
                r = paired(x, y, "two-sided")
                r.update({"metrica": metric, "comparacao": name, "origem": kind, "ataque": atk})
                fam.append(r)
            for r, pa in zip(fam, holm([r["p"] for r in fam])):
                r["p_holm"] = pa
                per.append(r)
    per = pd.DataFrame(per)
    pcols = ["metrica", "comparacao", "origem", "ataque", "media_d", "ic95_inf", "ic95_sup", "dz",
             "teste", "p", "p_holm"]
    per[pcols].to_csv(OUT / "comparacoes_por_ataque_2x2.csv", index=False)

    means = df.groupby(["attack", "model"])[["auc", "f1"]].mean().unstack("model")
    means.columns = [f"{m}_{LABEL[mod]}" for m, mod in means.columns]
    means.round(4).to_csv(OUT / "media_por_ataque_4_modelos.csv")
    print("\n=== AUC-ROC média por ataque e modelo ===")
    print(means[[f"auc_{LABEL[m]}" for m in MODELS]].round(3).to_string())
    for name in ("C - A (espaço sem tempo)", "B - C (tempo com espaço)", "interação (B - C) - (B' - A)"):
        sub = per[(per.metrica == "auc") & (per.comparacao == name)]
        print(f"\n=== AUC por ataque: {name} (bilateral; p com Holm nos 8 ataques) ===")
        print(sub[["ataque", "media_d", "ic95_inf", "ic95_sup", "dz", "teste", "p_holm"]]
              .round(4).to_string(index=False))

    # ---------- figuras ----------
    mu = {m: wide[m]["auc"].mean() for m in MODELS}
    sd = {m: wide[m]["auc"].std(ddof=1) for m in MODELS}
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar([0, 1], [mu["A"], mu["C"]], [sd["A"], sd["C"]], marker="o", capsize=3, label="sem tempo (A, C)")
    ax.errorbar([0, 1], [mu["Bp"], mu["B"]], [sd["Bp"], sd["B"]], marker="s", capsize=3, label="com tempo (B', B)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["sem espaço", "com espaço"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("AUC-ROC (média dos 8 ataques ± dp)")
    ax.set_title("Desenho 2 x 2: linhas paralelas = efeitos aditivos")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig_interacao_2x2.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.2
    x = np.arange(len(attacks))
    for i, m in enumerate(MODELS):
        g = df[df.model == m].groupby("attack")["auc"]
        ax.bar(x + (i - 1.5) * width, g.mean().reindex(attacks), width,
               yerr=g.std(ddof=1).reindex(attacks), capsize=2, label=LABEL[m])
    ax.axhline(0.5, color="k", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(attacks, rotation=30, ha="right")
    ax.set_ylabel("AUC-ROC (média ± dp das seeds)")
    ax.set_ylim(0.4, 1.02)
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(OUT / "fig_auc_por_ataque_4_modelos.png", dpi=150)
    plt.close(fig)
    print(f"\nTabelas e figuras salvas em {OUT}/")


if __name__ == "__main__":
    main()
