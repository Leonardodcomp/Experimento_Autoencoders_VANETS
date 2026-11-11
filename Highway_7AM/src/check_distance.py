"""Teste: o campo sender.distance_to_road_edge separa benigno de ataque?

Para cada subconjunto de ataque informado, calcula:
  - estatísticas do campo para attacker = 0 e attacker = 1;
  - % de valores negativos em cada classe;
  - AUC-ROC usando SÓ esse campo como pontuação (0,5 = nenhuma informação;
    1,0 = separa perfeitamente; o script reporta a separação máxima entre as
    duas direções possíveis);
  - um histograma salvo em results/.

Uso (na raiz do projeto, com o ambiente .venv ativo):
    python src/check_distance.py data/extracted/InTAS_highway_2_constantPositionOffset
    python src/check_distance.py            # usa todos os ataques extraídos
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXTRACTED = Path("data/extracted")
OUT = Path("results")
PREFIX = "InTAS_highway_7"


def load(folder):
    rows = []
    for f in folder.rglob("veh_*.json"):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        for m in data:
            rows.append((m.get("attacker"), m["sender"].get("distance_to_road_edge")))
    df = pd.DataFrame(rows, columns=["attacker", "dist"])
    df["attacker"] = pd.to_numeric(df["attacker"], errors="coerce")
    df["dist"] = pd.to_numeric(df["dist"], errors="coerce")
    return df.dropna()


def analyze(name, df):
    print("=" * 70)
    print(f"{name}: {len(df)} mensagens")
    result = {"subconjunto": name}
    for cls, label in ((0, "benigna"), (1, "ataque")):
        d = df.loc[df.attacker == cls, "dist"]
        if d.empty:
            continue
        q = d.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(2).tolist()
        pct_neg = 100 * (d < 0).mean()
        print(f"  {label:8s} n={len(d):6d} média={d.mean():8.2f} "
              f"quantis(5/25/50/75/95)={q}  negativos={pct_neg:.2f}%")
        result[f"pct_neg_{label}"] = round(pct_neg, 2)

    if df.attacker.nunique() == 2:
        auc = roc_auc_score(df.attacker, df.dist)
        sep = max(auc, 1 - auc)
        print(f"  AUC-ROC usando só distance_to_road_edge: {auc:.3f} "
              f"(separação = {sep:.3f}; 0,5 = nada, 1,0 = perfeito)")
        result["auc_separacao"] = round(sep, 3)

    lo, hi = df.dist.quantile([0.01, 0.99])
    bins = np.linspace(lo, hi, 60)
    plt.figure(figsize=(8, 4))
    plt.hist(df.loc[df.attacker == 0, "dist"].clip(lo, hi), bins=bins,
             alpha=0.6, density=True, label="benigna")
    plt.hist(df.loc[df.attacker == 1, "dist"].clip(lo, hi), bins=bins,
             alpha=0.6, density=True, label="ataque")
    plt.axvline(0, color="k", linestyle="--", linewidth=0.8)
    plt.xlabel("distance_to_road_edge (m)")
    plt.ylabel("densidade")
    plt.title(name)
    plt.legend()
    plt.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT / f"dist_{name}.png", dpi=120)
    plt.close()
    return result


def main():
    if len(sys.argv) > 1:
        folders = [Path(a) for a in sys.argv[1:]]
    else:
        folders = sorted(p for p in EXTRACTED.iterdir()
                         if p.is_dir() and p.name.startswith(PREFIX + "_"))
    if not folders:
        print("Nenhuma pasta de ataque encontrada. Passe o caminho como argumento.")
        return

    results = []
    for folder in folders:
        name = folder.name[len(PREFIX) + 1:] if folder.name.startswith(PREFIX + "_") else folder.name
        df = load(folder)
        if df.empty:
            print(f"{name}: sem dados (confira a extração dos zips internos)")
            continue
        results.append(analyze(name, df))

    if results:
        summary = pd.DataFrame(results)
        summary.to_csv(OUT / "distance_check.csv", index=False)
        print("\n" + "=" * 70)
        print(summary.to_string(index=False))
        print(f"\nTabela salva em {OUT / 'distance_check.csv'}; gráficos em {OUT}/dist_*.png")


if __name__ == "__main__":
    main()
