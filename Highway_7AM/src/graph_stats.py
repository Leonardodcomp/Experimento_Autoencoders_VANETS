"""Estatísticas para decidir como construir o grafo do modelo B (STGNN).

Lê as tabelas do baseline geradas por load_data.py e mede:
  - quantas mensagens cada receptor recebe;
  - quantas mensagens cada par (receptor, emissor) tem;
  - o intervalo entre mensagens consecutivas do mesmo emissor no mesmo receptor;
  - quantos emissores distintos um receptor ouve em janelas de 1, 5, 10 e 30 s
    (tamanho do grafo local de um receptor);
  - quantos emissores distintos existem em janelas de 5 s considerando todos
    os receptores juntos (tamanho de um grafo global).

Uso (na raiz do projeto, com o ambiente .venv ativo, depois de rodar load_data.py):
    python src/graph_stats.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
SPLITS = ["Train", "Validation", "Test"]
WINDOWS = [1, 5, 10, 30]


def describe(label, s):
    s = pd.Series(s)
    q = s.quantile([0.5, 0.9, 0.99])
    print(f"  {label}: média={s.mean():.2f}  mediana={q[0.5]:.1f}  "
          f"p90={q[0.9]:.1f}  p99={q[0.99]:.1f}  máx={s.max():.1f}")


def main():
    for split in SPLITS:
        path = PROCESSED / f"baseline_{split}.pkl"
        if not path.exists():
            print(f"{split}: arquivo não encontrado ({path}). Rode load_data.py antes.")
            continue
        df = pd.read_pickle(path)
        df["t"] = df["rcvTime"] / 1e9

        print("=" * 70)
        print(f"{split}: {len(df)} mensagens, {df['receiver_id'].nunique()} receptores, "
              f"{df['sender_id'].nunique()} emissores distintos")

        describe("Mensagens por receptor", df.groupby("receiver_id").size())
        describe("Mensagens por par (receptor, emissor)",
                 df.groupby(["receiver_id", "sender_id"]).size())

        gaps = (df.sort_values(["receiver_id", "sender_id", "t"])
                  .groupby(["receiver_id", "sender_id"])["t"].diff().dropna())
        describe("Intervalo entre mensagens do mesmo emissor (s)", gaps)

        print("  Emissores distintos ouvidos por um receptor, por janela de tempo:")
        for win in WINDOWS:
            counts = []
            for _, g in df.groupby("receiver_id"):
                t = g["t"].to_numpy()
                codes = pd.factorize(g["sender_id"])[0]
                for i in range(len(t)):
                    mask = (t >= t[i] - win) & (t <= t[i])
                    counts.append(len(np.unique(codes[mask])))
            describe(f"janela de {win:>2d} s", counts)

        bins = (df["sendTime"] / 1e9 // 5).astype(int)
        describe("Emissores distintos por janela global de 5 s",
                 df.groupby(bins)["sender_id"].nunique())


if __name__ == "__main__":
    main()
