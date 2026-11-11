"""Calibra a construção do grafo do modelo B (STGNN) no Highway 7 AM.

Amostra grafos de 1 segundo (um receptor, todos os emissores que ele ouviu naquele
segundo) e mede as distâncias ENTRE os emissores, a partir das posições informadas:
  - quantos nós tem o grafo;
  - quantos vizinhos cada nó teria para vários raios (e quantos nós ficariam isolados);
  - a que distância fica o k-ésimo vizinho mais próximo, para vários valores de k.

Uso (na raiz do projeto, com o ambiente .venv ativo):
    python src/graph_density.py
    python src/graph_density.py data/processed/baseline_Train.pkl 3000
(O arquivo deve ser o baseline do Highway 7 AM; a contagem de mensagens é impressa
para conferir: Train do 7 AM = 1.380.604.)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/baseline_Train.pkl")
N_SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
RADII = [25, 50, 100, 150, 200]
KS = [4, 8, 16]
SEED = 0


def main():
    df = pd.read_pickle(PATH)
    print(f"{PATH}: {len(df)} mensagens")
    df["sec"] = df["rcvTime"] // 10**9
    groups = df.groupby(["receiver_id", "sec"]).indices
    keys = list(groups.keys())
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(len(keys), size=min(N_SAMPLES, len(keys)), replace=False)
    print(f"Grafos de 1 s disponíveis: {len(keys)}; amostrados: {len(chosen)}")

    x_all = df["sender_x"].to_numpy()
    y_all = df["sender_y"].to_numpy()
    sid_all = df["sender_id"].to_numpy()

    sizes = []
    neigh = {r: [] for r in RADII}
    isolated = {r: [] for r in RADII}
    kth = {k: [] for k in KS}

    for i in chosen:
        idx = groups[keys[i]]
        # um nó por emissor (se houver duas mensagens no mesmo segundo, fica a primeira)
        _, first = np.unique(sid_all[idx], return_index=True)
        idx = idx[first]
        x, y = x_all[idx], y_all[idx]
        n = len(idx)
        sizes.append(n)
        if n < 2:
            continue
        d = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
        for r in RADII:
            deg = (d <= r).sum(axis=1) - 1
            neigh[r].append(deg)
            isolated[r].append(deg == 0)
        d_sorted = np.sort(d, axis=1)
        for k in KS:
            if n > k:
                kth[k].append(d_sorted[:, k])

    s = pd.Series(sizes)
    print("\nNós por grafo de 1 s: "
          f"média={s.mean():.1f}  mediana={s.median():.0f}  p90={s.quantile(.9):.0f}  máx={s.max():.0f}")

    print("\nPor raio de ligação entre emissores (posições informadas):")
    print("  raio(m)  vizinhos/nó (média)  mediana  p90  nós isolados")
    for r in RADII:
        deg = np.concatenate(neigh[r])
        iso = np.concatenate(isolated[r]).mean() * 100
        print(f"  {r:6d}  {deg.mean():18.1f}  {np.median(deg):7.0f}  "
              f"{np.quantile(deg, .9):3.0f}  {iso:11.1f}%")

    print("\nPor k vizinhos mais próximos (distância até o k-ésimo vizinho, em m):")
    print("  k   mediana    p90    p99")
    for k in KS:
        v = np.concatenate(kth[k])
        print(f"  {k:2d}  {np.median(v):7.1f}  {np.quantile(v, .9):6.1f}  {np.quantile(v, .99):6.1f}")


if __name__ == "__main__":
    main()
