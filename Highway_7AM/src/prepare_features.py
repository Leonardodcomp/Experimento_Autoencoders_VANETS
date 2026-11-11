"""Transforma as tabelas de load_data.py nas entradas numéricas dos modelos.

Para cada tabela em data/processed/ (baseline e ataques, Train/Validation/Test) gera
data/features/<subconjunto>_<conjunto>.npz com:
  X            (N, 11) float32  atributos por mensagem, já normalizados
  receiver     (N,)    int32    código do receptor
  sender       (N,)    int32    código do emissor (sender_id)
  t            (N,)    float64  instante de recebimento, em segundos
  attacker     (N,)    int8     0 benigna, 1 ataque, -1 sem rótulo (baseline)
  pos          (N, 2)  float32  posição informada do emissor (x, y), usada só nas arestas
  group_start  (G+1,)  int64    início de cada grupo (receptor, segundo) em X; o último
                                valor é N. As linhas já estão ordenadas por receptor e tempo.

Atributos por mensagem (11):
  dx, dy, dist        posição RELATIVA do emissor ao receptor (m) e a distância entre eles
  s_spd, s_acl        velocidade e aceleração do emissor
  s_hsin, s_hcos      direção do emissor (seno e cosseno)
  r_spd, r_acl        velocidade e aceleração do receptor
  r_hsin, r_hcos      direção do receptor (seno e cosseno)

Fora das entradas, de propósito: coordenadas absolutas (o Test é de outra área), tempos,
identificadores, campos *_noise, driversProfile e distance_to_road_edge (não existe no baseline).

A normalização (média e desvio) é ajustada SÓ no Train do baseline e aplicada em tudo.

Uso (de dentro da pasta do cenário, com o ambiente .venv ativo):
    python src/prepare_features.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path("data/processed")
OUT = Path("data/features")
RESULTS = Path("results")
NORM_PATH = OUT / "norm.json"

CONT = ["dx", "dy", "dist", "s_spd", "s_acl", "r_spd", "r_acl"]   # padronizados
ANG = ["s_hsin", "s_hcos", "r_hsin", "r_hcos"]                    # já estão em [-1, 1]
FEATURES = CONT + ANG


def make_features(df):
    f = pd.DataFrame(index=df.index)
    f["dx"] = df["sender_x"] - df["receiver_x"]
    f["dy"] = df["sender_y"] - df["receiver_y"]
    f["dist"] = np.hypot(f["dx"], f["dy"])
    f["s_spd"] = df["sender_spd"]
    f["s_acl"] = df["sender_acl"]
    f["r_spd"] = df["receiver_spd"]
    f["r_acl"] = df["receiver_acl"]
    for role, p in (("sender", "s"), ("receiver", "r")):
        rad = np.deg2rad(df[f"{role}_hed"])
        f[f"{p}_hsin"] = np.sin(rad)
        f[f"{p}_hcos"] = np.cos(rad)
    return f[FEATURES]


def fit_norm():
    path = PROCESSED / "baseline_Train.pkl"
    df = pd.read_pickle(path)
    f = make_features(df)
    stats = {c: {"mean": float(f[c].mean()), "std": float(f[c].std())} for c in CONT}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(NORM_PATH, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Normalização ajustada no Train do baseline ({len(df)} mensagens) -> {NORM_PATH}")
    for c in CONT:
        print(f"  {c:6s} média={stats[c]['mean']:10.3f}  desvio={stats[c]['std']:9.3f}")
    return stats


def process(path, stats):
    name, split = path.stem.rsplit("_", 1)
    df = pd.read_pickle(path)
    f = make_features(df)
    for c in CONT:
        f[c] = (f[c] - stats[c]["mean"]) / stats[c]["std"]

    X = f.to_numpy(dtype=np.float32)
    non_finite = int((~np.isfinite(X)).sum())
    receiver = pd.factorize(df["receiver_id"])[0].astype(np.int32)
    sender = pd.factorize(df["sender_id"])[0].astype(np.int32)
    t = df["rcvTime"].to_numpy(dtype=np.float64) / 1e9
    sec = np.floor(t).astype(np.int64)
    attacker = df["attacker"].fillna(-1).astype(np.int8).to_numpy()
    pos = df[["sender_x", "sender_y"]].to_numpy(dtype=np.float32)

    change = np.ones(len(df), dtype=bool)
    change[1:] = (receiver[1:] != receiver[:-1]) | (sec[1:] != sec[:-1])
    group_start = np.append(np.flatnonzero(change), len(df)).astype(np.int64)
    n_groups = len(group_start) - 1

    np.savez(OUT / f"{name}_{split}.npz", X=X, receiver=receiver, sender=sender, t=t,
             attacker=attacker, pos=pos, group_start=group_start)

    row = {
        "subconjunto": name, "conjunto": split, "mensagens": len(df),
        "grupos_receptor_segundo": n_groups,
        "tamanho_medio_do_grupo": round(len(df) / n_groups, 1),
        "attacker_0": int((attacker == 0).sum()), "attacker_1": int((attacker == 1).sum()),
        "valores_nao_finitos": non_finite,
    }
    print(f"  {name:26s} {split:10s} N={row['mensagens']:8d} grupos={n_groups:7d} "
          f"tam.médio={row['tamanho_medio_do_grupo']:5.1f}  attacker 0/1="
          f"{row['attacker_0']}/{row['attacker_1']}  não-finitos={non_finite}")
    return row, f


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not (PROCESSED / "baseline_Train.pkl").exists():
        print("Falta data/processed/baseline_Train.pkl. Rode load_data.py antes.")
        return
    stats = fit_norm()

    rows = []
    print("\nProcessando tabelas:")
    for path in sorted(PROCESSED.glob("*.pkl")):
        row, f = process(path, stats)
        rows.append(row)
        if row["subconjunto"] == "baseline" and row["conjunto"] == "Train":
            print("    conferência (baseline Train, após normalizar): "
                  f"média máx. das colunas contínuas = {f[CONT].mean().abs().max():.4f}; "
                  f"desvio entre {f[CONT].std().min():.3f} e {f[CONT].std().max():.3f}")

    pd.DataFrame(rows).to_csv(RESULTS / "features_summary.csv", index=False)
    print(f"\nResumo salvo em {RESULTS / 'features_summary.csv'}; arrays em {OUT}/")


if __name__ == "__main__":
    main()
