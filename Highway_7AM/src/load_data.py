"""Converte os JSONs do VeReMi NextGen (Highway 2 AM) em tabelas pandas limpas.

Para cada subconjunto (baseline ou ataque) e cada conjunto (Train/Validation/Test),
gera data/processed/<subconjunto>_<conjunto>.pkl com UMA LINHA POR MENSAGEM RECEBIDA,
ordenada por receptor e tempo de recebimento.

Cuidados que o script resolve:
  - converte textos em números (no baseline os campos vêm como texto);
  - separa "x,y,z" em colunas numéricas;
  - o baseline não tem `attacker`: a coluna fica vazia (NA);
  - mostra quantas mensagens têm distance_to_road_edge em cada subconjunto.

Uso (na raiz do projeto, com o ambiente .venv ativo):
    python src/load_data.py baseline constantPositionOffset     # teste rápido
    python src/load_data.py                                      # todos os subconjuntos extraídos
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXTRACTED = Path("data/extracted")
OUT = Path("data/processed")
PREFIX = "InTAS_highway_7"
SPLITS = ["Train", "Validation", "Test"]


def split_of(path):
    for part in path.parts:
        if part in SPLITS:
            return part
    return None


def parse_xyz(text):
    """'x,y,z' -> (x, y, z) como números."""
    parts = [float(p) for p in str(text).split(",")]
    return parts[0], parts[1], parts[2]


def flatten(msg, receiver_id):
    row = {
        "receiver_id": receiver_id,
        "rcvTime": int(msg["rcvTime"]),
        "sendTime": int(msg["sendTime"]),
        "sender_id": str(msg["sender_id"]),
        "sender_alias": int(msg["sender_alias"]),
        "messageID": int(msg["messageID"]),
        "attacker": msg.get("attacker", None),
    }
    for role in ("sender", "receiver"):
        d = msg[role]
        x, y, z = parse_xyz(d["pos"])
        nx, ny, nz = parse_xyz(d["pos_noise"])
        row[f"{role}_x"], row[f"{role}_y"], row[f"{role}_z"] = x, y, z
        row[f"{role}_pos_noise_x"] = nx
        row[f"{role}_pos_noise_y"] = ny
        row[f"{role}_pos_noise_z"] = nz
        for key in ("spd", "spd_noise", "acl", "acl_noise", "hed", "hed_noise"):
            row[f"{role}_{key}"] = float(d[key])
        row[f"{role}_driversProfile"] = str(d["driversProfile"]).upper()
    dist = msg["sender"].get("distance_to_road_edge")
    row["sender_distance_to_road_edge"] = float(dist) if dist is not None else np.nan
    return row


def folder_of(name):
    return EXTRACTED / (PREFIX if name == "baseline" else f"{PREFIX}_{name}")


def process(name):
    folder = folder_of(name)
    if not folder.exists():
        print(f"  Pasta não encontrada: {folder}")
        return
    files = sorted(folder.rglob("veh_*.json"))
    by_split = {s: [] for s in SPLITS}
    for f in files:
        s = split_of(f)
        if s:
            by_split[s].append(f)

    OUT.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        rows = []
        for f in by_split[s]:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            for m in data:
                rows.append(flatten(m, f.stem))
        if not rows:
            print(f"  {name} / {s}: sem dados")
            continue
        df = pd.DataFrame(rows)
        df["attacker"] = pd.to_numeric(df["attacker"], errors="coerce").astype("Int8")
        df["driver_sender"] = df["sender_driversProfile"].astype("category")
        df = df.sort_values(["receiver_id", "rcvTime"]).reset_index(drop=True)
        df.to_pickle(OUT / f"{name}_{s}.pkl")

        n_att = int((df["attacker"] == 1).sum())
        n_ben = int((df["attacker"] == 0).sum())
        has_dist = 100 * df["sender_distance_to_road_edge"].notna().mean()
        print(f"  {name:28s} {s:10s} {len(df):7d} mensagens | attacker 0: {n_ben:6d} "
              f"1: {n_att:6d} | distance_to_road_edge presente em {has_dist:5.1f}%")


def main():
    if len(sys.argv) > 1:
        names = sys.argv[1:]
    else:
        names = ["baseline"] + sorted(
            p.name[len(PREFIX) + 1:] for p in EXTRACTED.iterdir()
            if p.is_dir() and p.name.startswith(PREFIX + "_")
        )
    for name in names:
        print(f"Processando {name} ...", flush=True)
        process(name)
    print(f"\nTabelas salvas em {OUT}/")


if __name__ == "__main__":
    main()
