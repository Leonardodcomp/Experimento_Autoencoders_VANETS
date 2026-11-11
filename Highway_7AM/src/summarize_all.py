"""Resumo de todos os subconjuntos do Highway 2 AM (baseline + ataques).

Para cada pasta extraída em data/extracted/ que começa com InTAS_highway_2,
conta, por conjunto (Train/Validation/Test):
  - arquivos de veículos;
  - mensagens;
  - mensagens com attacker = 0, attacker = 1 e sem o campo attacker;
  - percentual de mensagens de ataque.

Salva o resultado em results/dataset_summary.csv.

Uso (na raiz do projeto, com o ambiente .venv ativo):
    python src/summarize_all.py
"""
import csv
import json
from pathlib import Path

EXTRACTED = Path("data/extracted")
OUT = Path("results/dataset_summary.csv")
PREFIX = "InTAS_highway_7"
SPLITS = ["Train", "Validation", "Test"]


def split_of(path):
    for part in path.parts:
        if part in SPLITS:
            return part
    return None


def summarize(folder):
    stats = {
        s: {"arquivos": 0, "mensagens": 0, "attacker_0": 0, "attacker_1": 0, "sem_attacker": 0}
        for s in SPLITS
    }
    for f in folder.rglob("veh_*.json"):
        s = split_of(f)
        if s is None:
            continue
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        st = stats[s]
        st["arquivos"] += 1
        for m in data:
            st["mensagens"] += 1
            a = m.get("attacker")
            if a in (0, "0"):
                st["attacker_0"] += 1
            elif a in (1, "1"):
                st["attacker_1"] += 1
            else:
                st["sem_attacker"] += 1
    return stats


def main():
    folders = sorted(p for p in EXTRACTED.iterdir() if p.is_dir() and p.name.startswith(PREFIX))
    if not folders:
        print(f"Nenhuma pasta {PREFIX}* encontrada em {EXTRACTED.resolve()}")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    header = ["subconjunto", "conjunto", "arquivos", "mensagens",
              "attacker_0", "attacker_1", "sem_attacker", "pct_ataque"]
    rows = []
    for folder in folders:
        name = "baseline" if folder.name == PREFIX else folder.name[len(PREFIX) + 1:]
        print(f"Processando {name} ...", flush=True)
        stats = summarize(folder)
        for s in SPLITS:
            st = stats[s]
            labeled = st["attacker_0"] + st["attacker_1"]
            pct = round(100 * st["attacker_1"] / labeled, 2) if labeled else ""
            rows.append([name, s, st["arquivos"], st["mensagens"],
                         st["attacker_0"], st["attacker_1"], st["sem_attacker"], pct])

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)

    print("\n" + " | ".join(header))
    for r in rows:
        print(" | ".join(str(x) for x in r))
    print(f"\nSalvo em {OUT}")


if __name__ == "__main__":
    main()
