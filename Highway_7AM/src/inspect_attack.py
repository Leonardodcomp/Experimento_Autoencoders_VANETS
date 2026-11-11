"""Inspeção de um subconjunto de ataque do VeReMi NextGen.

Responde:
  - onde aparece o campo `attacker` e quais valores ele tem;
  - quantos arquivos de veículos e quantas mensagens há em Train/Validation/Test;
  - quais campos as mensagens têm (e se diferem do baseline);
  - como o total de mensagens se compara com as listas de ground truth.

Uso (na raiz do projeto, com o ambiente .venv ativo):
    python src/inspect_attack.py data/extracted/InTAS_highway_2_constantPositionOffset
"""
import json
import sys
from collections import Counter
from pathlib import Path

if len(sys.argv) < 2:
    print("Informe a pasta extraída do ataque, por exemplo:")
    print("  python src/inspect_attack.py data/extracted/InTAS_highway_7")
    sys.exit(1)

ROOT = Path(sys.argv[1])
SPLITS = ["Train", "Validation", "Test"]
MAX_PRINT = 1800


def split_of(path):
    for part in path.parts:
        if part in SPLITS:
            return part
    return "?"


def main():
    files = sorted(ROOT.rglob("veh_*.json"))
    if not files:
        print(f"Nenhum veh_*.json encontrado em {ROOT.resolve()}")
        print("Confira se os zips internos (Train/Validation/Test) também foram extraídos.")
        return

    per_split = {s: [] for s in SPLITS + ["?"]}
    for f in files:
        per_split[split_of(f)].append(f)

    example_attack = None
    example_benign = None

    for s in SPLITS + ["?"]:
        flist = per_split[s]
        if not flist:
            continue
        n_msgs = 0
        attacker_counts = Counter()
        top_keys = Counter()
        sender_keys = Counter()
        for f in flist:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            for m in data:
                n_msgs += 1
                attacker_counts[m.get("attacker", "ausente")] += 1
                top_keys.update(m.keys())
                if isinstance(m.get("sender"), dict):
                    sender_keys.update(m["sender"].keys())
                if example_attack is None and m.get("attacker") == 1:
                    example_attack = (s, f.name, m)
                if example_benign is None and m.get("attacker") == 0:
                    example_benign = (s, f.name, m)

        print("=" * 70)
        print(f"{s}: {len(flist)} arquivos de veículos, {n_msgs} mensagens")
        total = sum(attacker_counts.values())
        for val, c in sorted(attacker_counts.items(), key=lambda x: str(x[0])):
            print(f"  attacker = {val}: {c} ({100 * c / total:.2f}%)")
        print(f"  Campos das mensagens: {dict(top_keys)}")
        print(f"  Campos de `sender`: {dict(sender_keys)}")

    for title, ex in (("Exemplo com attacker = 1", example_attack),
                      ("Exemplo com attacker = 0", example_benign)):
        print("=" * 70)
        if ex is None:
            print(f"{title}: nenhum encontrado")
            continue
        split, fname, msg = ex
        print(f"{title} ({split}, {fname}):")
        txt = json.dumps(msg, indent=2, ensure_ascii=False)
        print(txt if len(txt) <= MAX_PRINT else txt[:MAX_PRINT] + "\n...")


if __name__ == "__main__":
    main()
