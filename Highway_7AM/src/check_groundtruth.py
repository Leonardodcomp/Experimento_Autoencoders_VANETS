"""Como o ground truth se relaciona com as mensagens do baseline?

Hipótese: o ground truth tem uma entrada por mensagem ENVIADA, enquanto os
arquivos dos veículos têm uma linha por mensagem RECEBIDA (cada mensagem
enviada é recebida por vários veículos).

Para cada conjunto (Train/Validation/Test), o script mostra:
  - quantas mensagens recebidas existem no baseline;
  - quantas mensagens únicas (sender_id, messageID) existem no baseline;
  - quantas entradas tem o ground truth e quantas são únicas;
  - a interseção entre os dois conjuntos de pares;
  - todas as chaves de uma entrada do ground truth.

Uso (na raiz do projeto, com o ambiente .venv ativo):
    python src/check_groundtruth.py
"""
import json
from pathlib import Path

BASELINE = Path("data/extracted/InTAS_highway_7")
RAW = Path("data/raw")
GT_FILES = {
    "Train": "InTAS_highway_7_Train_groundTruth.json",
    "Validation": "InTAS_highway_7_Validation_groundTruth.json",
    "Test": "InTAS_highway_7_Test_ground_truth.json",
}


def split_of(path):
    for part in path.parts:
        if part in GT_FILES:
            return part
    return None


def key(m):
    return (str(m["sender_id"]), str(m["messageID"]))


def main():
    received = {s: 0 for s in GT_FILES}
    unique = {s: set() for s in GT_FILES}
    for f in BASELINE.rglob("veh_*.json"):
        s = split_of(f)
        if s is None:
            continue
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        for m in data:
            received[s] += 1
            unique[s].add(key(m))

    for s, fname in GT_FILES.items():
        path = RAW / fname
        print("=" * 70)
        print(f"{s}")
        if not path.exists():
            print(f"  Arquivo não encontrado: {path}")
            continue
        with open(path, encoding="utf-8") as fh:
            gt = json.load(fh)
        gt_pairs = [key(m) for m in gt]
        gt_unique = set(gt_pairs)
        inter = gt_unique & unique[s]

        print(f"  Baseline: {received[s]} mensagens recebidas, {len(unique[s])} únicas (sender_id, messageID)")
        if unique[s]:
            print(f"  Média de recepções por mensagem única: {received[s] / len(unique[s]):.2f}")
        print(f"  Ground truth: {len(gt)} entradas, {len(gt_unique)} únicas")
        print(f"  Pares do ground truth presentes no baseline: {len(inter)} "
              f"({100 * len(inter) / max(len(gt_unique), 1):.1f}% do ground truth)")
        print(f"  Pares do baseline presentes no ground truth: {len(inter)} "
              f"({100 * len(inter) / max(len(unique[s]), 1):.1f}% do baseline)")
        if gt:
            print(f"  Chaves de uma entrada do ground truth: {list(gt[0].keys())}")


if __name__ == "__main__":
    main()
