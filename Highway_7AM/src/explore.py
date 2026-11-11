"""Exploração inicial do VeReMi NextGen (Highway 2 AM).

Objetivo: descobrir a estrutura real dos arquivos (pastas, JSONs e campos)
antes de escrever o carregamento definitivo dos dados.

Uso (na raiz do projeto, com o ambiente .venv ativo):
    python src/explore.py
    python src/explore.py data/extracted/InTAS_highway_2
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/extracted/InTAS_highway_7")
RAW = Path("data/raw")
MAX_STR = 70


def fmt_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024


def short(v):
    s = repr(v)
    return s if len(s) <= MAX_STR else s[:MAX_STR] + "..."


def preview(obj, depth=0, max_depth=3, indent="  "):
    """Mostra a estrutura de um objeto JSON de forma resumida."""
    pad = indent * depth
    if isinstance(obj, dict):
        print(f"{pad}dict com {len(obj)} chaves")
        for k, v in list(obj.items())[:12]:
            if isinstance(v, (dict, list)) and depth < max_depth:
                print(f"{pad}{indent}{k!r}:")
                preview(v, depth + 2, max_depth, indent)
            else:
                print(f"{pad}{indent}{k!r}: {type(v).__name__} = {short(v)}")
        if len(obj) > 12:
            print(f"{pad}{indent}... (+{len(obj) - 12} chaves)")
    elif isinstance(obj, list):
        print(f"{pad}list com {len(obj)} itens")
        if obj:
            print(f"{pad}{indent}primeiro item:")
            if isinstance(obj[0], (dict, list)) and depth < max_depth:
                preview(obj[0], depth + 2, max_depth, indent)
            else:
                print(f"{pad}{indent}{indent}{type(obj[0]).__name__} = {short(obj[0])}")
    else:
        print(f"{pad}{type(obj).__name__} = {short(obj)}")


def load_any(path):
    """Carrega JSON comum; se falhar, tenta JSON Lines (um objeto por linha)."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    try:
        return json.loads(text), "json"
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows, "jsonl"


def tree_summary(root, max_depth=3, max_lines=40):
    print(f"Estrutura de pastas de {root} (até {max_depth} níveis):")
    shown = 0
    for folder in sorted(p for p in root.rglob("*") if p.is_dir()):
        depth = len(folder.relative_to(root).parts)
        if depth > max_depth:
            continue
        if shown >= max_lines:
            print("  ... (lista cortada)")
            break
        files = [f for f in folder.iterdir() if f.is_file()]
        exts = dict(Counter(f.suffix.lower() for f in files))
        print(f"{'  ' * depth}{folder.name}/  ({len(files)} arquivos {exts})")
        shown += 1
    top_files = [f for f in root.iterdir() if f.is_file()]
    if top_files:
        print(f"Arquivos na raiz: {[f.name for f in top_files[:10]]}")


def main():
    if not ROOT.exists():
        print(f"Pasta não encontrada: {ROOT.resolve()}")
        print("Rode o script na raiz do projeto ou passe o caminho como argumento.")
        return

    print("=" * 70)
    tree_summary(ROOT)

    print("=" * 70)
    all_files = [f for f in ROOT.rglob("*") if f.is_file()]
    total = sum(f.stat().st_size for f in all_files)
    by_ext = Counter(f.suffix.lower() for f in all_files)
    print(f"Total: {len(all_files)} arquivos, {fmt_size(total)}")
    print(f"Por extensão: {dict(by_ext)}")

    jsons = sorted(f for f in all_files if f.suffix.lower() == ".json")
    if jsons:
        print("=" * 70)
        first = jsons[0]
        print(f"Primeiro JSON: {first.relative_to(ROOT)} ({fmt_size(first.stat().st_size)})")
        data, kind = load_any(first)
        print(f"Formato detectado: {kind}")
        preview(data)
    else:
        print("Nenhum arquivo .json encontrado dentro da pasta extraída.")

    gt_files = sorted(RAW.glob("InTAS_highway_2_*.json"))
    for gt in gt_files:
        print("=" * 70)
        print(f"Ground truth: {gt.name} ({fmt_size(gt.stat().st_size)})")
        data, kind = load_any(gt)
        print(f"Formato detectado: {kind}")
        preview(data, max_depth=2)


if __name__ == "__main__":
    main()
