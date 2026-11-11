"""Experimento completo com o protocolo CONGELADO.

Roda as seeds 1 a 30 para os três modelos (A, Bp = B', B), na ordem seed por seed, e avalia no
TEST dos 8 ataques escolhidos. Os resultados vão para results/runs_test.csv.

  - É RETOMÁVEL: se for interrompido (Ctrl+C, queda de energia), rode de novo e ele continua
    do primeiro (modelo, seed) que falta.
  - Na primeira execução, grava results/protocolo_congelado.json com os parâmetros e um
    resumo (hash) dos arquivos de código. Nas execuções seguintes, se o código ou os
    parâmetros mudaram, ele RECUSA rodar (use --force só se souber o que está fazendo, e
    registre a mudança no trabalho).

Uso (de dentro da pasta do cenário, com o ambiente .venv ativo):
    python src/run_all.py
    python src/run_all.py --first 1 --last 3       # só algumas seeds (teste)
"""
import argparse
import datetime
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import torch

import run as R

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
PROTOCOL = RESULTS / "protocolo_congelado.json"
RUNS = RESULTS / "runs_test.csv"

ATTACKS = ["constantPositionOffset", "constantSpeedOffset", "reversedHeading", "feignedBraking",
           "timeDelayAttack", "dosAttack", "trafficCongestionSybil", "dataReplay"]
MODELS = ["A", "Bp", "B"]
CODE_FILES = ["nodes.py", "graphdata.py", "models.py", "run.py", "run_all.py"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def protocol():
    return {
        "T": R.T, "raio_m": R.RADIUS, "latente": R.LATENT, "oculto": R.HIDDEN, "lr": R.LR,
        "lote_mensagens_A": R.BS_MSG, "lote_grupos_B": R.BS_GRP, "max_epocas": R.MAX_EPOCHS,
        "paciencia": R.PATIENCE, "percentil_limiar": R.PERCENTILE,
        "modelos": MODELS, "ataques": ATTACKS, "conjunto_de_avaliacao": "Test",
        "codigo": {f: sha(HERE / f) for f in CODE_FILES},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=30)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    proto = protocol()
    if PROTOCOL.exists():
        old = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        if old["protocolo"] != proto and not args.force:
            print("O código ou os parâmetros mudaram desde o congelamento em "
                  f"{old['congelado_em']}. Recuso rodar para não misturar versões.")
            for k in proto:
                if old["protocolo"].get(k) != proto[k]:
                    print(f"  diferença em: {k}")
            return
    else:
        dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        PROTOCOL.write_text(json.dumps({
            "congelado_em": datetime.datetime.now().isoformat(timespec="seconds"),
            "ambiente": {"torch": torch.__version__, "dispositivo": dev_name},
            "protocolo": proto,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Protocolo congelado e gravado em {PROTOCOL}")

    done = set()
    if RUNS.exists():
        df = pd.read_csv(RUNS)
        counts = df.groupby(["model", "seed"]).size()
        done = {k for k, v in counts.items() if v == len(ATTACKS)}
    todo = [(k, s) for s in range(args.first, args.last + 1) for k in MODELS if (k, s) not in done]
    print(f"{len(done)} execuções já concluídas; {len(todo)} por fazer.")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    for i, (kind, seed) in enumerate(todo, 1):
        print("\n" + "=" * 78)
        print(f"[{i}/{len(todo)}] modelo {kind}, seed {seed}")
        rows = R.run_one(kind, seed, ATTACKS, "Test", R.MAX_EPOCHS, dev, verbose=False)
        pd.DataFrame(rows).to_csv(RUNS, mode="a", header=not RUNS.exists(), index=False)
        r0 = rows[0]
        elapsed = time.time() - t0
        eta = elapsed / i * (len(todo) - i)
        print(f"  treino {r0['train_seconds']:.0f} s ({r0['epochs']} épocas), avaliação {r0['eval_seconds']:.0f} s | "
              f"AUC média nos ataques: {sum(r['auc'] for r in rows) / len(rows):.4f} | "
              f"tempo restante estimado: {eta / 3600:.1f} h", flush=True)
    print("\nConcluído.")


if __name__ == "__main__":
    main()
