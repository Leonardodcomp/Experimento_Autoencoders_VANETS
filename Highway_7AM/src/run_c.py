"""Grupo C (GNN só espaço): piloto e experimento completo, SEM alterar o protocolo congelado.

ADENDO AO PROTOCOLO. O grupo C foi acrescentado DEPOIS de os resultados de A, B' e B serem
conhecidos (para completar o desenho 2 x 2: espaço sem tempo). Isso precisa constar no trabalho.
Para não misturar versões, este script:
  - NÃO altera nodes.py, graphdata.py, models.py, run.py nem run_all.py; ele só importa
    run.py e troca a função que cria o modelo (R.make_model). Antes de rodar, confere que o
    código e os parâmetros congelados continuam idênticos aos de results/protocolo_congelado.json
    e recusa rodar se algo mudou;
  - usa os mesmos parâmetros, as mesmas seeds (1 a 30), os mesmos 8 ataques e o mesmo Test;
  - grava em arquivos separados (results/runs_test_C.csv e results/adendo_protocolo_C.json).
(O C roda pelo caminho "B" do run.py, que monta a adjacência de 100 m; só o modelo muda.)

Etapas (de dentro da pasta do cenário, com o .venv ativo):
  1) python src/models_c.py            teste de formas, sem dados
  2) python src/run_c.py --pilot       piloto: seed 0, Validation, 8 ataques (results/runs_pilot_C.csv)
  3) python src/run_c.py               30 seeds no Test; retomável (results/runs_test_C.csv)
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
import run_all as RA
from models_c import GNNAutoencoder

HERE = Path(__file__).resolve().parent
RESULTS = Path("results")
RUNS_C = RESULTS / "runs_test_C.csv"
PILOT_C = RESULTS / "runs_pilot_C.csv"
ADENDO = RESULTS / "adendo_protocolo_C.json"
EXTRA_FILES = ["models_c.py", "run_c.py"]


def make_c(kind, dev):
    return GNNAutoencoder(12, R.HIDDEN, R.LATENT).to(dev)


def check_frozen():
    """Confere que o protocolo e o código dos 90 treinos originais continuam idênticos."""
    path = RA.PROTOCOL
    if not path.exists():
        raise SystemExit(f"Não achei {path}. Rode o experimento original (run_all.py) antes.")
    old = json.loads(path.read_text(encoding="utf-8"))
    now = RA.protocol()
    diffs = [k for k in now if old["protocolo"].get(k) != now[k]]
    if diffs:
        raise SystemExit("O código ou os parâmetros congelados mudaram (" + ", ".join(diffs) +
                         "). Recuso rodar para não misturar versões.")
    return old


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def run_c(seed, split, attacks, verbose):
    R.make_model = make_c            # troca só o modelo; o resto do pipeline é o do run.py
    rows = R.run_one("B", seed, attacks, split, R.MAX_EPOCHS, DEV, verbose=verbose)
    for r in rows:
        r["model"] = "C"
    return rows


def main():
    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="seed 0, Validation (não entra nas 30 seeds)")
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=30)
    args = ap.parse_args()

    old = check_frozen()
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS.mkdir(exist_ok=True)

    if args.pilot:
        print("PILOTO do modelo C (seed 0, Validation). Parâmetros esperados: 5.778")
        print("(A linha 'Modelo B' abaixo é só o nome interno do caminho do run.py; o modelo é o C.)")
        rows = run_c(0, "Validation", RA.ATTACKS, verbose=True)
        pd.DataFrame(rows).to_csv(PILOT_C, mode="a", header=not PILOT_C.exists(), index=False)
        print(f"Piloto gravado em {PILOT_C}. Compare o 'MSE nos nós normais' e as AUC com o piloto do B.")
        return

    if not ADENDO.exists():
        ADENDO.write_text(json.dumps({
            "adendo_em": datetime.datetime.now().isoformat(timespec="seconds"),
            "motivo": "Grupo C (GNN so espaco) acrescentado para completar o desenho 2 x 2.",
            "decidido_depois_de_ver": "os resultados de A, B' e B no Test (90 execucoes)",
            "protocolo_original_congelado_em": old["congelado_em"],
            "codigo_original_inalterado": old["protocolo"]["codigo"],
            "codigo_do_adendo": {f: sha(HERE / f) for f in EXTRA_FILES},
            "parametros": {"seeds": f"{args.first} a {args.last}", "ataques": RA.ATTACKS,
                           "conjunto_de_avaliacao": "Test", "T": R.T, "raio_m": R.RADIUS,
                           "oculto": R.HIDDEN, "latente": R.LATENT, "lr": R.LR},
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Adendo ao protocolo gravado em {ADENDO}")

    done = set()
    if RUNS_C.exists():
        df = pd.read_csv(RUNS_C)
        counts = df.groupby("seed").size()
        done = {int(s) for s, v in counts.items() if v == len(RA.ATTACKS)}
    todo = [s for s in range(args.first, args.last + 1) if s not in done]
    print(f"{len(done)} seeds já concluídas; {len(todo)} por fazer (modelo C).")

    t0 = time.time()
    for i, seed in enumerate(todo, 1):
        print("\n" + "=" * 78)
        print(f"[{i}/{len(todo)}] modelo C, seed {seed}")
        rows = run_c(seed, "Test", RA.ATTACKS, verbose=False)
        pd.DataFrame(rows).to_csv(RUNS_C, mode="a", header=not RUNS_C.exists(), index=False)
        r0 = rows[0]
        eta = (time.time() - t0) / i * (len(todo) - i)
        print(f"  treino {r0['train_seconds']:.0f} s ({r0['epochs']} épocas), avaliação {r0['eval_seconds']:.0f} s | "
              f"AUC média nos ataques: {sum(r['auc'] for r in rows) / len(rows):.4f} | "
              f"tempo restante estimado: {eta / 3600:.1f} h", flush=True)
    print("\nConcluído.")


if __name__ == "__main__":
    main()
