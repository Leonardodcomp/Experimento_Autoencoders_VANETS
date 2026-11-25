"""Análise de SENSIBILIDADE (fora do protocolo congelado): B e C com treino mais paciente.

Pergunta: o B (e o C) foram interrompidos cedo demais pela parada antecipada (paciência 5,
mediana de 39,5 e 32,5 épocas)? Se treinados com paciência 15 e até 200 épocas, a AUC de B - C muda?

O que NÃO muda: dados, 12 atributos, arquiteturas, lr, lotes, limiar (percentil 95 do Validation),
Test, 8 ataques e as mesmas seeds do experimento principal (começam com a mesma inicialização).
O que muda (só aqui): PATIENCE 5 -> 15 e MAX_EPOCHS 100 -> 200, para B e C.

Este script NÃO altera nenhum arquivo do experimento: importa run.py e troca R.PATIENCE em memória
DEPOIS de conferir que o protocolo congelado está intacto. Saídas separadas:
  results/runs_sens_BC.csv            (modelos rotulados B_p15 e C_p15)
  results/adendo_sensibilidade.json   (registro de que foi feita depois, fora do plano)

Uso (de dentro da pasta do cenário, com o .venv ativo):
  python src/run_sensibilidade.py --first 1 --last 10      # padrão: 10 seeds, retomável
Estimativa: ~10 a 15 min por seed e modelo (B e C), logo ~3 a 5 h para 10 seeds.
Intercala B e C na mesma seed, então resultados parciais já são pareados.
"""
import argparse
import datetime
import json
import time
from pathlib import Path

import pandas as pd
import torch

import run as R
import run_all as RA
import run_c as RC

RESULTS = Path("results")
OUT = RESULTS / "runs_sens_BC.csv"
ADENDO = RESULTS / "adendo_sensibilidade.json"
PATIENCE_SENS = 15
MAX_EPOCHS_SENS = 200
ORIG_MAKE_MODEL = R.make_model


def run_model(label, seed, dev):
    R.make_model = ORIG_MAKE_MODEL if label == "B" else RC.make_c
    rows = R.run_one("B", seed, RA.ATTACKS, "Test", MAX_EPOCHS_SENS, dev, verbose=False)
    for r in rows:
        r["model"] = f"{label}_p15"
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=10)
    args = ap.parse_args()

    RC.check_frozen()                     # confere o protocolo ANTES de mexer na paciência
    R.PATIENCE = PATIENCE_SENS            # só em memória; nenhum arquivo é alterado
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULTS.mkdir(exist_ok=True)

    if not ADENDO.exists():
        ADENDO.write_text(json.dumps({
            "feito_em": datetime.datetime.now().isoformat(timespec="seconds"),
            "natureza": "analise de sensibilidade, fora do protocolo congelado, decidida depois de ver os resultados",
            "pergunta": "B e C estavam subtreinados pela parada antecipada (paciencia 5)?",
            "mudanca": {"paciencia": f"5 -> {PATIENCE_SENS}", "max_epocas": f"100 -> {MAX_EPOCHS_SENS}", "modelos": ["B", "C"]},
            "seeds": f"{args.first} a {args.last}",
            "resto_identico": "dados, atributos, arquiteturas, lr, lotes, limiar, Test, ataques, seeds",
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    done = set()
    if OUT.exists():
        df = pd.read_csv(OUT)
        for (m, s), n in df.groupby(["model", "seed"]).size().items():
            if n == len(RA.ATTACKS):
                done.add((m, int(s)))
    jobs = [(lab, s) for s in range(args.first, args.last + 1) for lab in ("B", "C") if (f"{lab}_p15", s) not in done]
    print(f"{len(done)} execuções já concluídas; {len(jobs)} por fazer (paciência {PATIENCE_SENS}, máx. {MAX_EPOCHS_SENS}).")

    t0 = time.time()
    for i, (lab, seed) in enumerate(jobs, 1):
        print("\n" + "=" * 78 + f"\n[{i}/{len(jobs)}] modelo {lab} (paciência {PATIENCE_SENS}), seed {seed}", flush=True)
        rows = run_model(lab, seed, dev)
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
        r0 = rows[0]
        eta = (time.time() - t0) / i * (len(jobs) - i)
        print(f"  treino {r0['train_seconds']:.0f} s ({r0['epochs']} épocas) | AUC média {sum(r['auc'] for r in rows) / len(rows):.4f} "
              f"| restante estimado {eta / 3600:.1f} h", flush=True)
    print("\nConcluído. Rode: python src/analyze_sens.py")


if __name__ == "__main__":
    main()
