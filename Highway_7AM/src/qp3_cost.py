"""QP3: custo operacional (latência e memória) na inferência dos modelos A, B' (Bp) e B.

Medido no MESMO computador para os três modelos. Usa grafos reais do baseline/Test e pesos de
inicialização aleatória (os modelos treinados não foram salvos; latência e memória dependem da
arquitetura e do tamanho da entrada, não dos valores dos pesos).

O que é medido (por repetição; 30 repetições, cada uma com seed e amostra de grupos próprias, a
mesma amostra para os três modelos -> comparação pareada):
  lat_second_ms      tempo para pontuar UM SEGUNDO de tráfego de um receptor (todos os nós do
                     grupo), de ponta a ponta: montar o lote e a adjacência (B e B') + modelo.
                     No A, todas as mensagens do segundo vão em um único lote.
  lat_second_fwd_ms  o mesmo, só a chamada do modelo (entradas já prontas na GPU/CPU).
  lat_node_ms        lat_second_ms dividido pelo número de nós do grupo (por mensagem).
  lat_msg_single_ms  (só A) uma mensagem isolada, lote 1 (o "por mensagem, lote 1" do plano).
  lat_second_seq_ms  (só A) estimativa se o A fosse chamado mensagem a mensagem no segundo.
Memória (GPU): pico durante a inferência acima do que já estava alocado (ativações), em um
  segundo típico e no MAIOR grupo do Test (trafficCongestionSybil), e o tamanho dos parâmetros.

ORÇAMENTO PRÁTICO (fixado antes de medir): processar um segundo de tráfego de um receptor deve
levar no máximo BUDGET_MS = 100 ms (10% do intervalo de 1 s entre mensagens do mesmo emissor).

Uso (de dentro da pasta do cenário, com o ambiente .venv ativo):
    python src/qp3_cost.py                 # GPU e CPU, 30 repetições
    python src/qp3_cost.py --reps 3        # teste rápido
    python src/qp3_cost.py --no-cpu
"""
import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run as R
from analyze import paired
from graphdata import NodeTable, adjacency
from models import count_params
from models_c import GNNAutoencoder

FEAT = Path("data/features")
RESULTS = Path("results")
BUDGET_MS = 100.0
WARMUP = 30
MODELS = ["A", "Bp", "C", "B"]
LABEL = {"A": "A", "Bp": "B'", "C": "C", "B": "B"}


def make_model(kind, dev):
    """A, B' e B vêm do run.py (congelado); o C (GNN só espaço) vem de models_c.py."""
    if kind == "C":
        return GNNAutoencoder(12, R.HIDDEN, R.LATENT).to(dev)
    return R.make_model(kind, dev)


def timed(fn, dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def run_group(kind, model, tbl, g, dev):
    """Pontua um segundo inteiro de um receptor, de ponta a ponta."""
    if kind == "A":
        s, e = int(tbl.gstart[g]), int(tbl.gstart[g + 1])
        return model(tbl.X[s:e])
    x, pos, valid, nmask, idx = tbl.batch(torch.tensor([g], device=dev))
    adj = adjacency(pos, valid, R.RADIUS) if kind in ("B", "C") else None
    return model(x, adj, valid)


@torch.inference_mode()
def measure(kind, model, tbl, groups, dev):
    gt = torch.tensor(groups, device=dev)
    starts = tbl.gstart[gt].tolist()
    ends = tbl.gstart[gt + 1].tolist()
    sizes = np.array([e - s for s, e in zip(starts, ends)], dtype=float)
    out = {"nos_mediana": float(np.median(sizes))}

    if kind == "A":
        batches = [tbl.X[s:e] for s, e in zip(starts, ends)]
        singles = [tbl.X[s:s + 1] for s in starts]
        for b in batches[:WARMUP]:
            model(b)
        for b in singles[:WARMUP]:
            model(b)
        f_second = np.array([timed(lambda b=b: model(b), dev) for b in batches])
        f_single = np.array([timed(lambda b=b: model(b), dev) for b in singles])
        out.update(lat_second_ms=np.median(f_second), lat_second_fwd_ms=np.median(f_second),
                   lat_node_ms=np.median(f_second / sizes), lat_msg_single_ms=np.median(f_single),
                   lat_second_seq_ms=np.median(f_single * sizes))
        return out

    def build(g):
        x, pos, valid, nmask, idx = tbl.batch(torch.tensor([g], device=dev))
        adj = adjacency(pos, valid, R.RADIUS) if kind in ("B", "C") else None
        return x, adj, valid

    pre = [build(g) for g in groups]
    for p in pre[:WARMUP]:
        model(*p)
    for g in groups[:WARMUP]:
        model(*build(g))
    f_fwd = np.array([timed(lambda p=p: model(*p), dev) for p in pre])
    f_e2e = np.array([timed(lambda g=g: model(*build(g)), dev) for g in groups])
    out.update(lat_second_ms=np.median(f_e2e), lat_second_fwd_ms=np.median(f_fwd),
               lat_node_ms=np.median(f_e2e / sizes), lat_msg_single_ms=np.nan,
               lat_second_seq_ms=np.nan)
    return out


@torch.inference_mode()
def peak_memory_delta_mb(kind, model, tbl, groups, dev):
    """Pico de memória da GPU durante a inferência, acima do que já estava alocado."""
    for g in groups[:5]:
        run_group(kind, model, tbl, g, dev)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for g in groups:
        run_group(kind, model, tbl, g, dev)
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--groups", type=int, default=300)
    ap.add_argument("--cpu-threads", type=int, default=1)
    ap.add_argument("--no-cpu", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    devices = (["cuda"] if torch.cuda.is_available() else []) + ([] if args.no_cpu else ["cpu"])
    env = {"data": datetime.datetime.now().isoformat(timespec="seconds"), "torch": torch.__version__,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "threads_cpu": args.cpu_threads, "orcamento_ms": BUDGET_MS,
           "repeticoes": args.reps, "grupos_por_repeticao": args.groups}
    (RESULTS / "qp3_ambiente.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print("Ambiente:", env)

    rows, mem_rows = [], []
    for dev_name in devices:
        dev = torch.device(dev_name)
        if dev_name == "cpu":
            torch.set_num_threads(args.cpu_threads)
        tbl = NodeTable(FEAT / "baseline_Test.npz", dev, R.T, with_messages=False)
        print(f"\n--- {dev_name}: {args.reps} repetições x {args.groups} grupos ---", flush=True)
        for rep in range(1, args.reps + 1):
            rng = np.random.default_rng(rep)
            groups = rng.choice(tbl.n_groups, size=args.groups, replace=False).tolist()
            for kind in rng.permutation(MODELS).tolist():
                R.set_seed(rep)
                model = make_model(kind, dev).eval()
                m = measure(kind, model, tbl, groups, dev)
                m.update(device=dev_name, rep=rep, model=kind, params=count_params(model),
                         params_mb=sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6)
                rows.append(m)
            if rep % 5 == 0 or rep == args.reps:
                print(f"  repetição {rep}/{args.reps}", flush=True)

        if dev_name == "cuda":
            rng = np.random.default_rng(1)
            groups = rng.choice(tbl.n_groups, size=args.groups, replace=False).tolist()
            syb = NodeTable(FEAT / "trafficCongestionSybil_Test.npz", dev, R.T, with_messages=False)
            sizes = (syb.gstart[1:] - syb.gstart[:-1])
            g_max = int(torch.argmax(sizes).item())
            print(f"\nMaior grupo do Test (Sybil): {int(sizes[g_max])} nós", flush=True)
            for kind in MODELS:
                R.set_seed(1)
                model = make_model(kind, dev).eval()
                typ = peak_memory_delta_mb(kind, model, tbl, groups, dev)
                worst = peak_memory_delta_mb(kind, model, syb, [g_max] * 20, dev)
                mem_rows.append({"model": kind, "pico_tipico_mb": typ, "pico_pior_caso_mb": worst,
                                 "nos_pior_caso": int(sizes[g_max])})
            del syb

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "qp3_latencia_memoria.csv", index=False)
    mem = pd.DataFrame(mem_rows)
    if len(mem):
        mem.to_csv(RESULTS / "qp3_memoria.csv", index=False)

    # ---------- resumo ----------
    summary = []
    for dev_name in devices:
        sub = df[df.device == dev_name]
        for kind in MODELS:
            s = sub[sub.model == kind]
            summary.append({
                "dispositivo": dev_name, "modelo": LABEL[kind], "parametros": int(s.params.iloc[0]),
                "parametros_mb": round(s.params_mb.iloc[0], 3),
                "segundo_ms": s.lat_second_ms.median(), "segundo_iqr_ms": s.lat_second_ms.quantile(.75) - s.lat_second_ms.quantile(.25),
                "segundo_so_modelo_ms": s.lat_second_fwd_ms.median(), "por_no_ms": s.lat_node_ms.median(),
                "uma_mensagem_lote1_ms": s.lat_msg_single_ms.median(),
                "segundo_msg_a_msg_estimado_ms": s.lat_second_seq_ms.median(),
                "pct_do_orcamento": 100 * s.lat_second_ms.median() / BUDGET_MS,
            })
    summary = pd.DataFrame(summary)
    summary.to_csv(RESULTS / "qp3_resumo.csv", index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n=== Latência (mediana das repetições; ms) ===")
    print(summary.round(3).to_string(index=False))
    if len(mem):
        print("\n=== Memória da GPU na inferência (MB acima do já alocado) ===")
        print(mem.round(2).to_string(index=False))

    print("\n=== Comparações pareadas (d = primeiro - segundo, mesma repetição), latência de um segundo ===")
    out = []
    for dev_name in devices:
        w = df[df.device == dev_name].pivot(index="rep", columns="model", values="lat_second_ms")
        for a, b, alt in (("B", "A", "greater"), ("B", "Bp", "greater"), ("Bp", "A", "two-sided"),
                          ("C", "A", "two-sided"), ("B", "C", "greater"), ("C", "Bp", "two-sided")):
            r = paired(w[a], w[b], alt)
            r.update({"dispositivo": dev_name, "comparacao": f"{LABEL[a]} - {LABEL[b]}",
                      "razao_das_medianas": w[a].median() / w[b].median()})
            out.append(r)
    comp = pd.DataFrame(out)
    cols = ["dispositivo", "comparacao", "media_d", "ic95_inf", "ic95_sup", "dz", "razao_das_medianas", "teste", "p"]
    comp[cols].to_csv(RESULTS / "qp3_comparacoes.csv", index=False)
    print(comp[cols].round(4).to_string(index=False))

    runs = RESULTS / "runs_test.csv"
    if runs.exists():
        frames = [pd.read_csv(runs)]
        if (RESULTS / "runs_test_C.csv").exists():
            frames.append(pd.read_csv(RESULTS / "runs_test_C.csv"))
        tr = pd.concat(frames).groupby("model")[["train_seconds", "peak_gpu_mb"]].mean().round(1)
        print("\nCusto de TREINO (médias das 30 seeds, do experimento principal):")
        print(tr.to_string())
    print(f"\nOrçamento prático: {BUDGET_MS:.0f} ms por segundo de tráfego de um receptor.")
    print("Arquivos em results/: qp3_latencia_memoria.csv, qp3_resumo.csv, qp3_memoria.csv, "
          "qp3_comparacoes.csv, qp3_ambiente.json")


if __name__ == "__main__":
    main()
