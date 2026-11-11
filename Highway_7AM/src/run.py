"""Treina e avalia os modelos A, B' (Bp) e B no Highway 7 AM.

PROTOCOLO (congelado):
  - os três modelos recebem os MESMOS 12 atributos por mensagem/nó (11 da mensagem + a
    contagem de mensagens do emissor no segundo); só muda o tipo de rede:
      A  = autoencoder denso (uma mensagem isolada)
      Bp = B' = STGNN sem o caminho dos vizinhos (só tempo, T = 5 s)
      B  = STGNN completo (tempo + vizinhos a até 100 m)
  - treino: Train do baseline (sem ataques), parada antecipada pelo erro no Validation do baseline;
  - limiar: percentil 95 das pontuações dos nós do Validation do baseline (todos normais);
  - unidade de avaliação: o NÓ (emissor, segundo, receptor); rótulo = 1 se alguma mensagem do nó
    é ataque. No modelo A, a pontuação do nó é o MAIOR erro entre as mensagens dele;
  - métricas: AUC-ROC (principal, não depende do limiar), F1 no limiar, taxa de falsos alarmes e
    erro de reconstrução (MSE nos 12 atributos) nos nós normais do baseline.

Uso isolado (piloto, seed 0, só Validation):
    python src/run.py --model A --seed 0 --split Validation --attacks constantPositionOffset dosAttack
O experimento completo (seeds 1 a 30, Test) é rodado por run_all.py.
"""
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from graphdata import NodeTable, adjacency
from models import MLPAutoencoder, STAutoencoder, count_params

FEAT = Path("data/features")
RESULTS = Path("results")

T = 5
RADIUS = 100.0
LATENT = 6
HIDDEN = 32
LR = 1e-3
BS_MSG = 4096        # mensagens por lote (modelo A)
BS_GRP = 32          # grupos (receptor, segundo) por lote (B e B')
BS_EVAL = 64
MAX_EPOCHS = 100
PATIENCE = 5
PERCENTILE = 0.95


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(kind, dev):
    if kind == "A":
        return MLPAutoencoder(12, LATENT).to(dev)
    return STAutoencoder(12, HIDDEN, LATENT, spatial=(kind == "B")).to(dev)


def load_table(name, split, dev, kind):
    return NodeTable(FEAT / f"{name}_{split}.npz", dev, T, with_messages=(kind == "A"))


@torch.no_grad()
def node_scores(model, kind, tbl):
    """Pontuação por nó e erro de reconstrução médio (MSE nos 12 atributos)."""
    model.eval()
    dev = tbl.device
    if kind == "A":
        errs = []
        for i in range(0, tbl.Xm.shape[0], 65536):
            x = tbl.Xm[i:i + 65536]
            errs.append((model(x) - x).pow(2).mean(1))
        e = torch.cat(errs)
        score = torch.zeros(tbl.n_nodes, device=dev).scatter_reduce(
            0, tbl.msg_node, e, reduce="amax", include_self=False)
        return score, e.mean().item()

    score = torch.zeros(tbl.n_nodes, device=dev)
    total = torch.zeros((), device=dev)
    n = 0
    for i in range(0, tbl.n_groups, BS_EVAL):
        g = torch.arange(i, min(i + BS_EVAL, tbl.n_groups), device=dev)
        x, pos, valid, nmask, idx = tbl.batch(g)
        adj = adjacency(pos, valid, RADIUS) if kind == "B" else None
        e = (model(x, adj, valid) - x[:, -1]).pow(2).mean(-1)
        score[idx[nmask]] = e[nmask]
        total += e[nmask].sum()
        n += int(nmask.sum().item())
    return score, total.item() / n


def train(model, kind, tr, va, seed, dev, max_epochs, verbose=True):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    best, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    epochs_run = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = torch.zeros((), device=dev)
        steps = 0
        if kind == "A":
            perm = torch.randperm(tr.Xm.shape[0], device=dev, generator=gen)
            for i in range(0, len(perm), BS_MSG):
                x = tr.Xm[perm[i:i + BS_MSG]]
                loss = (model(x) - x).pow(2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.detach()
                steps += 1
        else:
            perm = torch.randperm(tr.n_groups, device=dev, generator=gen)
            for i in range(0, len(perm), BS_GRP):
                g = perm[i:i + BS_GRP]
                x, pos, valid, nmask, idx = tr.batch(g)
                adj = adjacency(pos, valid, RADIUS) if kind == "B" else None
                err = (model(x, adj, valid) - x[:, -1]).pow(2).mean(-1)
                loss = err[nmask].mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += loss.detach()
                steps += 1
        _, val_mse = node_scores(model, kind, va)
        epochs_run = epoch
        if verbose:
            print(f"  época {epoch:3d}  perda treino={total.item() / steps:.5f}  "
                  f"erro validação={val_mse:.5f}  ({time.time() - t0:.0f} s)", flush=True)
        if val_mse < best - 1e-7:
            best, best_state, bad = val_mse, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                if verbose:
                    print("  parada antecipada")
                break
    model.load_state_dict(best_state)
    return epochs_run, time.time() - t0


def attack_metrics(score, label, thr):
    m = label >= 0
    y = label[m].cpu().numpy()
    s = score[m].cpu().numpy()
    auc = float(roc_auc_score(y, s))
    pred = s > thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    den = 2 * tp + fp + fn
    f1 = 2 * tp / den if den else 0.0
    return auc, f1, tp, fp, fn, tn


def run_one(kind, seed, attacks, split, max_epochs, dev, verbose=True):
    """Treina um modelo com uma seed e o avalia nos ataques. Retorna uma lista de linhas."""
    set_seed(seed)
    tr = load_table("baseline", "Train", dev, kind)
    va = load_table("baseline", "Validation", dev, kind)
    model = make_model(kind, dev)
    n_params = count_params(model)
    if verbose:
        print(f"Modelo {kind} | seed {seed} | {dev} | avaliação em {split} | parâmetros: {n_params:,}")

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    epochs_run, train_s = train(model, kind, tr, va, seed, dev, max_epochs, verbose)
    peak_mb = torch.cuda.max_memory_allocated() / 1e6 if dev.type == "cuda" else float("nan")

    sc_va, _ = node_scores(model, kind, va)
    thr = torch.quantile(sc_va, PERCENTILE).item()
    base = va if split == "Validation" else load_table("baseline", split, dev, kind)
    _, mse_normal = node_scores(model, kind, base)
    if verbose:
        print(f"Limiar (percentil {int(PERCENTILE * 100)}): {thr:.5f} | "
              f"MSE nos nós normais do baseline/{split}: {mse_normal:.5f}")
    del tr
    if base is not va:
        del base

    t_eval = time.time()
    rows = []
    for name in attacks:
        tbl = load_table(name, split, dev, kind)
        score, _ = node_scores(model, kind, tbl)
        auc, f1, tp, fp, fn, tn = attack_metrics(score, tbl.label, thr)
        fpr = fp / (fp + tn) if (fp + tn) else float("nan")
        if verbose:
            print(f"  {name:26s} AUC-ROC={auc:.4f}  F1={f1:.4f}  falsos alarmes={100 * fpr:.1f}%  "
                  f"(TP={tp} FP={fp} FN={fn})")
        rows.append({
            "model": kind, "seed": seed, "attack": name, "split": split,
            "auc": auc, "f1": f1, "fpr": fpr, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "threshold": thr, "mse_normal_baseline": mse_normal, "params": n_params,
            "epochs": epochs_run, "train_seconds": round(train_s, 1), "peak_gpu_mb": round(peak_mb, 1),
        })
        del tbl, score
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    eval_s = round(time.time() - t_eval, 1)
    for r in rows:
        r["eval_seconds"] = eval_s
    if verbose:
        print(f"Treino: {train_s:.0f} s em {epochs_run} épocas | avaliação: {eval_s:.0f} s | "
              f"pico de memória da GPU: {peak_mb:.0f} MB")
    del model, va
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["A", "Bp", "B"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attacks", nargs="+", default=["constantPositionOffset"])
    ap.add_argument("--split", default="Validation", choices=["Validation", "Test"])
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = run_one(args.model, args.seed, args.attacks, args.split, args.epochs, dev)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "runs.csv"
    pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
    print(f"Resultados acrescentados em {out}")


if __name__ == "__main__":
    main()
