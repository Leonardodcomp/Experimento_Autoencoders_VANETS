"""Transforma as mensagens (data/features/*.npz) em NÓS e liga cada nó ao seu histórico.

Definições (decididas no projeto):
  - grupo   = um receptor em um segundo (os emissores que ele ouviu naquele segundo);
  - nó      = um emissor (sender_id) dentro de um grupo. Se o emissor mandou várias
              mensagens no mesmo segundo, o nó usa a ÚLTIMA e guarda a contagem;
  - rótulo  = do nó: 1 se alguma mensagem dele naquele segundo é ataque, 0 se todas são
              benignas e -1 se não há rótulo (baseline);
  - atributos do nó = os 11 atributos da mensagem + 1 de contagem: log(1 + (n - 1)),
              que vale 0 quando o emissor mandou uma só mensagem no segundo;
  - histórico: prev[k, i] é o índice do nó do MESMO emissor, no MESMO receptor, k segundos
              antes (k = 0 é o próprio nó), ou -1 se ele não apareceu naquele segundo.

Este módulo usa só NumPy. Os resultados são salvos em cache em data/nodes/.
"""
from pathlib import Path

import numpy as np

CACHE_DIR = Path("data/nodes")


def build_nodes(X_msg, receiver, sender, t, attacker, pos, group_start, T=5):
    n_msg = len(X_msg)
    G = len(group_start) - 1

    group_of_msg = np.repeat(np.arange(G, dtype=np.int64), np.diff(group_start))
    first = group_start[:-1]
    sec_of_group = np.floor(t[first]).astype(np.int64)
    recv_of_group = receiver[first].astype(np.int64)
    S = int(sender.max()) + 1

    # ordena as mensagens por (grupo, emissor); a ordem estável mantém a ordem no tempo
    key = group_of_msg * S + sender.astype(np.int64)
    order = np.argsort(key, kind="stable")
    ks = key[order]
    new_run = np.ones(n_msg, dtype=bool)
    new_run[1:] = ks[1:] != ks[:-1]
    run_start = np.flatnonzero(new_run)
    run_end = np.append(run_start[1:], n_msg)
    count = (run_end - run_start).astype(np.int64)
    n_nodes = len(run_start)
    last = order[run_end - 1]                       # última mensagem de cada nó

    msg_node = np.empty(n_msg, dtype=np.int64)
    msg_node[order] = np.repeat(np.arange(n_nodes, dtype=np.int64), count)

    label = np.maximum.reduceat(attacker[order].astype(np.int16), run_start).astype(np.int8)

    count_feat = np.log1p(count - 1).astype(np.float32)[:, None]
    X_nodes = np.concatenate([X_msg[last], count_feat], axis=1).astype(np.float32)
    pos_nodes = pos[last].astype(np.float32)

    node_group = group_of_msg[last]
    node_sender = sender[last].astype(np.int64)
    group_node_start = np.searchsorted(node_group, np.arange(G + 1)).astype(np.int64)

    # histórico: a chave de um nó é (receptor, segundo, emissor); voltar k segundos
    # equivale a subtrair k * S da chave (as chaves estão em ordem crescente)
    sec_span = int(sec_of_group.max()) + T + 2
    node_key = (recv_of_group[node_group] * sec_span + sec_of_group[node_group]) * S + node_sender
    prev = np.empty((T, n_nodes), dtype=np.int64)
    prev[0] = np.arange(n_nodes)
    for k in range(1, T):
        q = node_key - k * S
        idx = np.searchsorted(node_key, q)
        idx_c = np.minimum(idx, n_nodes - 1)
        found = (idx < n_nodes) & (node_key[idx_c] == q)
        prev[k] = np.where(found, idx_c, -1)

    return {
        "X": X_nodes, "pos": pos_nodes, "label": label, "count": count,
        "prev": prev.astype(np.int32), "msg_node": msg_node.astype(np.int64),
        "group_node_start": group_node_start,
    }


def load_or_build(npz_path, T=5, use_cache=True):
    """Lê data/features/<nome>.npz e devolve os nós (com cache em data/nodes/)."""
    npz_path = Path(npz_path)
    cache = CACHE_DIR / f"{npz_path.stem}_T{T}.npz"
    if use_cache and cache.exists():
        d = np.load(cache)
        return {k: d[k] for k in d.files}
    d = np.load(npz_path)
    out = build_nodes(d["X"], d["receiver"], d["sender"], d["t"], d["attacker"],
                      d["pos"], d["group_start"], T)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **out)
    return out
