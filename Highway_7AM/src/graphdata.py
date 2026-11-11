"""Tabela de nós na GPU, montagem de lotes de grafos e matriz de adjacência.

Um LOTE é um conjunto de grupos (receptor, segundo). Para cada grupo, os nós são os
emissores ouvidos naquele segundo, e o modelo recebe também os T-1 segundos anteriores
dos mesmos emissores. Os tensores de um lote têm o formato:

  x      [B, T, N, F]  atributos dos nós (F = 12); passo T-1 = segundo-alvo, 0 = o mais antigo
  pos    [B, T, N, 2]  posição informada do emissor em cada passo
  valid  [B, T, N]     o nó existe naquele passo (o emissor foi ouvido naquele segundo)
  nmask  [B, N]        o nó existe no segundo-alvo (False = preenchimento do lote)
  idx    [B, N]        índice global do nó (para gravar os erros de volta)

Tudo é montado por indexação direta na GPU, sem laços em Python.
"""
import numpy as np
import torch

from nodes import load_or_build


class NodeTable:
    def __init__(self, npz_path, device, T=5, with_messages=False):
        d = load_or_build(npz_path, T)
        self.T = T
        self.device = device
        self.X = torch.from_numpy(d["X"]).to(device)
        self.pos = torch.from_numpy(d["pos"]).to(device)
        self.label = torch.from_numpy(d["label"]).to(device)
        self.prev = torch.from_numpy(d["prev"].astype(np.int64)).to(device)   # [T, n_nodes]
        self.gstart = torch.from_numpy(d["group_node_start"]).to(device)
        self.n_nodes = self.X.shape[0]
        self.n_groups = self.gstart.shape[0] - 1
        self.Xm = None
        self.msg_node = None
        if with_messages:       # o modelo A trabalha com uma mensagem de cada vez
            m = np.load(npz_path)
            self.msg_node = torch.from_numpy(d["msg_node"]).to(device)
            xm = torch.from_numpy(m["X"]).to(device)
            # mesmos 12 atributos dos nós: o 12º é a contagem de mensagens do emissor no segundo
            self.Xm = torch.cat([xm, self.X[self.msg_node, 11:12]], dim=1)
            del xm

    def batch(self, groups):
        """groups: tensor long [B] com índices de grupos."""
        start = self.gstart[groups]
        cnt = self.gstart[groups + 1] - start
        nmax = int(cnt.max().item())
        ar = torch.arange(nmax, device=self.device)
        nmask = ar[None, :] < cnt[:, None]
        idx = torch.where(nmask, start[:, None] + ar[None, :], torch.zeros_like(nmask, dtype=torch.long))

        p = self.prev[:, idx]                              # [T, B, N]: nó do mesmo emissor k s antes
        valid = (p >= 0) & nmask[None]
        p = p.clamp(min=0)
        x = self.X[p] * valid.unsqueeze(-1)                # [T, B, N, F]
        pos = self.pos[p]                                  # [T, B, N, 2]

        # ordem cronológica: o último passo (T-1) é o segundo-alvo
        x = x.permute(1, 0, 2, 3).flip(1).contiguous()
        pos = pos.permute(1, 0, 2, 3).flip(1).contiguous()
        valid = valid.permute(1, 0, 2).flip(1).contiguous()
        return x, pos, valid, nmask, idx


def adjacency(pos, valid, radius):
    """Matriz de vizinhança [B, T, N, N] para agregação pela MÉDIA dos vizinhos.

    Dois nós são vizinhos se estão a até `radius` metros (posições informadas) e ambos
    existem naquele passo. Não há autoligação: o nó mantém o próprio sinal por um
    caminho separado no modelo. Cada linha soma 1 (nós com vizinhos) ou 0 (nó isolado).
    As diferenças de posição são calculadas diretamente (e não por produto de matrizes),
    porque as coordenadas absolutas são grandes e a outra forma perde precisão em float32.
    """
    N = pos.shape[2]
    eye = torch.eye(N, device=pos.device)
    diff = pos[:, :, :, None, :] - pos[:, :, None, :, :]
    close = diff.pow(2).sum(-1) <= radius ** 2
    both = valid[:, :, :, None] & valid[:, :, None, :]
    a = (close & both).float() * (1.0 - eye)
    deg = a.sum(-1, keepdim=True)
    return a / deg.clamp(min=1.0)
