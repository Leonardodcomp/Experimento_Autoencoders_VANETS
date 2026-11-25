"""Modelo C: autoencoder em grafo SÓ ESPAÇO (GNN sem tempo).

Completa o desenho 2 x 2 do experimento (nenhum dos arquivos congelados foi alterado):

                      sem tempo      com tempo (GRU, T = 5 s)
    sem espaço        A  (MLP)       B' (STAutoencoder, spatial=False)
    com espaço        C  (este)      B  (STAutoencoder, spatial=True)

O C é o B sem a GRU: usa SÓ o segundo-alvo (o último dos T passos) e as mesmas camadas de
grafo do B (caminho próprio + média dos vizinhos a até 100 m), o mesmo oculto (32) e o
mesmo latente (6). Cada nó vê o próprio sinal e o dos vizinhos NAQUELE segundo, mas não o
histórico. O B' é o B sem o caminho dos vizinhos; o C é o B sem o tempo.

Parâmetros: A = 6.226 | B' = 9.778 | C = 5.778 | B = 12.210 (C = B - os 6.432 da GRU).

Teste rápido (formas e passagem para trás), sem precisar dos dados:
    python src/models_c.py
"""
import torch
import torch.nn as nn

from models import GraphConv, count_params


class GNNAutoencoder(nn.Module):
    def __init__(self, n_feat=12, hidden=32, latent=6):
        super().__init__()
        self.gc1 = GraphConv(n_feat, hidden, spatial=True)
        self.gc2 = GraphConv(hidden, hidden, spatial=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.dec_in = nn.Linear(latent, hidden)
        self.dec_gc = GraphConv(hidden, hidden, spatial=True)
        self.dec_out = nn.Linear(hidden, n_feat)

    def forward(self, x, adj, valid):
        # x [B,T,N,F]; adj [B,T,N,N]; valid [B,T,N]. Só o último passo (segundo-alvo) é usado.
        xl, a = x[:, -1], adj[:, -1]
        h = torch.relu(self.gc1(xl, a))
        h = torch.relu(self.gc2(h, a))
        h = h * valid[:, -1].unsqueeze(-1).to(h.dtype)
        z = self.to_latent(h)                                # [B,N,latente]
        d = torch.relu(self.dec_in(z))
        d = torch.relu(self.dec_gc(d, a))
        return self.dec_out(d)                               # [B,N,F]


if __name__ == "__main__":
    from graphdata import adjacency
    from models import STAutoencoder

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    B, T, N, F = 4, 5, 20, 12
    pos = torch.rand(B, T, N, 2, device=dev) * 300 + torch.tensor([216000.0, 450000.0], device=dev)
    valid = torch.rand(B, T, N, device=dev) > 0.2
    valid[:, -1] = True
    x = torch.randn(B, T, N, F, device=dev) * valid.unsqueeze(-1)
    adj = adjacency(pos, valid, 100.0)

    c = GNNAutoencoder().to(dev)
    y = c(x, adj, valid)
    assert y.shape == (B, N, F), y.shape
    loss = (y - x[:, -1]).pow(2).mean()
    c.zero_grad()
    loss.backward()
    assert torch.isfinite(loss)
    print(f"OK: C: saída {tuple(y.shape)}, perda {loss.item():.3f}")

    with torch.no_grad():
        # os vizinhos devem mudar a saída (é um modelo espacial) ...
        d_nb = (c(x, adj, valid) - c(x, torch.zeros_like(adj), valid)).abs().max().item()
        # ... e os passos ANTERIORES ao último não devem mudar nada (não há tempo)
        x2 = x.clone()
        x2[:, :-1] = torch.randn_like(x2[:, :-1])
        d_time = (c(x, adj, valid) - c(x2, adj, valid)).abs().max().item()
    assert d_nb > 0, "os vizinhos deveriam mudar a saída do C"
    assert d_time == 0, "o C não deveria depender dos passos anteriores"
    print(f"OK: vizinhos mudam a saída (dif. máx. {d_nb:.4f}); o histórico não muda nada (dif. = {d_time})")

    n_b = count_params(STAutoencoder(spatial=True))
    n_c = count_params(c)
    print(f"Parâmetros: C = {n_c:,} | B = {n_b:,} | diferença (GRU) = {n_b - n_c:,}")
    assert n_c == 5778, n_c
    print(f"Dispositivo do teste: {dev}")
