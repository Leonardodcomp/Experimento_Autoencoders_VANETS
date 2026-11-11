"""Modelos do experimento.

  A   MLPAutoencoder  uma mensagem isolada (12 atributos: os 11 da mensagem + a contagem de
                      mensagens do emissor no segundo) -> reconstrói a mensagem.
  B   STAutoencoder   grafo de cada segundo + histórico de T = 5 s -> reconstrói os
                      atributos (12) dos nós do segundo-alvo. Cada nó combina o próprio
                      sinal com a MÉDIA dos vizinhos a até 100 m.
  B'  STAutoencoder   o mesmo modelo SEM o caminho dos vizinhos (spatial=False): cada nó
                      vê apenas o próprio histórico (só tempo).

Em cada camada de grafo há dois caminhos: um do próprio nó (W_self) e outro da média dos
vizinhos (W_nb). O B' é o B sem o caminho dos vizinhos; logo, o B contém o B' e pode
ignorar os vizinhos se eles não ajudarem. (Uma convolução que mistura o próprio nó na
média dos vizinhos dilui o sinal dele e impede a reconstrução, como o piloto mostrou.)

O espaço latente tem 6 dimensões, MENOR que a entrada (12), senão o autoencoder
aprenderia a copiar a entrada e reconstruiria bem até as anomalias.

Teste rápido (formas, passagem para trás e adjacência), sem precisar dos dados:
    python src/models.py
"""
import torch
import torch.nn as nn


class MLPAutoencoder(nn.Module):
    def __init__(self, n_feat=12, latent=6):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_feat, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, latent),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, n_feat),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


class GraphConv(nn.Module):
    """Caminho do próprio nó + (opcional) caminho da média dos vizinhos."""

    def __init__(self, n_in, n_out, spatial=True):
        super().__init__()
        self.self_lin = nn.Linear(n_in, n_out)
        self.nb_lin = nn.Linear(n_in, n_out, bias=False) if spatial else None

    def forward(self, x, adj=None):
        out = self.self_lin(x)
        if self.nb_lin is not None and adj is not None:
            out = out + adj @ self.nb_lin(x)
        return out


class STAutoencoder(nn.Module):
    """Codificador: camadas de grafo (espaço) em cada passo + GRU (tempo) por nó.
    Decodificador: reconstrói os atributos dos nós do segundo-alvo."""

    def __init__(self, n_feat=12, hidden=32, latent=6, spatial=True):
        super().__init__()
        self.gc1 = GraphConv(n_feat, hidden, spatial)
        self.gc2 = GraphConv(hidden, hidden, spatial)
        self.gru = nn.GRU(hidden + 1, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.dec_in = nn.Linear(latent, hidden)
        self.dec_gc = GraphConv(hidden, hidden, spatial)
        self.dec_out = nn.Linear(hidden, n_feat)

    def forward(self, x, adj, valid):
        # x [B,T,N,F]; adj [B,T,N,N] ou None (B'); valid [B,T,N]
        h = torch.relu(self.gc1(x, adj))
        h = torch.relu(self.gc2(h, adj))
        v = valid.unsqueeze(-1).to(h.dtype)                  # [B,T,N,1]
        h = torch.cat([h * v, v], dim=-1)                    # o GRU sabe quando o nó faltou
        B, T, N, H1 = h.shape
        seq = h.permute(0, 2, 1, 3).reshape(B * N, T, H1)    # uma sequência por nó
        _, hn = self.gru(seq)
        z = self.to_latent(hn[0]).reshape(B, N, -1)          # [B,N,latente]
        d = torch.relu(self.dec_in(z))
        d = torch.relu(self.dec_gc(d, None if adj is None else adj[:, -1]))
        return self.dec_out(d)                               # [B,N,F]


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from graphdata import adjacency

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    B, T, N, F = 4, 5, 20, 12

    # raio de 100 m com coordenadas absolutas grandes, como nos dados reais
    p2 = torch.zeros(1, 1, 2, 2, device=dev)
    p2[0, 0, 0] = torch.tensor([216000.0, 450000.0])
    p2[0, 0, 1] = torch.tensor([216050.0, 450000.0])
    v2 = torch.ones(1, 1, 2, dtype=torch.bool, device=dev)
    assert adjacency(p2, v2, 100.0)[0, 0, 0, 1] == 1.0, "dois pontos a 50 m deveriam ser vizinhos"
    p2[0, 0, 1] = torch.tensor([216150.0, 450000.0])
    assert adjacency(p2, v2, 100.0)[0, 0, 0, 1] == 0.0, "dois pontos a 150 m não deveriam ser vizinhos"
    print("OK: raio de 100 m correto, mesmo com coordenadas absolutas grandes")

    pos = torch.rand(B, T, N, 2, device=dev) * 300 + torch.tensor([216000.0, 450000.0], device=dev)
    valid = torch.rand(B, T, N, device=dev) > 0.2
    valid[:, -1] = True
    x = torch.randn(B, T, N, F, device=dev) * valid.unsqueeze(-1)
    adj = adjacency(pos, valid, 100.0)
    rs = adj.sum(-1)
    assert (((rs - 1).abs() < 1e-5) | (rs == 0)).all(), "cada linha deve somar 1 (ou 0, se isolado)"
    assert torch.diagonal(adj, dim1=-2, dim2=-1).abs().max() == 0, "não deve haver autoligação"
    n_nb = (adj > 0).float().sum().item()
    print(f"OK: linhas somam 1 ou 0, sem autoligação; pares vizinhos neste lote de teste: {int(n_nb)}")

    a_model = MLPAutoencoder().to(dev)
    b_model = STAutoencoder(spatial=True).to(dev)
    bp_model = STAutoencoder(spatial=False).to(dev)
    xa = torch.randn(32, 12, device=dev)
    assert a_model(xa).shape == xa.shape
    for name, m, a in (("B (com vizinhos)", b_model, adj), ("B' (sem vizinhos)", bp_model, None)):
        y = m(x, a, valid)
        assert y.shape == (B, N, F), y.shape
        loss = (y - x[:, -1]).pow(2).mean()
        m.zero_grad()
        loss.backward()
        assert torch.isfinite(loss)
        print(f"OK: {name}: saída {tuple(y.shape)}, perda {loss.item():.3f}")

    with torch.no_grad():
        diff = (b_model(x, adj, valid) - b_model(x, torch.zeros_like(adj), valid)).abs().max().item()
    assert diff > 0, "os vizinhos deveriam mudar a saída do B"
    print(f"OK: no B, os vizinhos mudam a saída (diferença máxima {diff:.4f})")

    print(f"\nParâmetros: A = {count_params(a_model):,} | B' = {count_params(bp_model):,} | B = {count_params(b_model):,}")
    print(f"Dispositivo do teste: {dev}")
