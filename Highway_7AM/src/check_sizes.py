"""Maior grupo (receptor, segundo) em cada conjunto Test: propriedade dos dados, não dos modelos.

O número de mensagens de um grupo é um limite superior para o número de nós dele. Serve para
prever o uso de memória da GPU na avaliação (a adjacência cresce com o quadrado dos nós).

Uso (de dentro da pasta do cenário, com o ambiente .venv ativo):
    python src/check_sizes.py
"""
from pathlib import Path

import numpy as np

NAMES = ["baseline", "constantPositionOffset", "constantSpeedOffset", "reversedHeading",
         "feignedBraking", "timeDelayAttack", "dosAttack", "trafficCongestionSybil", "dataReplay"]

for name in NAMES:
    path = Path("data/features") / f"{name}_Test.npz"
    if not path.exists():
        print(f"{name:26s} arquivo não encontrado")
        continue
    sizes = np.diff(np.load(path)["group_start"])
    print(f"{name:26s} grupos={len(sizes):7d}  mensagens por grupo: média={sizes.mean():6.1f}  "
          f"p99={np.quantile(sizes, 0.99):6.0f}  máx={sizes.max():5d}")
