# Experimento_Autoencoders_VANETS

# Espaço e tempo em autoencoders para detecção de anomalias em redes veiculares

Experimento computacional (*in silico*) que compara quatro autoencoders em um desenho 2×2 (espaço × tempo) para detecção de mensagens anômalas no **VeReMi NextGen**. Material de apoio do artigo *"Espaço e tempo em autoencoders para detecção de anomalias em redes veiculares: um experimento in silico no VeReMi NextGen"* (Leonardo Souza Silva, PPGCC/UFS, disciplina Engenharia de Software Experimental 2026.2).

## Pergunta de pesquisa

Modelar a vizinhança espacial (grafo) e a sequência temporal (GRU) de cada veículo melhora um autoencoder que, no caso base, olha cada mensagem isoladamente?

| Modelo | Espaço | Tempo | Descrição | Parâmetros |
|---|---|---|---|---|
| **A** | não | não | MLP por nó | 6.226 |
| **B'** | não | sim | GRU sobre janela de 5 s, sem arestas | 9.778 |
| **C** | sim | não | GNN sobre o grafo do último instante, sem GRU | 5.778 |
| **B** | sim | sim | STGNN (GNN + GRU) | 12.210 |

> O modelo C foi incluído depois de ver os resultados de A, B' e B, para separar o efeito do espaço do efeito do tempo. Isso está declarado no artigo (Seção 6.1).

## Desenho experimental

- **Dataset:** VeReMi NextGen v1 (InTAS, rodovia, hora de pico), com Eclipse MOSAIC. Dados: <https://zenodo.org/records/19665762>; documentação: <https://veremi-dataset.github.io/veremi-nextgen>. O dataset **não** está neste repositório.
- **Unidade de avaliação:** nó (emissor, segundo, receptor). Grafo por (receptor, segundo), com arestas entre veículos a até 100 m; janela de T = 5 passos para B' e B.
- **Atributos:** os mesmos 12 para os quatro modelos.
- **Ataques (8):** constantPositionOffset, constantSpeedOffset, reversedHeading, feignedBraking, timeDelay, dosAttack, trafficCongestionSybil e dataReplay.
- **Treino:** só tráfego normal (Train); limiar = percentil 95 do erro de reconstrução no Validation normal; parada antecipada com paciência 5 e no máximo 100 épocas; lr = 1e-3, camada oculta 32, latente 6.
- **Amostra:** 30 sementes × 4 modelos = 120 execuções de treino; 960 avaliações modelo-semente-ataque.
- **Análise:** diferenças pareadas por semente (d = primeiro − segundo), Shapiro–Wilk → *t* pareado ou Wilcoxon, teste bilateral (H0: μd = 0), dz de Cohen, IC 95%, Friedman e correção de Holm (α = 0,05). Não houve teste de equivalência.
- **Hipóteses operacionais:** O1 B−A, O2 C−A, O3 B−B', O4 B'−A, O5 B−C e O6 (B−C)−(B'−A).
- **Análise de sensibilidade (exploratória, pós-hoc):** B e C retreinados com paciência 15 e máximo de 200 épocas, sementes 1–10.

## Principais resultados (Test, 30 sementes)

| Modelo | AUC-ROC | F1 | MSE nos nós normais | Treino médio (s) |
|---|---|---|---|---|
| A | 0,611 | 0,255 | 0,0364 | 69,6 |
| B' | 0,615 | 0,262 | 0,0388 | 388 |
| C | 0,637 | 0,324 | 0,0396 | 181,9 |
| B | 0,643 | 0,318 | 0,0296 | 335 |

- O **espaço** melhora a detecção (O1, O2 e O3 rejeitam H0), com e sem tempo.
- O **tempo**, como operacionalizado (GRU sobre 5 s), não acrescentou detecção (O4, O5 e O6 não rejeitam H0). Isso não prova equivalência.
- O custo de inferência fica abaixo de 2,5% do orçamento de 100 ms nos quatro modelos; o custo relevante está no treino.
- A vantagem dos modelos com grafo não é uniforme: eles ganham em Sybil, dataReplay, timeDelay e constantPositionOffset e perdem em constantSpeedOffset e feignedBraking.

Detalhes, tabelas por ataque, limitações e ameaças à validade estão no artigo.

## Ambiente

Todo o experimento rodou localmente em um notebook com Windows 11 Home Single Language, GPU NVIDIA GeForce RTX 3050 Laptop (4 GB de VRAM), CPU Intel Core i5-11260H e 16 GB de RAM.

| Pacote | Versão |
|---|---|
| Python | 3.15.0 |
| PyTorch | 2.14.1 (CUDA 13.2) |
| NumPy | 2.5.2 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| SciPy | 1.18.1 |
| matplotlib | 3.11.2 |

## Estrutura do repositório

> Ajuste esta seção para refletir as pastas reais do repositório.

| Arquivo | Função |
|---|---|
| `prepare_features.py` | Prepara os 12 atributos a partir do dataset |
| `nodes.py` | Construção dos nós (emissor, segundo, receptor) |
| `graphdata.py` | Construção dos grafos e das janelas temporais |
| `models.py` | Modelos A, B' e B |
| `models_c.py` | Modelo C (apenas espaço) |
| `run.py`, `run_all.py` | Treino e avaliação (protocolo principal) |
| `run_c.py` | Treino e avaliação do modelo C |
| `run_sensibilidade.py` | Análise de sensibilidade (paciência 15) |
| `analyze.py`, `analyze_2x2.py` | Estatística do desenho 2×2 |
| `analyze_sens.py` | Estatística da sensibilidade |
| `qp3_cost.py` | Latência e memória (QP3) |
| `runs_test.csv` | Resultados brutos no Test (modelo × semente × ataque) |

Os scripts do protocolo principal (`nodes.py`, `graphdata.py`, `models.py`, `run.py`, `run_all.py`, `analyze.py`) foram congelados antes da execução principal. `models_c.py`, `run_c.py` e `analyze_2x2.py` foram adicionados depois, com a inclusão do modelo C.

## Como reproduzir

> Complete os comandos conforme os argumentos reais dos scripts.

1. Baixe o VeReMi NextGen v1 do Zenodo e coloque os arquivos em uma pasta local (`<CAMINHO_DO_DATASET>`).
2. Crie o ambiente e instale as dependências nas versões acima.
3. Gere os atributos: `python prepare_features.py ...`
4. Rode o protocolo principal (30 sementes): `python run_all.py ...` e `python run_c.py ...`
5. Rode a análise estatística: `python analyze.py ...` e `python analyze_2x2.py ...`
6. (Opcional) Custo operacional: `python qp3_cost.py ...`; sensibilidade: `python run_sensibilidade.py ...` e `python analyze_sens.py ...`

## Limitações

- Um cenário simulado (rodovia de alta densidade), oito dos quinze ataques do dataset e um único hardware.
- O "tempo" testado é uma GRU sobre 5 s dentro de um autoencoder de reconstrução, não o tempo em geral.
- Parada antecipada não implica convergência; as curvas de validação por época não foram registradas.
- A análise de sensibilidade é exploratória (10 sementes, sem correção para múltiplos testes).

## Como citar

> Silva, L. S. (2026). *Espaço e tempo em autoencoders para detecção de anomalias em redes veiculares: um experimento in silico no VeReMi NextGen*. Programa de Pós-Graduação em Ciência da Computação, Universidade Federal de Sergipe.

Ao usar o dataset, cite também Hermann et al. (2026), *VeReMi NextGen*, IEEE VNC.

## Uso de IA generativa

O código e o texto foram elaborados com auxílio do Claude (Anthropic). As decisões de projeto, a execução dos experimentos e a responsabilidade pelos resultados são do autor. Detalhes na declaração do artigo.

## Licença

> Defina a licença do código (por exemplo, MIT) e adicione o arquivo `LICENSE`.

## Contato

Leonardo Souza Silva, PROCC/UFS.
