# FlyWire Complex Networks

Repositório do projeto desenvolvido para a disciplina **MO438A — Redes Complexas**, do Instituto de Computação da Unicamp.

O projeto utiliza o conectoma da *Drosophila melanogaster* disponibilizado pelo **FlyWire**, representado como uma rede complexa de neurônios e conexões sinápticas.

## Projeto

**A biologia está na rede? Um estudo da identidade neuronal a partir do conectoma de Drosophila melanogaster.**

Este repositório reúne os dados, scripts e resultados utilizados no desenvolvimento do projeto.

## Estrutura do repositório

```text
.
├── data/
│   ├── processed/    # Dados processados e versão intermediária da rede (.gitignore)
│   └── raw/          # Dados originais (.gitignore)
│
├── graphml/          # Rede utilizada no projeto em formato GraphML
│
├── reports/
│   └── f1/
│       ├── figures/  # Figuras e visualização da rede
│       └── tables/   # Tabelas geradas na análise
│
├── scripts/          # Scripts de auditoria, construção e análise da rede
│
├── .gitattributes
├── .gitignore
└── README.MD
```

## Entrega Parcial — F1

A primeira etapa do projeto consiste na caracterização inicial da rede complexa utilizada no restante do trabalho.

O repositório contém:

- a rede utilizada no projeto em formato **GraphML**;
- os scripts utilizados na construção e análise da rede;
- as métricas estruturais solicitadas na Entrega Parcial;
- as figuras geradas durante a análise;
- uma visualização complementar da rede.

## Dados

Os dados originais são provenientes do projeto **FlyWire**:

- https://flywire.ai/
- https://codex.flywire.ai/

## Autor

**Eduardo A. Marras de Souza**  - 300638
MO438A — Redes Complexas  
Instituto de Computação — Unicamp
