# FlyWire Complex Networks

Repositório do projeto desenvolvido para a disciplina **MO438A — Redes Complexas**, do Instituto de Computação da Unicamp.

O projeto utiliza o conectoma da *Drosophila melanogaster* disponibilizado pelo **FlyWire**, representado como uma rede complexa de neurônios e conexões sinápticas.

## Projeto

**A biologia está na rede? Um estudo da identidade neuronal a partir do conectoma de Drosophila melanogaster.**

O projeto investiga a organização topológica do conectoma cerebral de Drosophila melanogaster e avalia em que medida informações derivadas da conectividade acrescentam capacidade preditiva à identificação biológica dos neurônios.

O conectoma utilizado foi disponibilizado pelo FlyWire e é representado como um grafo dirigido e ponderado, no qual os neurônios correspondem aos nós, as conexões sinápticas às arestas e o peso de cada aresta ao número de sinapses entre dois neurônios.

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
│   ├── f1/
│   │   ├── figures/      # Figuras da caracterização inicial
│   │   └── tables/       # Tabelas da caracterização inicial
│   │
│   └── f2/               # Features, modelos, auditorias e resultados da etapa final
│
├── scripts/          # Scripts de auditoria, construção e análise da rede
│
├── .gitattributes
├── .gitignore
└── README.MD
```

## Análises Realizadas

O projeto inclui:

- construção e caracterização estrutural do conectoma;

- análise de grau, força, densidade, reciprocidade, componentes conexas, clusterização, distâncias e diâmetro;

- comparação com modelos nulos que preservam a sequência de graus;

- análise de organização modular por Leiden e Infomap;

- cálculo de métricas de centralidade e posição modular;

- investigação da relação entre topologia e classificação biológica dos neurônios;

- análise de assortatividade e padrões de mistura entre superclasses;

- construção de atributos topológicos para tarefas de classificação;

- comparação entre atributos biológicos não topológicos (BIO), atributos derivados da rede (REDE) e sua combinação (BIO + REDE);

- avaliação com Regressão Logística e Random Forest;

- controles de robustez envolvendo flow, rewiring com preservação de grau, localização anatômica por neurópilo e transferência entre hemisférios.

A principal métrica preditiva utilizada foi o macro-F1, devido ao forte desbalanceamento entre as superclasses neuronais.

## Dados

Os dados originais são provenientes do projeto **FlyWire**:

- https://flywire.ai/
- https://codex.flywire.ai/

Os arquivos originais e parte dos dados processados não são versionados neste repositório devido ao tamanho dos arquivos e às condições de distribuição dos dados. O diretório graphml/ contém a representação da rede utilizada nas análises do projeto.

## Organização dos scripts

Os scripts estão organizados de acordo com as etapas do projeto.

- f1_*: construção da rede e caracterização estrutural inicial;

- f2_*: caracterização topológica avançada, comunidades, construção de features, modelagem preditiva e controles de robustez.

Os arquivos de saída gerados pelos scripts são armazenados principalmente em reports/f1/ e reports/f2/.

## Autor

**Eduardo A. Marras de Souza**  - 300638
MO438A — Redes Complexas  
Instituto de Computação — Unicamp
