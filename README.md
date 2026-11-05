# Previsão de Sucesso de Filmes com Ensembles (Bagging, Boosting e Stacking)

**PUC-SP · Ciência da Computação · Inteligência Artificial · LAB de Machine Learning com Scikit-Learn**

**Equipe:**
- Guilherme Schnekenberg Teixeira (RA00336189)
- Luís Augusto Coelho de Souza (RA00331675)

---

## Objetivo

Prever, **antes do lançamento**, se um filme será um **sucesso comercial**, usando três métodos de ensemble e comparando-os com uma árvore de decisão única.

Um filme é rotulado como **sucesso** quando a bilheteria mundial supera o orçamento de produção (`Worldwide_Gross > Production_Budget`). Caso contrário, é um **fracasso**. É um problema de classificação binária.

Este projeto continua o LAB anterior de Árvore de Decisão e reaproveita todo o EDA e as decisões de dados feitas nele.

Repositório do LAB anterior: https://github.com/Luis-coelho30/movie-success-prediction.git

## Dataset

Dataset **`movies`** do [Vega Datasets](https://github.com/vega/vega-datasets), carregado com:

```python
from vega_datasets import data
df = data.movies()
```

São 3.201 filmes originais. Depois da limpeza ficam **2.164 filmes (1996-2010)**, com 66,7% de sucessos.

### Features usadas (apenas informação pré-lançamento)

| Feature | Tipo |
|---|---|
| `Production_Budget` | numérica |
| `MPAA_Rating` | categórica |
| `Major_Genre` | categórica |
| `Source` | categórica |
| `Creative_Type` | categórica |
| `Distributor` | categórica |
| `release_season` (verão, feriados, outro) | categórica, derivada do mês |

### Principais decisões de dados
- Correção do erro de século em `Release_Date` (filmes clássicos datados 100 anos depois do real).
- Zeros em bilheteria ou orçamento tratados como ausentes; linhas sem rótulo removidas.
- Apenas filmes de **1996 em diante**, onde a cobertura de metadados é mais completa.
- Somente **casos completos** nas categóricas: "Unknown" seria um atalho que não existe para um filme novo.
- `Running_Time_min` e indicadores de ausência foram **excluídos**, porque refletem a qualidade do registro, não o filme.
- Categorias raras agrupadas.

## Metodologia

- **Split temporal:** treino até 22/04/2007 (1.731 filmes) e teste depois disso (433 filmes). O modelo aprende com o passado e é avaliado no futuro.
- **Validação expanding window:** para cada ano de 2002 a 2007, treina nos anos anteriores e valida naquele ano. Todos os hiperparâmetros foram escolhidos assim, sem tocar no teste.
- **Pipelines do Scikit-Learn:** one-hot encoding com agrupamento de níveis raros, ajustado dentro de cada fold, sem vazamento de dados.
- **Métricas:** **AUC** e **balanced accuracy**. A acurácia sozinha engana, porque prever "sucesso" para todos os filmes já acerta 70,4% do teste.
- **Bootstrap pareado** (2.000 reamostragens) para testar se as diferenças de AUC são reais.

## Modelos

| Modelo | Classe Scikit-Learn | Configuração escolhida |
|---|---|---|
| Referência | `DecisionTreeClassifier` | depth 4, leaf 40 |
| **Bagging** | `BaggingClassifier` | 200 árvores profundas (sem limite de depth, leaf 5) |
| **Boosting** | `HistGradientBoostingClassifier` | lr 0,03, depth 3, leaf 20, L2 = 1,0 |
| **Stacking** | `StackingClassifier` | base: árvore + bagging + boosting + regressão logística; meta-modelo: regressão logística |

## Resultados

### Validação (expanding window, apenas treino)

| Modelo | CV AUC |
|---|---|
| Árvore única | 0,567 ± 0,019 |
| Bagging | **0,602** ± 0,045 |
| Boosting | 0,594 ± 0,027 |
| Stacking | 0,591 ± 0,041 |

### Teste (433 filmes, abr/2007-2010)

| Modelo | AUC | Balanced acc. (limiar 0,657) | F1 |
|---|---|---|---|
| Majority dummy | 0,500 | 0,500 | 0,827 |
| Árvore única | 0,632 | 0,566 | 0,540 |
| Bagging | 0,577 | 0,564 | 0,542 |
| **Boosting** | **0,637** | 0,607 | 0,585 |
| Stacking | 0,630 | **0,611** | **0,599** |

Pelo bootstrap pareado, **nenhum ensemble difere significativamente da árvore única em AUC** (todos os intervalos de 95% contêm zero).

## Conclusões

- Na validação, os três ensembles superaram a árvore por cerca de 0,03 de AUC, mas o ganho **não se confirmou no teste**.
- O **bagging** se comportou como a teoria prevê: precisa de árvores profundas, e o AUC sobe de 1 para 10 árvores, com redução de variância. Mesmo assim, foi o pior no teste. Reduzir variância não cria sinal quando o sinal é fraco e a distribuição muda no tempo.
- **Boosting e stacking** produzem probabilidades mais bem calibradas. No limiar ajustado, a balanced accuracy sobe de 0,566 para cerca de 0,61.
- **Modelo recomendado:** o **boosting**, por ser o mais consistente e mais simples que o stacking com desempenho equivalente.
- O sucesso de um filme depende muito de fatores fora dos dados (elenco, marketing, críticas, concorrência), o que limita qualquer modelo.

## Limitações
- Um único corte temporal com 433 filmes de teste: o AUC tem cerca de ±0,05 de incerteza.
- Os dados terminam em 2010, e orçamentos e bilheterias estão em dólares nominais.
- Os resultados valem apenas para filmes com metadados completos.

## Como executar

```bash
pip install numpy pandas scikit-learn vega_datasets
jupyter notebook
```

Abra o notebook e execute todas as células em ordem. É preciso **scikit-learn ≥ 1.2**, e a execução completa leva cerca de 30 segundos.

## Estrutura

```
├── movie-success-prediction.ipynb   # LAB anterior: EDA + Árvore de Decisão
├── ensembles-movies.ipynb           # Este LAB: Bagging, Boosting e Stacking
└── README.md
```

## Referências
- [Scikit-Learn: Ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [GeeksforGeeks: Bagging vs Boosting vs Stacking](https://www.geeksforgeeks.org/machine-learning/bagging-vs-boosting-vs-stacking/)
- [Vega Datasets](https://github.com/vega/vega-datasets)
