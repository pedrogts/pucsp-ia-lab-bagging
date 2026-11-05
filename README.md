# LAB – Ensemble Models: Bagging, Boosting e Stacking com Scikit-learn

Projeto de laboratório da disciplina de **Inteligência Artificial (IA ML – Aprendizagem de Máquina)** do curso de **Ciência da Computação da PUC-SP**.

## Autores (dupla)

- **Pedro Gabriel Takenobu Serafim**
- **Rubens Rodrigues Maranesi**

## Objetivo

Explorar os métodos de **Ensemble** do Scikit-learn (Bagging, Boosting e Stacking) e aplicá-los ao dataset **`cars`** da biblioteca `vega_datasets`, comparando um modelo único (baseline) com modelos de Bagging e Boosting em uma tarefa de classificação.

**Problema:** prever se um carro é dos **EUA** ou de **outra origem** (Europa ou Japão) a partir de suas características técnicas.

## O que foi feito

O notebook mantém todo o material original da aula (teoria sobre Ensemble, Bagging, Boosting e Stacking, enunciado do LAB e o exemplo de Stacking) e acrescenta a implementação do LAB:

1. **Exploração dos dados:** dimensões, valores ausentes e distribuição das classes.
2. **Pré-processamento:** criação do alvo binário `Is_USA`, seleção das variáveis numéricas e divisão treino/teste.
3. **Baseline:** uma Árvore de Decisão simples como referência.
4. **Bagging:** `BaggingClassifier` (com árvores de decisão) e `RandomForestClassifier`.
5. **Boosting:** `AdaBoostClassifier` e `GradientBoostingClassifier`.
6. **Avaliação no conjunto de teste** e **validação cruzada repetida (5-fold x 5 repetições)** para uma comparação mais confiável.
7. **Análise do melhor modelo:** relatório de classificação, matriz de confusão e importância das variáveis.
8. **Conclusões.**

## Dataset

| Item | Valor |
|---|---|
| Fonte | `vega_datasets` (`data.cars()`) |
| Tamanho | 406 carros, 9 colunas |
| Classes | USA: 254 · Japan: 79 · Europe: 73 |
| Alvo | `Is_USA` (1 = EUA, 0 = Europa/Japão), cerca de 62% de carros dos EUA |
| Variáveis | `Miles_per_Gallon`, `Cylinders`, `Displacement`, `Horsepower`, `Weight_in_lbs`, `Acceleration` |
| Valores ausentes | `Miles_per_Gallon` (8) e `Horsepower` (6) |

## Metodologia

- **Divisão dos dados:** 75% treino / 25% teste, estratificada pela classe (`random_state=42`).
- **Valores ausentes:** preenchidos com a **mediana** dentro de cada `Pipeline`, de modo que a mediana é calculada apenas com os dados de treino (evita *data leakage*).
- **Hiperparâmetros usados:**

| Modelo | Tipo | Principais parâmetros |
|---|---|---|
| Árvore de Decisão | Baseline | padrão |
| Bagging | Bagging | 100 árvores, bootstrap, `oob_score=True` |
| Random Forest | Bagging | 100 árvores |
| AdaBoost | Boosting | 200 estimadores, `learning_rate=0.5`, árvores de profundidade 2 |
| Gradient Boosting | Boosting | 200 estimadores, `learning_rate=0.05`, `max_depth=3`, `subsample=0.8` |

- **Validação cruzada:** `RepeatedStratifiedKFold` com 5 folds e 5 repetições (25 avaliações por modelo) sobre todo o dataset. Isso foi feito porque o conjunto de teste tem apenas 102 amostras, e a diferença de um ou dois carros já altera a acurácia em cerca de 1 ponto percentual.

## Resultados

### Validação cruzada (5x5)

| Modelo | Tipo | Acurácia (média ± desvio padrão) |
|---|---|---|
| Bagging (árvores) | Bagging | 0,919 ± 0,026 |
| Gradient Boosting | Boosting | 0,911 ± 0,027 |
| AdaBoost | Boosting | 0,904 ± 0,023 |
| Random Forest | Bagging | 0,903 ± 0,027 |
| Árvore de Decisão | Baseline | 0,890 ± 0,035 |

### Conjunto de teste (102 carros)

| Modelo | Acurácia |
|---|---|
| Bagging (árvores) | 0,951 |
| Gradient Boosting | 0,912 |
| AdaBoost | 0,912 |
| Árvore de Decisão (baseline) | 0,902 |
| Random Forest | 0,902 |

Acurácia *out-of-bag* do Bagging (estimada só com dados de treino): **0,898**.

Para comparação, o exemplo de Stacking fornecido na aula (Random Forest + SVM + Rede Neural, com Regressão Logística como meta-modelo) obteve **0,902** no mesmo conjunto de teste.

### Melhor modelo (Bagging) no conjunto de teste

| Classe | Precisão | Recall | F1-score | Suporte |
|---|---|---|---|---|
| Outro | 0,95 | 0,92 | 0,93 | 38 |
| EUA | 0,95 | 0,97 | 0,96 | 64 |

## Conclusões

1. **Todos os ensembles superaram, em média, o modelo único** (árvore de decisão), e o desvio padrão da acurácia caiu de 0,035 para cerca de 0,025, ou seja, os modelos ficaram mais estáveis.
2. **O Bagging de árvores foi o melhor modelo**, com 95,1% no teste contra 90,2% do baseline (5 carros a mais classificados corretamente em 102). A acurácia *out-of-bag* (89,8%) ficou abaixo da do teste, o que indica que o resultado do teste foi um pouco otimista.
3. **A diferença entre os ensembles é pequena**, menor que o desvio padrão da validação cruzada. Com apenas 406 carros, não é possível afirmar que um ensemble seja melhor que outro, apenas que todos tendem a ser melhores que uma árvore isolada.
4. **Por que o Boosting não ganhou?** Uma hipótese: o Boosting reduz o *viés*, e uma única árvore já separa razoavelmente bem as classes neste problema, então o maior ganho vem de reduzir a *variância*, que é o que o Bagging faz. Para confirmar seria necessário testar outros hiperparâmetros e outros datasets.
5. **Variáveis mais importantes:** `Displacement` (cilindrada) foi a mais relevante nos dois modelos analisados (Random Forest e Gradient Boosting), seguida de `Horsepower` e `Weight_in_lbs`; `Cylinders` teve a menor importância. Como cilindrada e número de cilindros são muito correlacionados, a importância pode ter se concentrado em uma delas.

**Limitações:** dataset pequeno e levemente desbalanceado; não foi feito ajuste de hiperparâmetros.

**Próximos passos possíveis:** ajuste de hiperparâmetros com `GridSearchCV`, teste do `HistGradientBoostingClassifier` e um Stacking combinando os modelos acima.

## Como executar

### Opção 1: Google Colab

1. Abra o arquivo `.ipynb` no [Google Colab](https://colab.research.google.com/) (Arquivo > Fazer upload de notebook).
2. Execute as células em ordem (Ambiente de execução > Executar tudo).

### Opção 2: Localmente

```bash
pip install pandas numpy scikit-learn matplotlib vega_datasets jupyter
jupyter notebook
```

Depois abra o notebook e execute todas as células. O dataset `cars` já vem embutido no pacote `vega_datasets`, então não é necessário baixar nenhum arquivo.

## Tecnologias

- Python 3
- Scikit-learn
- Pandas e NumPy
- Matplotlib
- vega_datasets

## Estrutura do repositório

```
.
├── Cópia_de_PUCSP_CS_AI_12_AI_ML_Baggin.ipynb   # notebook completo (teoria, exemplo e LAB)
└── README.md
```

## Referências

- [Scikit-learn – Ensemble methods](https://scikit-learn.org/stable/modules/ensemble.html)
- [BaggingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)
- [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
- [StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)
- [GeeksforGeeks – Bagging vs Boosting vs Stacking](https://www.geeksforgeeks.org/machine-learning/bagging-vs-boosting-vs-stacking/)
- [Vega Datasets](https://github.com/altair-viz/vega_datasets)
