# Telco Customer Churn
### Classificação de cancelamento de clientes com Python e scikit-learn

Projeto acadêmico de Machine Learning desenvolvido por **Eduardo Braz de Oliveira**, organizado em duas etapas complementares (AP1 e AP2) e disponibilizado como material de portfólio.

O trabalho investiga características associadas ao cancelamento de clientes de telecomunicações e compara quatro algoritmos de classificação, com foco no tratamento do desbalanceamento e na interpretação dos erros.

**[Abrir o notebook executado](ML_Eduardo_Braz.ipynb)** · **[Dependências](ambiente_reprodutibilidade/requirements.txt)** · **[Ambiente registrado](ambiente_reprodutibilidade/ambiente_execucao.json)**

## Objetivo

Classificar o churn registrado na base e avaliar como os modelos podem apoiar análises das áreas de retenção, CRM, marketing e atendimento.

A variável-alvo é `Churn`:

- **0:** cliente não cancelou;
- **1:** cliente cancelou.

O estudo diferencia identificação de cancelamentos e falsos alertas: ampliar o Recall pode aumentar o número de clientes abordados sem cancelamento observado.

## Dados

Base pública **Telco Customer Churn**, disponibilizada por BlastChar no Kaggle e baseada no conjunto de exemplo da IBM.

- **7.043 clientes** e **21 colunas originais**;
- **19 variáveis explicativas originais**, excluindo identificador e alvo;
- variáveis numéricas e categóricas;
- **26,54% de churn** e **73,46% de não churn**.

**Fonte:** [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

O notebook obtém o CSV com `kagglehub`. Os dados não estão incluídos neste repositório; a execução depende de acesso à fonte ou de cache local disponível. Os termos aplicáveis à base devem ser consultados na página de origem.

## Etapas do projeto

| Etapa | Implementação |
|---|---|
| Qualidade dos dados | Conversão de TotalCharges, tratamento de 11 ausências, investigação de duplicidades e outliers por IQR |
| Pré-processamento | Exclusão de customerID, conversão do alvo, imputação, One-Hot Encoding e StandardScaler |
| Engenharia de atributos | NumServices, TenureGroup, AutomaticPayment e AvgMonthlySpend |
| Exploração | Taxas de churn por categoria e por seis faixas nas numéricas de alta cardinalidade |
| Avaliação da engenharia | Comparação dos atributos originais com originais + derivados por validação cruzada |
| Separação | 80% treino e 20% teste, estratificação e random_state=42 |
| Modelagem | Regressão Logística, SGD Classifier, Árvore de Decisão e Floresta Aleatória |
| Desbalanceamento | Cada algoritmo com class_weight="balanced" e um controle com class_weight=None |
| Otimização | GridSearchCV com dois hiperparâmetros por algoritmo e cinco divisões estratificadas |
| Seleção | Maior Balanced Accuracy média na validação cruzada |
| Interpretação | Métricas convencionais e balanceadas, matrizes de confusão e análise dos atributos |

As quatro melhorias da segunda etapa estão identificadas no índice do notebook:

1. **Atividade 1:** análise exploratória de churn por categorias e faixas numéricas.
2. **Atividade 2:** tratamento do desbalanceamento com `class_weight`.
3. **Atividade 3:** GridSearchCV e avaliação de sobreajuste.
4. **Atividade 4:** comparação de métricas e seleção por métrica balanceada.

Imputação, codificação e escalonamento dos modelos são ajustados dentro dos pipelines nas divisões da validação cruzada.

## Principais resultados

### Seleção na validação cruzada

| Modelo ponderado | Balanced Accuracy média |
|---|---:|
| **Regressão Logística** | **76,60%** |
| SGD Classifier | 76,54% |
| Floresta Aleatória | 76,51% |
| Árvore de Decisão | 75,58% |

A **Regressão Logística ponderada** foi selecionada com `C=10`, `fit_intercept=True` e `class_weight="balanced"`.

As médias dos três primeiros modelos são próximas. O resultado identifica a maior média observada, sem demonstrar superioridade estatística.

### Desempenho no teste

Valores percentuais. Precision, Recall e F1 referem-se à classe **Churn = 1**.

| Modelo | Ponderação | Accuracy | Precision | Recall | F1 | Balanced Accuracy | F1 macro |
|---|---|---:|---:|---:|---:|---:|---:|
| Regressão Logística | Sem | 79,91 | 65,32 | 51,87 | 57,82 | 70,96 | 72,32 |
| **Regressão Logística** | **Balanced** | **73,31** | **49,83** | **78,88** | **61,08** | **75,09** | **70,39** |
| SGD Classifier | Sem | 79,49 | 65,02 | 49,20 | 56,01 | 69,82 | 71,32 |
| SGD Classifier | Balanced | 74,80 | 51,66 | 78,88 | 62,43 | 76,11 | 71,74 |
| Árvore de Decisão | Sem | 78,64 | 61,16 | 53,48 | 57,06 | 70,60 | 71,42 |
| Árvore de Decisão | Balanced | 74,59 | 51,40 | 78,34 | 62,08 | 75,79 | 71,49 |
| Floresta Aleatória | Sem | 79,70 | 65,07 | 50,80 | 57,06 | 70,47 | 71,88 |
| Floresta Aleatória | Balanced | 74,52 | 51,30 | 78,88 | 62,17 | 75,91 | 71,48 |

O SGD ponderado apresentou a maior Balanced Accuracy no teste. A seleção da Regressão Logística foi mantida porque a regra utiliza a validação, evitando escolher novamente após observar o teste.

### O que mudou com a ponderação?

Na comparação entre as versões otimizadas da Regressão Logística:

| Resultado no teste | Sem ponderação | Ponderada |
|---|---:|---:|
| Cancelamentos identificados — TP | 194 | 295 |
| Cancelamentos não identificados — FN | 180 | 79 |
| Falsos alertas — FP | 103 | 297 |
| Não cancelamentos corretamente classificados — TN | 932 | 738 |

Foram identificados **101 cancelamentos adicionais**, acompanhados de **194 falsos alertas adicionais**. O ganho de Recall e Balanced Accuracy não representa melhoria de todas as métricas nem comprova benefício financeiro.

Como cada cenário foi otimizado separadamente, a comparação envolve ponderação e hiperparâmetros selecionados em conjunto.

## Aprendizados da análise

- Clientes com até 12 meses de permanência apresentaram **47,44%** de churn; entre 61 e 72 meses, **6,61%**.
- Contratos mensais apresentaram **42,71%**, contra **2,83%** nos contratos de dois anos.
- A linha de **26,54%** dos gráficos é a referência global; cada barra representa a taxa do respectivo grupo.
- Os atributos derivados não demonstraram ganho na configuração avaliada: Balanced Accuracy média de **76,54%** com originais e **76,50%** com originais + derivados.
- As associações observadas não estabelecem causalidade.

## Como reproduzir

### 1. Obter o projeto

```bash
git clone https://github.com/BrazTec/telco-customer-churn.git
cd telco-customer-churn
```

### 2. Criar o ambiente e instalar as dependências

O ambiente registrado utiliza **Python 3.11.9**. No Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r ambiente_reprodutibilidade/requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --user --name telco-churn --display-name "Python (telco-churn)"
```

No macOS ou Linux, com Python 3.11 instalado:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r ambiente_reprodutibilidade/requirements.txt
.venv/bin/python -m ipykernel install --user --name telco-churn --display-name "Python (telco-churn)"
```

### 3. Executar o notebook

Abra `ML_Eduardo_Braz.ipynb` no VS Code com suporte a Jupyter ou em uma interface Jupyter já instalada. Selecione o kernel **Python (telco-churn)** e execute **Restart Kernel and Run All Cells**.

Para instalar uma interface JupyterLab separadamente no ambiente, se necessário:

```powershell
# Windows
.\.venv\Scripts\python.exe -m pip install jupyterlab
.\.venv\Scripts\python.exe -m jupyterlab
```

```bash
# macOS/Linux
.venv/bin/python -m pip install jupyterlab
.venv/bin/python -m jupyterlab
```

JupyterLab é uma ferramenta adicional de execução; não integra o arquivo de dependências diretas registrado no experimento.

Execute a partir da raiz do projeto. A célula de ambiente grava os arquivos em `ambiente_reprodutibilidade/`, relativa ao diretório de trabalho do kernel, atualizando o registro para o ambiente da nova execução.

As buscas realizam **300 ajustes de validação**, além dos reajustes finais e do experimento de engenharia. O tempo depende do computador.

## Ambiente registrado

| Componente | Versão |
|---|---|
| Python | 3.11.9 |
| NumPy | 2.3.3 |
| pandas | 3.0.1 |
| scikit-learn | 1.8.0 |
| Matplotlib | 3.10.6 |
| kagglehub | 1.0.2 |
| IPython | 9.7.0 |
| ipykernel | 7.1.0 |

Execução registrada em **Windows / AMD64**. O arquivo de dependências fixa versões diretas; não é um bloqueio de todas as dependências transitivas. A reprodução em outras plataformas não foi validada neste projeto.

## Estrutura do repositório

```text
telco-customer-churn/
├── README.md
├── .gitignore
├── ML_Eduardo_Braz.ipynb
└── ambiente_reprodutibilidade/
    ├── requirements.txt
    └── ambiente_execucao.json
```

## Limitações e aplicação

Este é um estudo acadêmico de classificação do churn observado. A base não define, no experimento, uma data de referência dos atributos e uma janela futura de cancelamento. Os resultados não demonstram antecipação do churn com determinada antecedência.

AP1 e AP2 são etapas do mesmo trabalho e mantêm a divisão para comparar a evolução. O teste já foi observado durante essa continuidade; as buscas e a escolha final implementadas utilizam o treino e a validação cruzada.

Uma aplicação preventiva exigiria validação temporal, dados novos e análise dos custos e da efetividade das intervenções. Os coeficientes e as taxas por grupo não devem ser interpretados como efeitos causais.

## Autor

**Eduardo Braz de Oliveira** · [BrazTec](https://github.com/BrazTec)

Trabalho desenvolvido na disciplina de Machine Learning, sob orientação do professor **Arthur Schneider Figueira**.
