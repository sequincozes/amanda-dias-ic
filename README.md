# XGBoost + SHAP com dados do ERENO

Projeto introdutório para classificar registros de rede com **XGBoost** e
explicar as decisões com **SHAP**. O código prioriza legibilidade: cada
responsabilidade está em uma classe separada e `main.py` apenas inicia a
aplicação.

## Aviso sobre o CSV incluído

O arquivo `data/top_50.csv` contém **49 registros, 70 colunas e somente a
classe `normal`**. Ele permite verificar a leitura em UTF-16 e a limpeza dos
tipos, mas não permite treinar um classificador supervisionado: o modelo
precisa observar pelo menos uma classe normal e uma classe de ataque.

Por isso, ao executar o exemplo anexado, o programa encerra com uma mensagem
explicativa. Para realizar o experimento completo, substitua o arquivo por uma
amostra maior do ERENO que contenha registros normais e ataques.

## Preparação do ambiente

É recomendável usar Python 3.11 ou 3.12.

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux ou macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Como executar

Coloque o dataset completo na pasta `data` e execute:

```bash
python main.py --dados data/ereno_completo.csv
```

O modo padrão é binário:

- `normal` → classe 0;
- qualquer outro rótulo → classe 1 (`intrusão`).

Para distinguir cada tipo de ataque original:

```bash
python main.py --dados data/ereno_completo.csv --modo-alvo multiclasse
```

Se o nome da coluna ou do rótulo normal for diferente:

```bash
python main.py --dados data/ereno_completo.csv --alvo class --rotulo-normal normal
```

Use `python main.py --help` para ver todas as opções.

## Pré-processamento adotado

O arquivo do ERENO mistura medições elétricas, contadores, indicadores e
identificadores de protocolo. O projeto aplica:

1. detecção automática entre UTF-16 e codificações UTF-8 usuais;
2. remoção de espaços extras em nomes e valores textuais;
3. conversão de colunas textuais para número somente quando todos os valores
   preenchidos forem numéricos;
4. preenchimento de números ausentes pela mediana;
5. preenchimento de categorias ausentes pela categoria mais frequente;
6. `OneHotEncoder(handle_unknown="ignore")` para categorias nominais, pois
   códigos como endereço MAC, APPID e identificadores GOOSE não possuem uma
   ordem quantitativa natural;
7. remoção de colunas constantes aprendida somente no conjunto de treino;
8. remoção padrão de `Time`, `GooseTimestamp` e `t`, reduzindo o risco de o
   modelo memorizar a ordem da captura. As diferenças temporais continuam
   disponíveis.

Não há normalização ou padronização de escala: árvores de decisão não dependem
de distância entre valores como KNN ou SVM.

## Separação correta entre treino e teste

O programa divide os registros de forma estratificada. O preprocessador é
ajustado **somente com o treino** e depois aplicado ao teste. Isso evita que
medianas, categorias ou colunas constantes do teste influenciem o treinamento.

Para uma publicação científica, uma divisão aleatória ainda pode ser otimista
quando pacotes vizinhos pertencem à mesma captura. Nesse caso, substitua-a por
uma separação temporal ou por grupos de cenário/captura independentes.

## Arquivos gerados

Depois de uma execução válida, a pasta `resultados` terá:

- `avaliacao/metricas.json`: precision, recall, F1 e, no binário, ROC AUC;
- `avaliacao/relatorio_classificacao.txt`: relatório fácil de ler;
- `avaliacao/matriz_confusao.png`;
- `previsoes_teste.csv`: classe real, prevista e probabilidades;
- `modelo/pacote_modelo.joblib`: modelo, preprocessador e codificador;
- `modelo/modelo_xgboost.json`: representação nativa do XGBoost;
- `modelo/metadados_modelo.json`: colunas e transformações utilizadas;
- `shap/shap_importancia_global.png`: média do valor SHAP absoluto;
- `shap/shap_beeswarm.png`: direção e distribuição dos efeitos;
- `shap/shap_waterfall_local.png`: explicação de uma única decisão;
- `shap/shap_dependencia_top_atributo.png`: efeito do atributo globalmente mais
  importante;
- `shap/metadados_shap.json`: classe e atributo explicados.

No XGBoost, o `TreeExplainer` explica por padrão a saída interna do modelo
(margem). Portanto, um valor SHAP positivo aumenta essa saída para a classe
explicada, mas não deve ser lido diretamente como “pontos percentuais”.

## Organização do código

- `src/carregador_dados.py`: leitura, limpeza e validação inicial;
- `src/codificador_alvo.py`: alvo binário ou multiclasse;
- `src/preprocessamento.py`: imputação, One-Hot Encoding e colunas constantes;
- `src/classificador.py`: treinamento e predição com XGBoost;
- `src/avaliador.py`: métricas e matriz de confusão;
- `src/preditor.py`: tabela de classificações e probabilidades;
- `src/explicador_shap.py`: gráficos globais e locais;
- `src/persistencia.py`: salvamento do modelo e metadados;
- `src/aplicacao.py`: sequência completa do experimento;
- `src/cli.py`: argumentos de linha de comando;
- `main.py`: uma única chamada para iniciar a aplicação.

## Leituras oficiais

- [XGBoost: interface Python e scikit-learn](https://xgboost.readthedocs.io/en/stable/python/python_intro.html)
- [SHAP: referência de gráficos](https://shap.readthedocs.io/en/stable/api.html)
- [scikit-learn: OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [ERENO Framework](https://ieeexplore.ieee.org/document/10339874/)
