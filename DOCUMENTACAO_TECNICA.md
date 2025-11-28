# Documentação Técnica - Sistema de Análise de Clusterização de Clientes Inadimplentes

## 1. Visão Geral do Projeto

Este projeto implementa um pipeline completo de análise de clusterização para segmentação de clientes inadimplentes. O sistema utiliza técnicas de Machine Learning não-supervisionado para identificar grupos homogêneos de clientes com base em características sociodemográficas, financeiras e comportamentais.

### 1.1 Objetivo

O objetivo principal é identificar padrões e segmentos distintos entre clientes inadimplentes, permitindo:
- Estratégias de cobrança personalizadas
- Análise de risco diferenciada
- Otimização de recursos de recuperação de crédito
- Compreensão de perfis comportamentais

### 1.2 Tecnologias Utilizadas

- **Python 3.x**
- **Pandas**: Manipulação e análise de dados
- **NumPy**: Operações numéricas
- **Scikit-learn**: Algoritmos de clusterização e pré-processamento
- **Matplotlib/Seaborn**: Visualização de dados
- **PCA**: Redução de dimensionalidade para visualização

## 2. Arquitetura do Sistema

O projeto segue uma arquitetura modular, dividida em módulos especializados:

```
clustering_analyse/
├── main.py                 # Orquestrador principal do pipeline
├── data_generator.py        # Geração de dados sintéticos
├── preprocessing.py         # Pré-processamento e transformação
├── clustering_models.py     # Implementação dos algoritmos
├── evaluation.py            # Métricas e avaliação
└── visualization.py         # Visualizações e gráficos
```

## 3. Módulos Detalhados

### 3.1 data_generator.py

**Responsabilidade**: Geração de dados sintéticos realistas de clientes inadimplentes.

#### Função Principal: `gerar_dados_sinteticos()`

**Parâmetros**:
- `n_clientes` (int): Número de registros a gerar (padrão: 30.000)
- `seed` (int): Semente para reprodutibilidade (padrão: 42)

**Características dos Dados Gerados**:

1. **Variáveis Demográficas**:
   - `cliente_id`: Identificador único
   - `idade`: Distribuição normal (μ=45, σ=15), limitada entre 18-85 anos
   - `sexo`: Binomial (48% Masculino, 52% Feminino)
   - `estado_civil`: Categórica (Solteiro 35%, Casado 45%, Divorciado 15%, Viúvo 5%)
   - `numero_dependentes`: Distribuição de Poisson (λ=1.2), limitada entre 0-8
   - `nivel_educacional`: Categórica (Fundamental 15%, Médio 50%, Superior 25%, Pós-graduação 10%)
   - `tipo_emprego`: Categórica (CLT 50%, Autônomo 20%, Funcionário Público 10%, Empresário 10%, Desempregado 10%)

2. **Variáveis Financeiras**:
   - `renda_mensal`: Calculada com base em educação e tipo de emprego, com variação aleatória
   - `score_credito`: Baseado em renda, idade e histórico (300-950)
   - `historico_pagamento_recente`: Distribuição Beta diferenciada para perfis de risco (30% arriscados, 70% bons pagadores)
   - `valor_divida`: Proporcional à renda (20% a 200% da renda mensal)
   - `tempo_de_debito_meses`: Distribuição exponencial (média 18 meses), limitada entre 1-60 meses

3. **Variáveis Categóricas**:
   - `produto_origem_divida`: Cartão de Crédito (40%), Empréstimo Pessoal (30%), Financiamento Veículo (15%), Cheque Especial (15%)

**Distribuições Utilizadas**:
- **Normal**: Para idade (mais realista que uniforme)
- **Poisson**: Para número de dependentes (modela contagens)
- **Beta**: Para histórico de pagamento (permite diferentes perfis de risco)
- **Exponencial**: Para tempo de débito (maioria recente, poucos antigos)

**Retorno**: DataFrame pandas com todas as variáveis geradas

### 3.2 preprocessing.py

**Responsabilidade**: Transformação e preparação dos dados para modelagem.

#### Função: `selecionar_e_transformar_features()`

**Processo**:
1. Separa features numéricas e categóricas
2. Remove `cliente_id` das features numéricas (não é preditora)
3. Aplica **One-Hot Encoding** nas variáveis categóricas
   - `drop_first=True`: Remove primeira categoria para evitar multicolinearidade
   - `dtype=float`: Garante tipo numérico
4. Combina features numéricas originais com categóricas codificadas

**Retorno**: Tupla contendo:
- `df_numerico_original`: Features numéricas originais (para análise de perfil)
- `df_final_features`: Dataset completo transformado (para modelagem)

#### Função: `padronizar_dados()`

**Método**: StandardScaler (Z-score normalization)
- Média = 0
- Desvio padrão = 1

**Justificativa**: Necessário para algoritmos baseados em distância (K-Means, Hierárquico, DBSCAN)

#### Função: `normalizar_para_radar()`

**Método**: MinMaxScaler
- Escala 0-1
- Usado para visualização em gráficos radar

### 3.3 clustering_models.py

**Responsabilidade**: Implementação dos algoritmos de clusterização.

#### Função: `encontrar_k_otimo()`

**Método**: Avalia múltiplos valores de K (2 a max_k)

**Métricas Calculadas**:
- **Inércia (WCSS)**: Soma dos quadrados das distâncias intra-cluster
- **Coeficiente de Silhueta**: Mede quão bem separados estão os clusters

**Uso**: Determina o número ótimo de clusters via método do cotovelo e análise de silhueta

**Otimização**: Utiliza cache do Streamlit (prefixo `_` no parâmetro) para melhor performance em dashboards

#### Função: `aplicar_kmeans()`

**Algoritmo**: K-Means
- `init='k-means++'`: Inicialização inteligente dos centróides
- `random_state=42`: Reprodutibilidade
- `n_init=10`: Múltiplas inicializações para evitar mínimos locais

**Características**:
- Requer número pré-definido de clusters
- Baseado em distância euclidiana
- Eficiente para grandes datasets

#### Função: `aplicar_cluster_hierarquico()`

**Algoritmo**: Agglomerative Clustering (Hierárquico Aglomerativo)

**Características**:
- Não requer número pré-definido de clusters (mas usa n_clusters para comparação)
- Cria dendrograma de relacionamentos
- **Limitação**: Complexidade O(n²) - aplicado apenas em amostra de 10.000 registros

**Uso**: Aplicado em amostra quando dataset > 10.000 registros

#### Função: `aplicar_dbscan()`

**Algoritmo**: DBSCAN (Density-Based Spatial Clustering)

**Parâmetros**:
- `eps=2.5`: Raio de vizinhança
- `min_samples=20`: Número mínimo de pontos para formar cluster

**Características**:
- Não requer número pré-definido de clusters
- Identifica ruído (label -1)
- Detecta clusters de forma arbitrária
- Robusto a outliers

**Retorno**: Labels incluindo -1 para pontos de ruído

### 3.4 evaluation.py

**Responsabilidade**: Avaliação quantitativa e qualitativa dos modelos.

#### Função: `avaliar_modelos()`

**Métricas Implementadas**:

1. **Coeficiente de Silhueta**:
   - Range: -1 a 1
   - Valores próximos de 1: Clusters bem separados
   - Valores próximos de 0: Clusters sobrepostos
   - Valores negativos: Pontos atribuídos ao cluster errado

2. **Índice de Davies-Bouldin**:
   - Range: 0 a ∞
   - Valores menores: Melhor separação entre clusters
   - Considera distância intra-cluster e inter-cluster

**Tratamento de Casos Especiais**:
- Ignora modelos com menos de 2 clusters
- Ignora modelos onde todos os pontos são ruído (DBSCAN)

#### Função: `analisar_perfis_clusters()`

**Objetivo**: Caracterizar cada cluster por suas médias numéricas

**Processo**:
1. Adiciona labels ao DataFrame original
2. Remove pontos de ruído (label -1) se existirem
3. Calcula média de cada variável numérica por cluster
4. Adiciona contagem de clientes por cluster

**Retorno**: DataFrame com perfil médio de cada cluster

#### Função: `analisar_perfis_categoricos()`

**Objetivo**: Caracterizar cada cluster pela moda (valor mais frequente) das variáveis categóricas

**Processo**:
1. Agrupa por cluster
2. Calcula moda para cada variável categórica
3. Adiciona contagem de clientes

**Retorno**: DataFrame com perfil categórico de cada cluster

### 3.5 visualization.py

**Responsabilidade**: Geração de visualizações para análise exploratória e resultados.

#### Funções de Análise Exploratória:

- `plotar_matriz_correlacao()`: Heatmap de correlações entre variáveis numéricas
- `plotar_distribuicoes()`: Histogramas com KDE para cada variável numérica

#### Funções de Análise de K Ótimo:

- `plotar_metodo_cotovelo()`: Gráfico de inércia vs. número de clusters
- `plotar_score_silhueta()`: Gráfico de coeficiente de silhueta vs. número de clusters

#### Funções de Visualização de Clusters:

- `plotar_cluster_pca_individual()`: 
  - Reduz dimensionalidade via PCA (2 componentes)
  - Scatter plot colorido por cluster
  - Permite visualizar separação dos clusters em 2D

#### Funções de Análise de Perfil:

- `plotar_radar_individual()`: 
  - Gráfico radar (spider chart) para perfil de um cluster
  - Normaliza valores para escala 0-1
  - Visualiza múltiplas dimensões simultaneamente

### 3.6 main.py

**Responsabilidade**: Orquestração do pipeline completo.

#### Fluxo de Execução:

1. **Geração/Carregamento de Dados**:
   - Verifica existência de `base_sintetica_dividas.xlsx`
   - Se existe: carrega
   - Se não existe: gera e salva

2. **Pré-processamento**:
   - Seleção e transformação de features
   - Padronização dos dados

3. **Determinação de K Ótimo**:
   - Calcula métricas para K de 2 a 10
   - Exibe gráficos de cotovelo e silhueta
   - Define K_OTIMO = 4 (baseado na análise)

4. **Aplicação dos Modelos**:
   - **K-Means**: Dataset completo
   - **Hierárquico**: Amostra de 10.000 (se dataset > 10.000)
   - **DBSCAN**: Dataset completo

5. **Avaliação**:
   - Calcula métricas para cada modelo
   - Gera tabela comparativa
   - Separa avaliação por dataset completo vs. amostra

6. **Visualização**:
   - Gráficos PCA para cada modelo
   - Salva visualizações individuais

7. **Análise de Perfis**:
   - Perfil numérico médio (K-Means e Hierárquico)
   - Perfil categórico (moda) (K-Means e Hierárquico)
   - Exibe tabelas de caracterização

## 4. Decisões de Design

### 4.1 Amostragem para Clusterização Hierárquica

**Decisão**: Aplicar apenas em amostra de 10.000 registros

**Justificativa**:
- Complexidade O(n²) do algoritmo
- Limitações de memória para datasets grandes
- Trade-off entre precisão e viabilidade computacional

### 4.2 Padronização dos Dados

**Decisão**: StandardScaler (Z-score)

**Justificativa**:
- Algoritmos baseados em distância são sensíveis à escala
- Variáveis com escalas diferentes (ex: renda vs. idade) precisam ser normalizadas
- Preserva distribuição original (apenas muda escala)

### 4.3 One-Hot Encoding com drop_first

**Decisão**: Remover primeira categoria de cada variável categórica

**Justificativa**:
- Evita multicolinearidade perfeita
- Reduz dimensionalidade
- Mantém informação completa (n-1 categorias representam n categorias)

### 4.4 K Ótimo = 4

**Decisão**: Definido manualmente após análise dos gráficos

**Justificativa**:
- Baseado em método do cotovelo e análise de silhueta
- Balanceia número de clusters vs. interpretabilidade
- Pode ser ajustado conforme necessidade de negócio

### 4.5 Parâmetros DBSCAN

**Decisão**: eps=2.5, min_samples=20

**Justificativa**:
- Valores ajustados empiricamente
- min_samples alto reduz clusters pequenos (ruído)
- eps ajustado para densidade do dataset padronizado

## 5. Métricas de Avaliação

### 5.1 Coeficiente de Silhueta

**Fórmula**: 
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

Onde:
- `a(i)`: Distância média intra-cluster
- `b(i)`: Distância média ao cluster mais próximo

**Interpretação**:
- 0.7-1.0: Estrutura forte
- 0.5-0.7: Estrutura razoável
- 0.25-0.5: Estrutura fraca
- < 0.25: Sem estrutura significativa

### 5.2 Índice de Davies-Bouldin

**Fórmula**:
```
DB = (1/k) * Σ max(i≠j) [(σi + σj) / d(ci, cj)]
```

Onde:
- `σi`: Dispersão média do cluster i
- `d(ci, cj)`: Distância entre centróides

**Interpretação**:
- Valores menores indicam melhor separação
- Ideal: próximo de 0

## 6. Limitações e Considerações

### 6.1 Dados Sintéticos

- Dados gerados não representam clientes reais
- Relações entre variáveis são simplificadas
- Útil para desenvolvimento e testes, mas requer validação com dados reais

### 6.2 Escalabilidade

- K-Means e DBSCAN: Escalam bem para grandes datasets
- Hierárquico: Limitado a amostras menores
- PCA: Pode ser custoso para datasets muito grandes

### 6.3 Interpretabilidade

- Clusters identificados são estatísticos, não necessariamente interpretáveis para negócio
- Requer análise de perfil para compreensão
- Validação com especialistas de domínio recomendada

### 6.4 Parâmetros dos Algoritmos

- K-Means: Requer K pré-definido
- DBSCAN: Parâmetros eps e min_samples sensíveis
- Hierárquico: Método de linkage não especificado (padrão: ward)

## 7. Extensões Futuras

### 7.1 Melhorias Técnicas

- Implementar validação cruzada para clusterização
- Adicionar mais métricas de avaliação (Calinski-Harabasz, etc.)
- Implementar otimização automática de hiperparâmetros
- Adicionar análise de estabilidade dos clusters

### 7.2 Funcionalidades

- Dashboard interativo (Streamlit já parcialmente implementado)
- Exportação de relatórios em PDF
- API REST para predição de novos clientes
- Integração com banco de dados real

### 7.3 Análises Avançadas

- Análise de tendências temporais
- Segmentação dinâmica
- Análise de coorte
- Modelos preditivos baseados nos clusters

## 8. Referências Técnicas

- **K-Means**: MacQueen, J. (1967). "Some Methods for classification and Analysis of Multivariate Observations"
- **DBSCAN**: Ester, M. et al. (1996). "A density-based algorithm for discovering clusters"
- **Hierárquico**: Ward, J. H. (1963). "Hierarchical grouping to optimize an objective function"
- **Silhueta**: Rousseeuw, P. J. (1987). "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis"

## 9. Estrutura de Dados

### 9.1 Input

**Formato**: Excel (.xlsx) ou DataFrame pandas

**Colunas Esperadas**:
- `cliente_id`: Identificador único
- `idade`: Numérico (18-85)
- `sexo`: Categórico (Masculino, Feminino)
- `estado_civil`: Categórico
- `nivel_educacional`: Categórico
- `numero_dependentes`: Numérico (0-8)
- `tipo_emprego`: Categórico
- `renda_mensal`: Numérico
- `score_credito`: Numérico (300-950)
- `historico_pagamento_recente`: Numérico (0-1)
- `produto_origem_divida`: Categórico
- `tempo_de_debito_meses`: Numérico (1-60)
- `valor_divida`: Numérico

### 9.2 Output

**Arquivos Gerados**:
- `base_sintetica_dividas.xlsx`: Dataset gerado/carregado
- Gráficos PCA: Salvos individualmente por modelo
- Tabelas de avaliação: Exibidas no console

**Estrutura de Retorno**:
- Labels de cluster para cada modelo
- DataFrames de avaliação
- DataFrames de perfil (numérico e categórico)

