# Segmentação de clientes para otimizar estratégias de negociação de dívidas

## 📋 Sobre o Projeto

Este projeto implementa um **pipeline completo de análise de clusterização** para segmentação de clientes inadimplentes utilizando técnicas de Machine Learning não-supervisionado. O sistema identifica grupos homogêneos de clientes com base em características sociodemográficas, financeiras e comportamentais, permitindo estratégias personalizadas de cobrança e análise de risco.

## 🎯 Objetivos

- **Segmentação de Clientes**: Identificar grupos distintos de clientes inadimplentes
- **Análise de Perfis**: Caracterizar cada segmento por características numéricas e categóricas
- **Comparação de Modelos**: Avaliar diferentes algoritmos de clusterização (K-Means, Hierárquico, DBSCAN)
- **Visualização**: Gerar gráficos e análises visuais dos clusters identificados
- **Base para Decisões**: Fornecer insights para estratégias de recuperação de crédito

## 🏗️ Arquitetura do Projeto

O projeto segue uma arquitetura modular, organizada em módulos especializados:

```
clustering_analyse/
├── main.py                 # Orquestrador principal do pipeline
├── data_generator.py        # Geração de dados sintéticos realistas
├── preprocessing.py         # Pré-processamento e transformação de dados
├── clustering_models.py     # Implementação dos algoritmos de clusterização
├── evaluation.py            # Métricas e avaliação dos modelos
├── visualization.py         # Visualizações e gráficos
├── dashboard.py             # Dashboard interativo (Streamlit)
└── DOCUMENTACAO_TECNICA.md  # Documentação técnica detalhada
```

## 🚀 Funcionalidades

### 1. Geração de Dados Sintéticos
- Criação de base de dados realista com até 30.000 registros
- Variáveis demográficas, financeiras e comportamentais
- Distribuições estatísticas realistas (Normal, Poisson, Beta, Exponencial)
- Reprodutibilidade garantida via seed

### 2. Pré-processamento
- Seleção automática de features numéricas e categóricas
- One-Hot Encoding para variáveis categóricas
- Padronização (Z-score) para algoritmos baseados em distância
- Preparação de dados para múltiplos algoritmos

### 3. Algoritmos de Clusterização

#### K-Means
- Aplicado no dataset completo
- Inicialização inteligente (k-means++)
- Determinação do K ótimo via método do cotovelo e análise de silhueta

#### Clusterização Hierárquica
- Aplicado em amostra de 10.000 registros (limitação de complexidade)
- Algoritmo aglomerativo
- Útil para análise exploratória de relacionamentos

#### DBSCAN
- Aplicado no dataset completo
- Identificação automática de número de clusters
- Detecção de outliers e ruído
- Baseado em densidade

### 4. Avaliação de Modelos
- **Coeficiente de Silhueta**: Mede separação e coesão dos clusters
- **Índice de Davies-Bouldin**: Avalia qualidade da separação
- Tabela comparativa de desempenho

### 5. Análise de Perfis
- **Perfil Numérico**: Médias das variáveis numéricas por cluster
- **Perfil Categórico**: Moda (valor mais frequente) das variáveis categóricas
- Caracterização completa de cada segmento

### 6. Visualizações
- Gráfico do Método do Cotovelo
- Análise de Silhueta
- Visualização 2D via PCA (Principal Component Analysis)
- Gráficos de distribuição e correlação

## 📦 Instalação

### Pré-requisitos

- Python 3.7 ou superior
- pip (gerenciador de pacotes Python)

### Passos de Instalação

1. **Clone o repositório** (ou navegue até o diretório do projeto):
```bash
cd clustering_analyse
```

2. **Instale as dependências**:
```bash
pip install -r requirements.txt
```

3. **Execute o projeto**:
```bash
python main.py
```

## 📚 Dependências

As principais bibliotecas utilizadas são:

- **pandas**: Manipulação e análise de dados
- **numpy**: Operações numéricas
- **scikit-learn**: Algoritmos de ML e pré-processamento
- **matplotlib**: Visualização de dados
- **seaborn**: Visualizações estatísticas avançadas
- **openpyxl**: Leitura/escrita de arquivos Excel

Consulte o arquivo `requirements.txt` para a lista completa de dependências e versões.

## 🔧 Como Usar

### Execução Básica

Execute o arquivo principal:

```bash
python main.py
```

O pipeline executará automaticamente todas as etapas:

1. **Geração/Carregamento de Dados**: Verifica se existe `base_sintetica_dividas.xlsx`. Se não existir, gera uma nova base.

2. **Pré-processamento**: Transforma e padroniza os dados.

3. **Determinação de K Ótimo**: Calcula métricas para diferentes valores de K e exibe gráficos.

4. **Aplicação dos Modelos**: Executa K-Means, Hierárquico e DBSCAN.

5. **Avaliação**: Calcula métricas de qualidade e exibe tabela comparativa.

6. **Visualização**: Gera gráficos PCA para cada modelo.

7. **Análise de Perfis**: Exibe características numéricas e categóricas de cada cluster.

### Personalização

#### Alterar Número de Clusters

No arquivo `main.py`, linha 49:
```python
K_OTIMO = 4  # Altere para o valor desejado
```

#### Ajustar Parâmetros do DBSCAN

No arquivo `main.py`, linha 66:
```python
labels_dbscan = clustering_models.aplicar_dbscan(
    df_padronizado, 
    eps=2.5,        # Raio de vizinhança
    min_samples=20  # Mínimo de pontos por cluster
)
```

#### Modificar Tamanho da Amostra (Hierárquico)

No arquivo `main.py`, linhas 59-63:
```python
if len(df_padronizado) > 10000:
    df_amostra = df_padronizado.sample(n=10000, random_state=42)
else:
    df_amostra = df_padronizado
```

## 📊 Estrutura dos Dados

### Variáveis do Dataset

#### Numéricas
- `cliente_id`: Identificador único
- `idade`: Idade do cliente (18-85 anos)
- `numero_dependentes`: Número de dependentes (0-8)
- `renda_mensal`: Renda mensal em reais
- `score_credito`: Score de crédito (300-950)
- `historico_pagamento_recente`: Histórico de pagamento (0-1)
- `tempo_de_debito_meses`: Tempo em débito (1-60 meses)
- `valor_divida`: Valor da dívida em reais

#### Categóricas
- `sexo`: Masculino, Feminino
- `estado_civil`: Solteiro, Casado, Divorciado, Viúvo
- `nivel_educacional`: Fundamental, Médio, Superior, Pós-graduação
- `tipo_emprego`: CLT, Autônomo, Funcionário Público, Empresário, Desempregado
- `produto_origem_divida`: Cartão de Crédito, Empréstimo Pessoal, Financiamento Veículo, Cheque Especial

## 📈 Resultados Esperados

### Saídas do Sistema

1. **Arquivo Excel**: `base_sintetica_dividas.xlsx` (gerado na primeira execução)

2. **Gráficos Exibidos**:
   - Método do Cotovelo
   - Análise de Silhueta
   - Visualizações PCA (um por modelo)

3. **Tabelas no Console**:
   - Tabela de Avaliação Comparativa (Silhueta e Davies-Bouldin)
   - Perfil Numérico Médio dos Clusters
   - Perfil Categórico (Moda) dos Clusters

### Interpretação dos Resultados

#### Coeficiente de Silhueta
- **0.7-1.0**: Estrutura de clusters muito forte
- **0.5-0.7**: Estrutura razoável
- **0.25-0.5**: Estrutura fraca
- **< 0.25**: Sem estrutura significativa

#### Índice de Davies-Bouldin
- **Valores menores**: Melhor separação entre clusters
- **Ideal**: Próximo de 0

## 🔍 Exemplos de Uso

### Exemplo 1: Análise Completa

```python
python main.py
```

Executa o pipeline completo com os parâmetros padrão.

### Exemplo 2: Usar Dados Próprios

1. Prepare um arquivo Excel com as colunas esperadas (veja seção "Estrutura dos Dados")
2. Renomeie para `base_sintetica_dividas.xlsx`
3. Coloque na raiz do projeto
4. Execute `python main.py`

O sistema carregará automaticamente seus dados.

## 🧪 Testes e Validação

O projeto utiliza dados sintéticos para desenvolvimento e testes. Para uso em produção:

1. **Valide com dados reais**: Teste com uma amostra pequena primeiro
2. **Ajuste parâmetros**: Otimize eps, min_samples e K conforme seus dados
3. **Valide com especialistas**: Confirme se os clusters fazem sentido para o negócio
4. **Monitore performance**: Acompanhe métricas ao longo do tempo

## 📖 Documentação Adicional

Para informações técnicas detalhadas sobre:
- Arquitetura dos módulos
- Algoritmos implementados
- Decisões de design
- Fórmulas e métricas
- Limitações e extensões futuras

Consulte o arquivo **[DOCUMENTACAO_TECNICA.md](DOCUMENTACAO_TECNICA.md)**.

## 🤝 Contribuindo

Este é um projeto de análise e pode ser estendido com:

- Novos algoritmos de clusterização
- Métricas de avaliação adicionais
- Visualizações interativas
- Integração com bancos de dados
- APIs para predição em tempo real

## 📝 Licença

Este projeto é fornecido como está, para fins educacionais e de análise.

## 👤 Autor

Frederico Antônio Domingues

## 🔗 Referências

- **Scikit-learn**: Documentação oficial de clusterização
- **Pandas**: Guia de manipulação de dados
- **Matplotlib/Seaborn**: Documentação de visualização

Para referências acadêmicas dos algoritmos, consulte a seção "Referências Técnicas" em `DOCUMENTACAO_TECNICA.md`.

---

**Nota**: Este projeto utiliza dados sintéticos. Para uso em produção com dados reais, certifique-se de seguir todas as regulamentações de proteção de dados (LGPD, GDPR, etc.).
