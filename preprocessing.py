# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def selecionar_e_transformar_features(df):
    """
    Seleciona as features relevantes, aplica One-Hot Encoding nas categóricas
    e retorna os dataframes prontos para a próxima etapa.

    Args:
        df (pd.DataFrame): O DataFrame original completo.

    Returns:
        tuple: Contendo dois DataFrames:
               - df_numerico_original: Apenas com as features numéricas originais (para análise de perfil).
               - df_final_features: Completo, com features numéricas e categóricas transformadas (para modelagem).
    """
    # Copia o dataframe para evitar o SettingWithCopyWarning
    df_processado = df.copy()

    # 1. Separa as colunas numéricas que serão usadas no clustering
    features_numericas = df_processado.select_dtypes(include=['number']).columns.tolist()
    if 'cliente_id' in features_numericas:
        features_numericas.remove('cliente_id')
    
    # Guarda o dataframe com apenas as features numéricas originais para a análise de perfil
    df_numerico_original = df_processado[features_numericas]

    # 2. Identifica as colunas categóricas para transformação
    features_categoricas = df_processado.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Features numéricas identificadas: {features_numericas}")
    print(f"Features categóricas para One-Hot Encoding: {features_categoricas}")

    # 3. Aplica One-Hot Encoding nas variáveis categóricas
    # drop_first=True remove a primeira categoria de cada feature para evitar multicolinearidade
    # dtype=float garante que as novas colunas sejam numéricas
    df_encoded = pd.get_dummies(df_processado[features_categoricas], drop_first=True, dtype=float)
    
    # 4. Combina as features numéricas originais com as categóricas transformadas
    df_final_features = pd.concat([df_numerico_original, df_encoded], axis=1)
    
    print("One-Hot Encoding aplicado com sucesso.")
    print(f"Dimensões do DataFrame final para modelagem: {df_final_features.shape}")
    
    # 5. Retorna ambos os DataFrames
    return df_numerico_original, df_final_features

def padronizar_dados(df_features):
    """
    Padroniza os dados usando StandardScaler (média 0, desvio padrão 1).
    """
    scaler = StandardScaler()
    dados_padronizados = scaler.fit_transform(df_features)
    df_padronizado = pd.DataFrame(dados_padronizados, columns=df_features.columns)
    print("Dados padronizados com sucesso.")
    return df_padronizado

def normalizar_para_radar(df):
    """
    Normaliza os dados para a visualização em Radar Chart (escala 0-1).
    """
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    return df_normalized