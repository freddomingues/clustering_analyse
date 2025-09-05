# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import StandardScaler

def selecionar_features(df):
    """
    Seleciona as colunas numéricas relevantes para a clusterização.
    """
    # Cria uma cópia para evitar SettingWithCopyWarning
    df_features = df.copy()
    features = df_features.select_dtypes(include=['number']).columns.tolist()
    
    if 'id_cliente' in features:
        features.remove('id_cliente')
        
    print(f"Features selecionadas para a modelagem: {features}")
    return df_features[features]

def padronizar_dados(df_features):
    """
    Padroniza os dados usando StandardScaler.
    """
    scaler = StandardScaler()
    dados_padronizados = scaler.fit_transform(df_features)
    df_padronizado = pd.DataFrame(dados_padronizados, columns=df_features.columns)
    print("Dados padronizados com sucesso.")
    return df_padronizado

