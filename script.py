#REQ 1
# faça os imports que julgar necessários
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
#REQ 2
#essa função deve devolver a base de dados
def ler_base():
  return pd.read_csv('C:/Users/gu-gu/OneDrive/Desktop/Programação Banco de Dados/P1/20251_fatec_ipi_pbdi_p2_modelo/dados.csv')

#REQ 3
#essa função recebe a base lida anteriormente
#ela deve devolver uma tupla contendo as features e a classe
def dividir_em_features_e_classe(base):
    features = base.iloc[:, :-1]
    classe = base.iloc[:, -1]
    return features, classe 


#REQ 4
#essa função recebe as features
#ela deve devolver as features da seguinte forma
#Valores faltantes da coluna "Gastos com pesquisa e desenvolvimento": substituir pela média
#Valores faltantes da coluna "Gastos com administracao": substituir pela mediana
#Valores faltantes da coluna "Gastos com marketing": Substituir por zero
#Valores faltantes da coluna "Estado": Substituir pela moda
def lidar_com_valores_faltantes(features):
    transformer = ColumnTransformer([
        ('pesquisa_media', SimpleImputer(strategy='mean'), ['Gastos com pesquisa e desenvolvimento']),
        ('admin_mediana', SimpleImputer(strategy='median'), ['Gastos com administracao']),
        ('marketing_zero', SimpleImputer(strategy='constant', fill_value=0), ['Gastos com marketing']),
        ('estado_moda', SimpleImputer(strategy='most_frequent'), ['Estado'])
    ], remainder='passthrough')
    features_tratadas = transformer.fit_transform(features)
    df_features_tratadas = pd.DataFrame(features_tratadas, columns=features.columns)
    return df_features_tratadas

#REQ 5
#essa função recebe as features
#ela deve devolver as features da seguinte forma
#Variável "Estado": Codificar com OneHotEncoding
def codificar_categoricas(features):
    coluna_estado = features[['Estado']]
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    estado_codificado = encoder.fit_transform(coluna_estado)
    
    df_estado_codificado = pd.DataFrame(
        estado_codificado,
        columns=encoder.get_feature_names_out(['Estado']),
        index=features.index
    )
    
    features_sem_estado = features.drop('Estado', axis=1)
    features_finais = pd.concat([features_sem_estado, df_estado_codificado], axis=1)
    
    return features_finais


#REQ 6
#essa função recebe as features e a classe
#ela deve devolver uma tupla com 4 itens
# features de treinamento, features de teste, classe de treinamento, classe de teste
# a base de treinamento deve ter 75% das instâncias
def obter_bases_de_treinamento_e_teste(features, classe):
  return train_test_split(features, classe, test_size=0.25, random_state=42)

#REQ 7
#essa função recebe as features de treinamento e de teste
#ela deve devolver uma tupla com 2 itens, da seguinte forma
#todas as variáveis normalizadas com o método MinMax
def normalizar(features_treinamento, features_teste):
  scaler = MinMaxScaler()
    
  colunas = features_treinamento.columns
  indice_treinamento = features_treinamento.index
  indice_teste = features_teste.index
    
  scaler.fit(features_treinamento)
    
  features_treinamento_norm = scaler.transform(features_treinamento)
  features_teste_norm = scaler.transform(features_teste)
    
  df_treinamento_norm = pd.DataFrame(features_treinamento_norm, index=indice_treinamento, columns=colunas)
  df_teste_norm = pd.DataFrame(features_teste_norm, index=indice_teste, columns=colunas)
    
  return df_treinamento_norm, df_teste_norm

#REQ 8
  #chame as suas funções aqui
  #exiba as quatro bases aqui

def vai():
    base = ler_base()
    features, classe = dividir_em_features_e_classe(base)
    features_sem_na = lidar_com_valores_faltantes(features)
    print("--- Features após tratar valores faltantes  ---")
    print(features_sem_na.head())

    features_codificadas = codificar_categoricas(features_sem_na)
    print("\n--- Features após codificar 'Estado'  ---")
    print(features_codificadas.head())
    
    features_treinamento, features_teste, classe_treinamento, classe_teste = \
        obter_bases_de_treinamento_e_teste(features_codificadas, classe)
        
    features_treinamento_final, features_teste_final = \
        normalizar(features_treinamento, features_teste)
    
    print("\n\n --- BASES DE DADOS FINAIS PRONTAS --- ")
    print("\n1. Features de Treinamento (Normalizadas):")
    print(features_treinamento_final.head())
    print("\n2. Features de Teste (Normalizadas):")
    print(features_teste_final.head())
    print("\n3. Classe de Treinamento:")
    print(classe_treinamento.head())
    print("\n4. Classe de Teste:")
    print(classe_teste.head())

vai()