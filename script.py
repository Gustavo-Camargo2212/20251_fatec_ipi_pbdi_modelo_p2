#REQ 1
# faça os imports que julgar necessários
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
import numpy as np
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
    features_tratadas = features.copy()
    media_pesquisa = features_tratadas['Gastos com pesquisa e desenvolvimento'].mean()
    features_tratadas['Gastos com pesquisa e desenvolvimento'].fillna(media_pesquisa, inplace=True)
    mediana_admin = features_tratadas['Gastos com administracao'].median()
    features_tratadas['Gastos com administracao'].fillna(mediana_admin, inplace=True)
    features_tratadas['Gastos com marketing'].fillna(0, inplace=True)
    moda_estado = features_tratadas['Estado'].mode()[0]
    features_tratadas['Estado'].fillna(moda_estado, inplace=True)
    return features_tratadas

#REQ 5
#essa função recebe as features
#ela deve devolver as features da seguinte forma
#Variável "Estado": Codificar com OneHotEncoding
def codificar_categoricas(features):
  pass

#REQ 6
#essa função recebe as features e a classe
#ela deve devolver uma tupla com 4 itens
# features de treinamento, features de teste, classe de treinamento, classe de teste
# a base de treinamento deve ter 75% das instâncias
def obter_bases_de_treinamento_e_teste(features, classe):
  pass

#REQ 7
#essa função recebe as features de treinamento e de teste
#ela deve devolver uma tupla com 2 itens, da seguinte forma
#todas as variáveis normalizadas com o método MinMax
def normalizar(features_treinamento, features_teste):
  pass

#REQ 8
def vai():
  #chame as suas funções aqui
  #exiba as quatro bases aqui
     pass

vai()