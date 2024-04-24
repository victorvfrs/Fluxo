import pandas as pd
import random

# Caminho para o arquivo CSV
caminho_arquivo = 'Dataset/sem_outlier.csv'

# Carregar o dataset
dataset = pd.read_csv(caminho_arquivo)
#Escala 0.01
#[13.18, 14.17, 14.4, 19.05, 22.92, 25.53, 26.14, 30.18, 31.4, 31.43] 19 para 18
#[13.79, 19.66, 24.92, 30.79] 20 para 18

#Escala 0.001
#de 3 para 2
#[10.05, 10.279, 10.355, 11.041, 11.27, 11.346, 11.956, 12.167, 12.261, 12.337, 13.167, 13.328, 13.557, 13.633, 13.862, 14.167, 14.243, 14.548, 14.624, 15.463, 15.539, 15.844, 15.92, 16.53, 16.911, 17.826, 17.902, 18.131, 18.207, 20.113, 20.189, 20.418, 21.167, 21.409, 21.485, 21.714, 22.095, 22.167, 22.4, 22.476, 23.391, 23.696, 23.772, 24.205, 24.687, 24.763, 25.754, 25.983, 26.167, 26.772, 27.05, 27.194, 27.355, 28.041, 28.346, 28.651, 28.956, 29.261, 29.337, 29.642, 30.167, 30.252, 30.328, 30.633, 31.167, 31.548, 31.624, 32.539, 32.615, 32.92, 33.149, 33.911, 34.167, 34.206, 34.826, 34.902]
# de 4 para 2
#[10.66, 10.965, 11.651, 12.566, 12.947, 13.252, 13.9338, 14.167, 14.929, 15.234, 15.678, 16.225, 17.216, 18.512, 18.817, 19.122, 19.503, 19.808, 20.494, 20.799, 21.104, 21.79, 22.781, 23.086, 24.077, 24.382, 25.373, 25.678, 26.059, 26.364, 26.669, 27.66, 27.965, 28.651, 28.956, 29.947, 30.938, 31.243, 31.929, 32.234, 33.225, 33.53, 34.216, 34.521]
# de 7 para 2
#[31.429]
# de 3 para 2 
#[14.243, 12.642, 25.068]
# de 4 para 2
#[13.938, 19.198, 17.521]


# Lista de números para arredondamento
lista_arredondamento =  [13.938, 19.198, 17.521] # Adicione os números que deseja verificar

# Extrair a coluna que contém os números
numeros_originais = dataset['Segundo'].tolist()  # Substitua 'Segundo' pelo nome da coluna que contém os números

# Dicionário para armazenar os números a serem excluídos para cada número da lista_arredondamento
numeros_para_excluir_por_numero = {numero: [] for numero in lista_arredondamento}

# Verificar quais números devem ser excluídos para cada número na lista_arredondamento
for numero in lista_arredondamento:
    # Encontrar os índices dos números na coluna 'Segundo' que correspondem ao número da lista_arredondamento
    indices_numeros = [i for i, x in enumerate(numeros_originais) if round(x, 3) == numero]
    # Se houver mais de dois índices, escolha aleatoriamente dois deles
    if len(indices_numeros) > 2:
        numeros_excluir = random.sample(indices_numeros, 2)
    else:
        numeros_excluir = indices_numeros
    # Adicionar os números a serem excluídos ao dicionário
    numeros_para_excluir_por_numero[numero].extend(numeros_excluir)

# Converter o dicionário em uma lista de índices para excluir
indices_para_excluir = [indice for indices in numeros_para_excluir_por_numero.values() for indice in indices]

# Excluir os números escolhidos aleatoriamente do dataset
dataset_sem_outlier = dataset.drop(indices_para_excluir)

# Salvar o dataset sem os outliers
caminho_saida = 'Dataset/sem_outlier.csv'
dataset_sem_outlier.to_csv(caminho_saida, index=False)
