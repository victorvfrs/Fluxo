import pandas as pd
import os

# Caminho para o arquivo CSV
caminho_arquivo = 'Datasets_treinamento/escala_0.01_10.0.2.2_to_10.0.4.4.csv'

# Carregar o dataset
dataset = pd.read_csv(caminho_arquivo)

# Encontrar os índices dos valores na coluna 'Quantidade_pacotes' que são > 2
indices = dataset.index[dataset['Quantidade_pacotes'] <= 18].tolist()

# Criar um novo dataset apenas com os valores correspondentes aos índices encontrados
outliers_dataset = dataset.loc[indices]

# Verificar se o diretório 'Datasets_outliers' existe, se não, criar
diretorio_saida = 'Datasets_treinamento'
if not os.path.exists(diretorio_saida):
    os.makedirs(diretorio_saida)

# Nome do arquivo para salvar o dataset filtrado
nome_arquivo_saida = 'sem_outlier_escala_0.01_outliers_10.0.2.2_to_10.0.4.4.csv'
caminho_arquivo_saida = os.path.join(diretorio_saida, nome_arquivo_saida)

# Salvar o novo dataset
outliers_dataset.to_csv(caminho_arquivo_saida, index=False)