import pandas as pd
import os
import matplotlib.pyplot as plt

def normalizar_dataset(caminho_dataset, colunas):
    # Carregar o dataset
    dataset = pd.read_csv(caminho_dataset)
    
    # Normalizar as colunas especificadas
    for coluna in colunas:
        dataset.iloc[:, coluna] = (dataset.iloc[:, coluna] - dataset.iloc[:, coluna].min()) / (dataset.iloc[:, coluna].max() - dataset.iloc[:, coluna].min())
    
    #dataset.iloc[:, 1] = (1)
    # Salvar o novo dataset normalizado
    caminho_saida = caminho_dataset.replace('.csv', '_normalizado.csv')
    dataset.to_csv(caminho_saida, index=False)

    plt.figure(figsize=(10, 5))
    plt.scatter(dataset.iloc[:, 0], dataset.iloc[:, 1], s=1)
    plt.title('Gráfico Dataset Normalizado')
    plt.xlabel('Tempo')
    plt.ylabel('Bytes Por tempo')
    plt.grid(True)
    plt.show()
    

def desnormalizar_datasets(diretorio, caminho_parametro, colunas):
    # Criar o diretório 'desnormalizado' dentro do diretório de entrada, se não existir
    diretorio_desnormalizado = os.path.join(diretorio, 'desnormalizado')
    if not os.path.exists(diretorio_desnormalizado):
        os.makedirs(diretorio_desnormalizado)
    
    # Carregar o dataset de parâmetro
    dataset_parametro = pd.read_csv(caminho_parametro)
    
    # Percorrer todos os arquivos no diretório
    for arquivo in os.listdir(diretorio):
        # Verificar se o arquivo é um CSV
        if arquivo.endswith(".csv"):
            # Construir o caminho completo do arquivo
            caminho_arquivo = os.path.join(diretorio, arquivo)
            
            # Carregar o dataset normalizado
            dataset_normalizado = pd.read_csv(caminho_arquivo)
            
            # Renomear as colunas
            dataset_normalizado = dataset_normalizado.rename(columns={'0': 'Segundo', '1': 'Bytes_Por_Segundo'})
            
            for coluna in colunas:
                dataset_normalizado.iloc[:, coluna] = dataset_normalizado.iloc[:, coluna] * (dataset_parametro.iloc[:, coluna].max() - dataset_parametro.iloc[:, coluna].min()) + dataset_parametro.iloc[:, coluna].min()
             
            # Salvar o dataset desnormalizado no diretório 'desnormalizado'
            caminho_saida = os.path.join(diretorio_desnormalizado, arquivo.replace('.csv', '_desnormalizado.csv'))
            dataset_normalizado.to_csv(caminho_saida, index=False)

            # Plotar o gráfico
            #plt.figure(figsize=(10, 5))
            #plt.scatter(dataset_normalizado.iloc[:, 0], dataset_normalizado.iloc[:, 1], s=1)
            #plt.title('Gráfico Dataset Desnormalizado')
            #plt.xlabel('Tempo')
            #plt.ylabel('Bytes Por tempo')
            #plt.grid(True)
            #plt.show()

# Exemplo de uso das funções
#Coloque o nome do arquivo no qual deseja normalizar 
colunas = [0, 1]
caminho_dataset = 'Datasets_treinamento/escala_0.01_10.0.2.2_to_10.0.4.4_preenchido_com_zero.csv'
#normalizar_dataset(caminho_dataset, colunas)

#Coloque o diretório no qual deseja desnormalizar os datasets 
diretorio_datasets = 'escala_0.01_10.0.2.2_to_10.0.4.4_preenchido_com_zero_normalizado_csv_A_2702_B_64_E_1100'
#Coloque o caminho do arquivo base para desnormalização
desnormalizar_datasets(diretorio_datasets, caminho_dataset, colunas)


