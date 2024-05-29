import os
import pandas as pd
import matplotlib.pyplot as plt

def salvar_histogramas(diretorio_csv, diretorio_histogramas):
    # Garantindo que o diretório de histogramas existe
    if not os.path.exists(diretorio_histogramas):
        os.makedirs(diretorio_histogramas)

    # Listando todos os arquivos CSV no diretório especificado
    arquivos_csv = [f for f in os.listdir(diretorio_csv) if f.endswith('.csv')]

    for arquivo_csv in arquivos_csv:
        caminho_arquivo_csv = os.path.join(diretorio_csv, arquivo_csv)
        
        # Carregando os dados do arquivo CSV
        dados = pd.read_csv(caminho_arquivo_csv)
        
        # Usando iloc para definir os dados de 'Intervalos' e 'Densidade de Probabilidade'
        intervalos = dados.iloc[:, 0]
        densidade_probabilidade = dados.iloc[:, 1]
        
        # Criando uma figura para o histograma
        plt.figure(figsize=(8, 6))
        plt.hist(intervalos, weights=densidade_probabilidade, bins=30, color='g', alpha=0.6)
        plt.title('Histograma')
        plt.xlabel('Intervalos')
        plt.ylabel('Densidade de Probabilidade')

        # Salvando o histograma como arquivo PNG
        nome_arquivo_histograma = os.path.splitext(arquivo_csv)[0] + '_histograma.png'
        caminho_arquivo_histograma = os.path.join(diretorio_histogramas, nome_arquivo_histograma)
        plt.savefig(caminho_arquivo_histograma)
        plt.close()


# Exemplo de uso da função
diretorio_csv = 'Csv_gerados_original/desnormalizado'
diretorio_histogramas = 'Csv_gerados_original/desnormalizado/histogramas'
salvar_histogramas(diretorio_csv, diretorio_histogramas)


