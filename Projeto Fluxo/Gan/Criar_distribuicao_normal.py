import numpy as np
import pandas as pd
import os

def gerar_histograma_csv(diretorio, media, desvio_padrao, num_valores, num_bins):
    # Gerando uma distribuição normal
    valores = np.random.normal(media, desvio_padrao, num_valores)

    # Calculando o histograma
    densidades, bins = np.histogram(valores, bins=num_bins, density=True)

    # Calculando os pontos médios dos intervalos (eixo x)
    intervalos = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]

    # Criando um DataFrame com os valores dos eixos x e y
    dados = {'Intervalos': intervalos, 'Densidade de Probabilidade': densidades}
    df = pd.DataFrame(dados)

    # Salvando os dados em um arquivo CSV
    if not os.path.exists(diretorio):
        os.makedirs(diretorio)
    arquivo_csv = os.path.join(diretorio, 'distribuicao_normal.csv')
    df.to_csv(arquivo_csv, index=False)

    return arquivo_csv

# Exemplo de uso da função
diretorio = 'datasets_treinamento'
media = 0
desvio_padrao = 1
num_valores = 40000
num_bins = 40000
arquivo_csv = gerar_histograma_csv(diretorio, media, desvio_padrao, num_valores, num_bins)
print(f'Arquivo CSV salvo em: {arquivo_csv}')












