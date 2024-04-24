import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def preencher_com_zeros(caminho_dataset_busca, caminho_dataset_salva, escala_zeros, valor_preenchimento, limite_superior):
    dataset = pd.read_csv(caminho_dataset_busca)

    # Create a range of new seconds from 0 to the smallest existing second, incrementing by 0.0001
    leading_seconds = np.arange(0, dataset.iloc[:, 0].min(), escala_zeros)
    leading_data = pd.DataFrame(leading_seconds, columns=[dataset.columns[0]])
    leading_data[dataset.columns[1]] = valor_preenchimento


    if dataset.iloc[:, 0].max() < limite_superior:
        trailing_seconds = np.arange(dataset.iloc[:, 0].max() + escala_zeros, limite_superior, escala_zeros)  # Avoid overlap with the max_seconds value
        trailing_data = pd.DataFrame(trailing_seconds, columns=[dataset.columns[0]])
        trailing_data[dataset.columns[1]] = valor_preenchimento
    else:
        trailing_data = pd.DataFrame(columns=[dataset.columns[0], dataset.columns[1]])

    # Concatenate the new data with the original, handling leading and trailing zeros
    final_data = pd.concat([leading_data, dataset, trailing_data], ignore_index=True)

    # Sort and reset the DataFrame index
    final_data.sort_values(final_data.columns[0], inplace=True)
    final_data.reset_index(drop=True, inplace=True)

    # Save the updated DataFrame

    final_data.to_csv(caminho_dataset_salva, index=False)

    # Plot the graph
    plt.figure(figsize=(10, 5))
    plt.scatter(final_data.iloc[:, 0], final_data.iloc[:, 1], s=1)
    plt.title('Gráfico Dataset preenchido com zeros')
    plt.xlabel('Tempo')
    plt.ylabel('Bytes Por tempo')
    plt.grid(True)
    plt.show()


caminho_dataset_busca = 'Datasets_treinamento/escala_0.01_10.0.2.2_to_10.0.4.4.csv'
# Removendo a extensão .csv do caminho do arquivo original
caminho_sem_extensao = os.path.splitext(caminho_dataset_busca)[0]
# Adicionando o sufixo '_preenchido_com_zero.csv' ao nome do arquivo sem a extensão
caminho_dataset_salva = caminho_sem_extensao + '_preenchido_com_zero.csv'
escala_zeros = 0.1
valor_preenchimento = 0
limite_superior = 45
preencher_com_zeros(caminho_dataset_busca, caminho_dataset_salva, escala_zeros, valor_preenchimento, limite_superior)