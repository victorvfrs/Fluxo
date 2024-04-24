import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_csv_files(directory_path):
    # Definir o diretório de saída para os gráficos
    output_directory = directory_path + '/graficos'
    os.makedirs(output_directory, exist_ok=True)
    
    # Listar todos os arquivos CSV no diretório
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    
    # Loop para processar cada arquivo CSV
    for file in csv_files:
        # Caminho completo para o arquivo CSV
        file_path = os.path.join(directory_path, file)
        # Ler o arquivo CSV
        data = pd.read_csv(file_path)
        
        # Criar uma figura e um subplot
        plt.figure(figsize=(10, 6))
        
        # Checando se o DataFrame tem ao menos duas colunas para plotagem
        if data.shape[1] < 2:
            print(f"Not enough data to plot: {file}")
            continue
        
        # Suposição: plotar a primeira coluna como X e a segunda coluna como Y
        x = data.iloc[:, 0]
        y = data.iloc[:, 1]
        plt.scatter(x, y, c='blue', alpha=0.5, marker='o')
        
        # Adicionar título e labels aos eixos
        plt.title(f'Gráfico de Pontos para {file}')
        plt.xlabel(data.columns[0])
        plt.ylabel(data.columns[1])
        
        # Salvar o gráfico no diretório de saída
        output_file_path = os.path.join(output_directory, file.replace('.csv', '.png'))
        plt.savefig(output_file_path)
        #plt.show()
        plt.close()

# Exemplo de uso da função:
directory_path = 'Datasets_Base/Escala_10.0.2.2_to_10.0.4.4'
plot_csv_files(directory_path)