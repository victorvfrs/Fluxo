import os
import pandas as pd
import matplotlib.pyplot as plt

# Diretório dos arquivos CSV
dir_path = 'datasets_treinamento'
csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]

# Iterar sobre cada arquivo CSV
for csv_file in csv_files:
    csv_file_path = os.path.join(dir_path, csv_file)
    # Ler o arquivo CSV
    df = pd.read_csv(csv_file_path)
    
    # Plotar o histograma
    plt.figure(figsize=(10, 6))
    plt.hist(df['x'], bins=20, color='blue', alpha=0.5)
    plt.title(f'Histograma - {csv_file}')
    plt.xlabel('Valores')
    plt.ylabel('Frequência')

    # Diretório para salvar o gráfico
    graph_dir_path = os.path.join(dir_path, 'graficos')
    if not os.path.exists(graph_dir_path):
        os.makedirs(graph_dir_path)

    # Caminho do arquivo PNG
    graph_file_path = os.path.join(graph_dir_path, f'histograma_{csv_file[:-4]}.png')

    # Salvar o gráfico como arquivo PNG
    plt.savefig(graph_file_path)

    # Exibir mensagem de conclusão
    print(f'Histograma para {csv_file} salvo em {graph_file_path}')

    # Limpar o plot para o próximo arquivo
    plt.clf()

