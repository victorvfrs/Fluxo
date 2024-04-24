import pandas as pd

# Caminho do arquivo
file_path = 'Datasets_treinamento/escala_0.01_10.0.2.2_to_10.0.4.4.csv'

# Carregar o dataset
df = pd.read_csv(file_path)

# Filtrar os dados
# Condições: Valores acima de 27000 ou entre 23000 e 25000 na coluna 'Bytes_Por_Segundo'
condition = (df['Quantidade_pacotes'] < 18) | (df['Quantidade_pacotes'] > 18)
filtered_df = df[~condition]

file_path_save = 'Datasets_treinamento/escala_0.01_sem_outlier2_10.0.2.2_to_10.0.4.4.csv'
# Salvar o dataset filtrado no mesmo diretório com o mesmo nome
filtered_df.to_csv(file_path_save, index=False)
