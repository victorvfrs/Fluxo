import os
import pandas as pd

import os
import pandas as pd

def calcular_escala(casas_decimais):
    if casas_decimais < 0:
        return 0
    else:
        return 1 / (10 ** casas_decimais)

def escala(caminho_dataset_busca, caminho_dataset_salva, escala_int):
    dataframes = []
    escala = calcular_escala(escala_int)

    for filename in os.listdir(caminho_dataset_busca):
        if filename.endswith('.csv'):
            filepath = os.path.join(caminho_dataset_busca, filename)
            df = pd.read_csv(filepath)
            dataframes.append((filename, df))

    for filename, dataset in dataframes:
        # Arredondamento da coluna x
        dataset.iloc[:, 0] = dataset.iloc[:, 0].round(escala_int)

        # Agrupar os dados por coluna_x e somar os valores de 'Bytes_Por_Segundo' nesses grupos
        grouped_data = dataset.groupby(dataset.columns[0])[dataset.columns[1]].sum().reset_index()

        # Criar uma coluna 'Quantidade_pacotes' que é calculada dividindo 'Bytes_Por_Segundo' por 1444
        #grouped_data['Quantidade_pacotes'] = (grouped_data.iloc[:, 1] / 1444).astype(int)

        # Verifique se o diretório existe, senão crie-o
        if not os.path.exists(caminho_dataset_salva):
            os.makedirs(caminho_dataset_salva)

        # Caminho completo para o arquivo CSV
        csv_file_path = os.path.join(caminho_dataset_salva, f'escala_{escala}_{filename}')

        # Salve o dataset como um arquivo CSV
        grouped_data.to_csv(csv_file_path, index=False)


caminho_datasets_busca = 'escala_0.01_10.0.2.2_to_10.0.4.4_preenchido_com_zero_normalizado_csv_A_2702_B_64_E_1100/desnormalizado'

for escala_int in range(2):
    caminho_datasets_salva = f'{caminho_datasets_busca}/escala_{calcular_escala(escala_int)}'
    escala(caminho_datasets_busca, caminho_datasets_salva, escala_int)

