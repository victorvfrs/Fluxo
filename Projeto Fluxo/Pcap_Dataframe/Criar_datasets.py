import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from scapy.all import rdpcap

# Função para extrair os campos de interesse e criar o dataset
def create_dataset(file_path, directory, csv_file_name):
    # Use rdpcap para ler o arquivo pcapng e obter uma lista de pacotes
    packets = rdpcap(file_path)
    
    # Se o arquivo estiver vazio, retornar um DataFrame vazio
    if not packets:
        print('Sem pacotes no pcap')
    
    # Selecione o primeiro pacote como referência
    reference_packet = packets[0]
    
    # Obtenha o timestamp do primeiro pacote como referência
    reference_time = reference_packet.time
    
    # Lista para armazenar os dados dos pacotes
    dataset = []
    
    # Iterar sobre os pacotes
    for pkt in packets:
        # Verificar se o pacote possui a camada IP
        if 'IP' in pkt:
            # Extrair os campos de interesse
            src_ip = pkt['IP'].src
            dst_ip = pkt['IP'].dst
            proto = pkt['IP'].proto
            length = len(pkt)
            time = pkt.time  # Captura o tempo do pacote
            
            # Verificar se o pacote possui camada de transporte (TCP ou UDP)
            if 'TCP' in pkt:
                src_port = pkt['TCP'].sport
                dst_port = pkt['TCP'].dport
            elif 'UDP' in pkt:
                src_port = pkt['UDP'].sport
                dst_port = pkt['UDP'].dport
            else:
                src_port = None
                dst_port = None
            
            # Calcule o tempo desde a referência ou do primeiro quadro
            time_since_reference = time - reference_time
            
            # Adicionar os dados do pacote ao dataset
            dataset.append({'Source IP': src_ip, 'Destination IP': dst_ip,
                            'Source Port': src_port, 'Destination Port': dst_port,
                            'Protocol': proto, 'Length': length,
                            'Time Since Reference': time_since_reference})

    # Verifique se a pasta "datasets" existe, senão crie-a
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Caminho completo para o arquivo CSV dentro da pasta "datasets"
    csv_file_path = os.path.join(directory, csv_file_name)

    dataset = pd.DataFrame(dataset)
    # Salvar o dataset em um arquivo CSV dentro da pasta "datasets"
    dataset.to_csv(csv_file_path, index=False)

def criar_dataset_trafego(csv_file_path, directory, csv_file_name, round):

    dataset = pd.read_csv(csv_file_path)
    # Arredonde o tempo desde a referência ou o primeiro quadro para o segundo mais próximo
    if round != -1:
        dataset['Time Since Reference'] = dataset['Time Since Reference'].round(round)

    # Agrupe os dados por segundo e some as lengths dos pacotes em cada segundo
    grouped_data = dataset.groupby('Time Since Reference')['Length'].sum()

    # Criar um DataFrame com os valores de Xlabel (Tempo em segundos) e Ylabel (Bytes por segundo)
    df = pd.DataFrame({'Segundo': grouped_data.index, 'Bytes_Por_Segundo': grouped_data.values})

    # Verifique se o diretório existe, senão crie-o
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Caminho completo para o arquivo CSV
    csv_file_path = os.path.join(directory, csv_file_name)

    # Salve o dataset como um arquivo CSV
    df.to_csv(csv_file_path, index=False)

def criar_dataset_fluxo(csv_file_path, directory, round):
    # Carregar o arquivo CSV em um DataFrame
    dataset = pd.read_csv(csv_file_path)
    if round != -1:
        dataset['Time Since Reference'] = dataset['Time Since Reference'].round(round)
    grouped = dataset.groupby(['Source IP', 'Destination IP', 'Time Since Reference'])['Length'].sum()
    grouped = grouped.reset_index()
    unique_pairs = grouped[['Source IP', 'Destination IP']].drop_duplicates()

    if not os.path.exists(directory):
        os.makedirs(directory)

    print(f'Número total de pares únicos: {len(unique_pairs)}')

    # Iterar sobre cada par único e salvar em um arquivo CSV
    for index, (source_ip, destination_ip) in enumerate(unique_pairs.values, start=1):
        # Filtrar os dados para cada par único e criar uma cópia independente
        df_subset = grouped[(grouped['Source IP'] == source_ip) & (grouped['Destination IP'] == destination_ip)].copy()

        # Arredonde o tempo desde a referência para o segundo mais próximo diretamente no df_subset usando .loc
        if round != -1:
            df_subset.loc[:, 'Time Since Reference'] = df_subset['Time Since Reference'].round(round)

        # Agrupe os dados por segundo e some as lengths dos pacotes em cada segundo
        grouped_data = df_subset.groupby('Time Since Reference')['Length'].sum()

        # Criar um DataFrame com os valores de Segundo (Tempo em segundos) e Bytes por segundo
        df = pd.DataFrame({
            'Segundo': grouped_data.index,
            'Bytes_Por_Segundo': grouped_data.values
        })

        # Criar nome do arquivo baseado nos IPs
        filename = f'{source_ip}_to_{destination_ip}.csv'
        filepath = os.path.join(directory, filename)  # substitui pontos por underscores para compatibilidade de nomes de arquivo

        # Salve o DataFrame final como um arquivo CSV
        df.to_csv(filepath, index=False)


# Criar o dataset
file_path = 'Pcaps/s1_tcpreplay_1.pcapng'
directory_base_csv = 'Dataset_Base'
directory = 'Datasets_csv'
csv_file_path = 'Dataset_Base/tupla_5_s1_tcpreplay_1.csv'
csv_file_name1 = 'tupla_5_s1_tcpreplay_1.csv'
csv_file_name2 = 'Trafego.csv'
escala = -1

#create_dataset(file_path, directory_base_csv, csv_file_name1)
criar_dataset_trafego(csv_file_path, directory, csv_file_name2, escala)
criar_dataset_fluxo(csv_file_path, directory, escala)
