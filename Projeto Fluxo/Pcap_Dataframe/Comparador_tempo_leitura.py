import time
import dpkt
import pyshark
import scapy.all as scapy

# Função para ler o dataset e calcular o tempo de leitura
def read_pcapng_dkpt(file_path):
    start_time = time.time()
    # Abra o arquivo .pcapng
    with open(file_path, 'rb') as f:
        pcap = dpkt.pcapng.Reader(f)
        # Itere sobre os pacotes no arquivo .pcapng
        for ts, buf in pcap:
            # Processamento dos pacotes (opcional)
            pass
    end_time = time.time()
    # Calcule o tempo total de leitura
    total_time = end_time - start_time
    return total_time

# Função para ler o dataset e calcular o tempo de leitura
def read_pcapng_scapy(file_path):
    start_time = time.time()
    # Abra o arquivo pcapng com Scapy
    packets = scapy.rdpcap(file_path)
    # Itere sobre os pacotes no arquivo pcapng
    for pkt in packets:
        # Processamento dos pacotes (opcional)
        pass
    end_time = time.time()
    # Calcule o tempo total de leitura
    total_time = end_time - start_time
    return total_time

def read_pcapng_pyshark(file_path):
    start_time = time.time()
    # Abra o arquivo pcapng com PyShark
    cap = pyshark.FileCapture(file_path)
    # Itere sobre os pacotes no arquivo pcapng
    for pkt in cap:
        # Processamento dos pacotes (opcional)
        pass
    end_time = time.time()
    # Calcule o tempo total de leitura
    total_time = end_time - start_time
    return total_time

# Defina o caminho para o arquivo .pcapng
file_path = 'Pcaps/s1_tcpreplay_1.pcapng'

# Chame a função para ler o arquivo .pcapng e calcular o tempo de leitura
total_time = read_pcapng_dkpt(file_path)
# Imprima o tempo total de leitura
print(f"Tempo total de leitura do arquivo .pcapng utilizando dpkt: {total_time: .2f} segundos")

# Chame a função para ler o arquivo pcapng e calcular o tempo de leitura
total_time = read_pcapng_scapy(file_path)
# Imprima o tempo total de leitura
print(f"Tempo total de leitura do arquivo .pcapng utilizando Scapy: {total_time:.2f} segundos")

# Chame a função para ler o arquivo pcapng e calcular o tempo de leitura
total_time = read_pcapng_pyshark(file_path)
# Imprima o tempo total de leitura
print(f"Tempo total de leitura do arquivo .pcapng utilizando PyShark: {total_time:.2f} segundos")



