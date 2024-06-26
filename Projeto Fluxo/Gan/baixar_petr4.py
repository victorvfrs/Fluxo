import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split

def download_stock_data(ticker):
    # Baixa todos os dados disponíveis do Yahoo Finance
    stock_data = yf.download(ticker)
    return stock_data

def save_data_to_csv(data, filename):
    # Salva os dados em um arquivo CSV
    data.to_csv(filename, index=False)
    print(f"Dados salvos como {filename}")

ticker = 'PETR4.SA'
    
# Baixa todos os dados disponíveis para o ticker
stock_data = download_stock_data(ticker)
    
# Divide os dados em 70% para treinamento e 30% para teste sem embaralhar
train_size = int(len(stock_data) * 0.7)
train_data = stock_data[:train_size]
test_data = stock_data[train_size:len(stock_data) - 20]
final_data = stock_data[len(stock_data) - 20:]
    
# Salva os dados em arquivos CSV
save_data_to_csv(train_data, 'petr4_treinamento.csv')
save_data_to_csv(test_data, 'petr4_teste.csv')
save_data_to_csv(final_data, 'petr4_final.csv')    
