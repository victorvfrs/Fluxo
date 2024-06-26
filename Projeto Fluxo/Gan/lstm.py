import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler

# Carrega os dados de treinamento
base = pd.read_csv('petr4_treinamento.csv')
base = base.dropna()
base_treinamento = base.iloc[:, 0:1].values

# Normaliza os dados de treinamento
normalizador = MinMaxScaler(feature_range=(0,1))
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)

# Prepara os dados para o treinamento
previsores = []
preco_real = []
for i in range(90, len(base_treinamento_normalizada)):
    previsores.append(base_treinamento_normalizada[i-90:i, 0])
    preco_real.append(base_treinamento_normalizada[i, 0])
previsores, preco_real = np.array(previsores), np.array(preco_real)
previsores = np.reshape(previsores, (previsores.shape[0], previsores.shape[1], 1))

# Cria e compila o modelo LSTM
regressor = Sequential()
regressor.add(LSTM(units=100, return_sequences=True, input_shape=(previsores.shape[1], 1)))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.3))
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.3))
regressor.add(Dense(units=1, activation='linear'))
regressor.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Treina o modelo
regressor.fit(previsores, preco_real, epochs=20, batch_size=32)

# Salva o modelo e os pesos
model_json = regressor.to_json()
with open("modelo_lstm.json", "w") as json_file:
    json_file.write(model_json)
regressor.save_weights("modelo_lstm.weights.h5")

# Carrega os dados de teste
base_teste = pd.read_csv('petr4_teste.csv')
preco_real_teste = base_teste.iloc[:, 0:1].values

# Prepara os dados de entrada para as previsões
base_completa = pd.concat((base['Open'], base_teste['Open']), axis=0)
entradas = base_completa[(len(base_treinamento)  - 90):].values
entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)

X_teste = []
for i in range(90, len(entradas)):
    X_teste.append(entradas[i-90:i, 0])
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))

# Faz previsões
previsoes = regressor.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes)

# Como `X_teste` foi criado a partir de uma janela deslizante de 90 dias, precisamos garantir que os preços reais do teste estejam alinhados
preco_real_teste_periodo = preco_real_teste[-len(previsoes):]

# Plotar previsões e preços reais de teste para o período completo
plt.figure(figsize=(14, 7))
plt.plot(preco_real_teste_periodo, color='red', label='Preço Real')
plt.plot(previsoes, color='blue', label='Previsões')
plt.title('Previsão do Preço das Ações')
plt.xlabel('Tempo')
plt.ylabel('Preço das Ações')
plt.legend()

# Cria diretório se não existir
if not os.path.exists("LSTM/PETR4"):
    os.makedirs("LSTM/PETR4")

# Salva o gráfico no diretório especificado
plt.savefig("LSTM/PETR4/Gerado_Lstm_PETR4.png")
plt.show()

