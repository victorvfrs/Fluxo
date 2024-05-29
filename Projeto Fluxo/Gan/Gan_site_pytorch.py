import torch
from torch import nn
import os
import math
import matplotlib.pyplot as plt
from torchvision.utils import save_image

import pandas as pd
import torch

def desnormalizar_datasets(diretorio, arquivo, caminho_parametro, colunas):
    # Criar o diretório 'desnormalizado' dentro do diretório de entrada, se não existir
    diretorio_desnormalizado = os.path.join(diretorio, 'desnormalizado')
    
    if not os.path.exists(diretorio_desnormalizado):
        os.makedirs(diretorio_desnormalizado)
    
    # Carregar o dataset de parâmetro
    dataset_parametro = pd.read_csv(caminho_parametro)
    
    # Percorrer todos os arquivos no diretório

    # Construir o caminho completo do arquivo
    caminho_arquivo = os.path.join(diretorio, arquivo)
            
    # Carregar o dataset normalizado
    dataset_normalizado = pd.read_csv(caminho_arquivo)
            
    # Renomear as colunas
    dataset_normalizado = dataset_normalizado.rename(columns={'0': 'Segundo', '1': 'Bytes_Por_Segundo'})
            
    for coluna in colunas:
        dataset_normalizado.iloc[:, coluna] = dataset_normalizado.iloc[:, coluna] * (dataset_parametro.iloc[:, coluna].max() - dataset_parametro.iloc[:, coluna].min()) + dataset_parametro.iloc[:, coluna].min()
             
            # Salvar o dataset desnormalizado no diretório 'desnormalizado'
    arquivo = arquivo.replace('.csv', '_desnormalizado.csv')
    caminho_saida = os.path.join(diretorio_desnormalizado, arquivo)
    dataset_normalizado.to_csv(caminho_saida, index=False)

    return arquivo, diretorio_desnormalizado

def calcular_escala(casas_decimais):
    if casas_decimais < 0:
        return 0
    else:
        return 1 / (10 ** casas_decimais)

def escala(arquivo, caminho_dataset_busca, caminho_dataset_salva, escala_int):
    escala = calcular_escala(escala_int)
    filepath = os.path.join(caminho_dataset_busca, arquivo)
    dataset = pd.read_csv(filepath)



    # Arredondamento da coluna x
    dataset.iloc[:, 0] = dataset.iloc[:, 0].round(escala_int)

    # Agrupar os dados por coluna_x e somar os valores de 'Bytes_Por_Segundo' nesses grupos
    grouped_data = dataset.groupby(dataset.columns[0])[dataset.columns[1]].sum().reset_index()

    # Criar uma coluna 'Quantidade_pacotes' que é calculada dividindo 'Bytes_Por_Segundo' por 1444
    #grouped_data['Quantidade_pacotes'] = (grouped_data.iloc[:, 1] / 1444).astype(int)

    # Verifique se o diretório existe, senão crie-o
    if not os.path.exists(caminho_dataset_salva):
        os.makedirs(caminho_dataset_salva)

    arq = f'escala_{escala}_{arquivo}'
    # Caminho completo para o arquivo CSV
    csv_file_path = os.path.join(caminho_dataset_salva, arq)

    # Salve o dataset como um arquivo CSV
    grouped_data.to_csv(csv_file_path, index=False)

    return arq, caminho_dataset_salva

def plot_csv_files(arquivo, directory_path):
    # Definir o diretório de saída para os gráficos
    output_directory = directory_path + '/graficos'
    os.makedirs(output_directory, exist_ok=True)

    # Caminho completo para o arquivo CSV
    file_path = os.path.join(directory_path, arquivo)
    # Ler o arquivo CSV
    data = pd.read_csv(file_path)
        
    # Criar uma figura e um subplot
    plt.figure(figsize=(10, 6))
        
    # Checando se o DataFrame tem ao menos duas colunas para plotagem
    if data.shape[1] < 2:
        print(f"Not enough data to plot: {arquivo}")
        
        # Suposição: plotar a primeira coluna como X e a segunda coluna como Y
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    plt.scatter(x, y, c='blue', alpha=0.5, marker='o')
        
        # Adicionar título e labels aos eixos
    plt.title(f'Gráfico de Pontos para {arquivo}')
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
        
        # Salvar o gráfico no diretório de saída
    output_file_path = os.path.join(output_directory, arquivo.replace('.csv', '.png'))
    plt.savefig(output_file_path)
    #plt.show()
    plt.close()

def save_samples(generated_samples, epoch, batch_idx, folder="generated_samples"):
    # Cria o diretório se não existir
    os.makedirs(folder, exist_ok=True)
    # Salva as imagens
    save_image(generated_samples,
               os.path.join(folder, f"epoch_{epoch}_batch_{batch_idx}.png"),
               normalize=True)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        output = self.model(x)
        return output


torch.manual_seed(111)
batch_size = 256
num_epochs = 4100
amostras = 4000

# Carregar o arquivo CSV em um DataFrame
csv_file_path = 'Datasets_treinamento/distribuicao_normal_normalizado.csv'  # Substitua 'seu_arquivo.csv' pelo nome do seu arquivo
dataset = pd.read_csv(csv_file_path)

# Inicializar o tensor PyTorch
train_data_length = len(dataset)
train_data = torch.zeros((train_data_length, 2))

# Transferir dados para o tensor
train_data[:, 0] = torch.tensor(dataset.iloc[:, 0].values)  # Primeira coluna para train_data[:, 0]
train_data[:, 1] = torch.tensor(dataset.iloc[:, 1].values)  # Segunda coluna para train_data[:, 1]
train_labels = torch.zeros(train_data_length)
train_set = [
    (train_data[i], train_labels[i]) for i in range(train_data_length)
]


train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, drop_last=True
)

generator = Generator()

discriminator = Discriminator()

lr = 0.0002

loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)


for epoch in range(num_epochs):
    for n, (real_samples, _) in enumerate(train_loader):
        # Data for training the discriminator
        real_samples_labels = torch.ones((batch_size, 1))
        latent_space_samples = torch.randn((batch_size, 2))
        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1))
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels)
        )

        # Training the discriminator
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(
            output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Data for training the generator
        latent_space_samples = torch.randn((batch_size, 2))

        # Training the generator
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels
        )
        loss_generator.backward()
        optimizer_generator.step()
        #print(f"Epoch: {epoch}/n {n} Loss D.: {loss_discriminator} Loss G.: {loss_generator}")
            
        # Show loss
        if epoch % 100 == 0 and n == 0:
            samples_directory = f"Csv_gerados_original"
            samples_png = f"Csv_gerados_original/Csv_gerados_original_graficos"
            
            #save_model(epoch, generator, discriminator, optimizer_generator, optimizer_discriminator)
            #64 é uma boa
            latent_space_samples = torch.randn(amostras, 2)
            generated_samples = generator(latent_space_samples)
            generated_samples = generated_samples.detach()
            # Verifique se o diretório existe, senão crie-o

            if not os.path.exists(samples_directory):
                os.makedirs(samples_directory)

            # Salve o dataset como um arquivo CSV
            df = pd.DataFrame(generated_samples.detach().numpy())
            
            arquivo_gerado = f"Epoca_{epoch}.csv"
            # Caminho completo para o arquivo CSV
            csv_file_path = os.path.join(samples_directory, arquivo_gerado)

            # Salvar o DataFrame em um arquivo CSV
            df.to_csv(csv_file_path, index=False)

            if not os.path.exists(samples_png):
                os.makedirs(samples_png)
            plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
            plt.title(f"Intervalos X Densidade de probabilidade")
            plt.savefig(os.path.join(samples_png, f"Epoca_{epoch}.png"))
            plt.close()
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator} Loss G.: {loss_generator} n {n}")

            caminho_dataset_desmormalizar = 'Datasets_treinamento/distribuicao_normal.csv'
            colunas = [0, 1]
            arquivo_gerado_desnormalizado, diretorio_desnormalizado = desnormalizar_datasets(samples_directory, arquivo_gerado, caminho_dataset_desmormalizar, colunas)
            plot_csv_files(arquivo_gerado_desnormalizado, diretorio_desnormalizado)

            #for escala_int in range(5):
            #    caminho_datasets_salva = f'{diretorio_desnormalizado}/escala_{calcular_escala(escala_int)}'
            #    arquivo_gerado_desnormalizado_escalado, diretorio_desnormalizado_escalado = escala(arquivo_gerado_desnormalizado, diretorio_desnormalizado, caminho_datasets_salva, escala_int)
            #    plot_csv_files(arquivo_gerado_desnormalizado_escalado, diretorio_desnormalizado_escalado)

            #print(f'Epoca {epoch} concluída')


            #print(f"Epoch: {epoch} Loss G.: {loss_generator}")

