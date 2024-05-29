import torch
from torch import nn
import os
import math
import matplotlib.pyplot as plt
from torchvision.utils import save_image

import pandas as pd
import torch

class Processamento:
    """
    Classe para realizar operações de processamento de dados.
    """

    @staticmethod
    def desnormalizar_datasets(diretorio, arquivo, caminho_parametro, colunas):
        """
        Desnormaliza um conjunto de dados.

        Args:
            diretorio (str): O diretório onde os dados estão localizados.
            arquivo (str): O nome do arquivo de dados.
            caminho_parametro (str): O caminho para o arquivo de parâmetros.
            colunas (list): Uma lista de índices das colunas a serem desnormalizadas.

        Returns:
            tuple: Uma tupla contendo o nome do arquivo desnormalizado e o diretório onde foi salvo.
        """
        diretorio_desnormalizado = os.path.join(diretorio, 'desnormalizado')

        if not os.path.exists(diretorio_desnormalizado):
            os.makedirs(diretorio_desnormalizado)

        dataset_parametro = pd.read_csv(caminho_parametro)
        caminho_arquivo = os.path.join(diretorio, arquivo)
        dataset_normalizado = pd.read_csv(caminho_arquivo)
        dataset_normalizado = dataset_normalizado.rename(columns={'0': 'Segundo', '1': 'Bytes_Por_Segundo'})

        for coluna in colunas:
            dataset_normalizado.iloc[:, coluna] = dataset_normalizado.iloc[:, coluna] * (dataset_parametro.iloc[:, coluna].max() - dataset_parametro.iloc[:, coluna].min()) + dataset_parametro.iloc[:, coluna].min()

        arquivo = arquivo.replace('.csv', '_desnormalizado.csv')
        caminho_saida = os.path.join(diretorio_desnormalizado, arquivo)
        dataset_normalizado.to_csv(caminho_saida, index=False)

        return arquivo, diretorio_desnormalizado

    @staticmethod
    def calcular_escala(casas_decimais):
        """
        Calcula a escala com base no número de casas decimais.

        Args:
            casas_decimais (int): O número de casas decimais.

        Returns:
            float: O valor da escala.
        """
        if casas_decimais < 0:
            return 0
        else:
            return 1 / (10 ** casas_decimais)

    @staticmethod
    def escala(arquivo, caminho_dataset_busca, caminho_dataset_salva, escala_int):
        """
        Aplica uma escala aos dados.

        Args:
            arquivo (str): O nome do arquivo de dados.
            caminho_dataset_busca (str): O caminho para os dados originais.
            caminho_dataset_salva (str): O caminho para salvar os dados escalados.
            escala_int (int): O valor da escala.

        Returns:
            tuple: Uma tupla contendo o nome do arquivo escalado e o diretório onde foi salvo.
        """
        escala = Processamento.calcular_escala(escala_int)
        filepath = os.path.join(caminho_dataset_busca, arquivo)
        dataset = pd.read_csv(filepath)
        dataset.iloc[:, 0] = dataset.iloc[:, 0].round(escala_int)
        grouped_data = dataset.groupby(dataset.columns[0])[dataset.columns[1]].sum().reset_index()

        if not os.path.exists(caminho_dataset_salva):
            os.makedirs(caminho_dataset_salva)

        arq = f'escala_{escala}_{arquivo}'
        csv_file_path = os.path.join(caminho_dataset_salva, arq)
        grouped_data.to_csv(csv_file_path, index=False)

        return arq, caminho_dataset_salva

    @staticmethod
    def plot_csv_files(arquivo, directory_path):
        """
        Plota os dados de um arquivo CSV.

        Args:
            arquivo (str): O nome do arquivo CSV.
            directory_path (str): O diretório onde o arquivo está localizado.

        Returns:
            None
        """
        output_directory = os.path.join(directory_path, 'graficos')
        os.makedirs(output_directory, exist_ok=True)
        file_path = os.path.join(directory_path, arquivo)
        data = pd.read_csv(file_path)

        plt.figure(figsize=(10, 6))
        x = data.iloc[:, 0]
        y = data.iloc[:, 1]
        plt.scatter(x, y, c='blue', alpha=0.5, marker='o')
        plt.title(f'Gráfico de Pontos para {arquivo}')
        plt.xlabel(data.columns[0])
        plt.ylabel(data.columns[1])
        output_file_path = os.path.join(output_directory, arquivo.replace('.csv', '.png'))
        plt.savefig(output_file_path)
        plt.close()

    @staticmethod
    def save_samples(generated_samples, epoch, batch_idx, folder="generated_samples"):
        """
        Salva amostras geradas.

        Args:
            generated_samples (tensor): As amostras geradas.
            epoch (int): O número da época.
            batch_idx (int): O índice do batch.
            folder (str): O diretório para salvar as amostras.

        Returns:
            None
        """
        os.makedirs(folder, exist_ok=True)
        save_image(generated_samples, os.path.join(folder, f"epoch_{epoch}_batch_{batch_idx}.png"), normalize=True)

class GAN(nn.Module):
    def __init__(self, amostras):
        super().__init__()
        self.amostras = amostras
        self.discriminator = self.Discriminator()
        self.generator = self.Generator()
        self.lr = 0.0002
        self.loss_function = nn.BCELoss()
        self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)
        self.optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=self.lr)

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

    def train(self, num_epochs, train_loader):
        for epoch in range(num_epochs):
            for n, (real_samples, _) in enumerate(train_loader):
                # Data for training the discriminator
                real_samples_labels = torch.ones((real_samples.size(0), 1))
                latent_space_samples = torch.randn((real_samples.size(0), 2))
                generated_samples = self.generator(latent_space_samples)
                generated_samples_labels = torch.zeros((real_samples.size(0), 1))
                all_samples = torch.cat((real_samples, generated_samples))
                all_samples_labels = torch.cat(
                    (real_samples_labels, generated_samples_labels)
                )

                # Training the discriminator
                self.discriminator.zero_grad()
                output_discriminator = self.discriminator(all_samples)
                loss_discriminator = self.loss_function(
                    output_discriminator, all_samples_labels)
                loss_discriminator.backward()
                self.optimizer_discriminator.step()

                # Data for training the generator
                latent_space_samples = torch.randn((real_samples.size(0), 2))

                # Training the generator
                self.generator.zero_grad()
                generated_samples = self.generator(latent_space_samples)
                output_discriminator_generated = self.discriminator(generated_samples)
                loss_generator = self.loss_function(
                    output_discriminator_generated, real_samples_labels
                )
                loss_generator.backward()
                self.optimizer_generator.step()

                if epoch % 100 == 0 and n == 0:
                    samples_directory = f"Csv_gerados_original"
                    samples_png = f"Csv_gerados_original/Csv_gerados_original_graficos"
                    
                    #save_model(epoch, generator, discriminator, optimizer_generator, optimizer_discriminator)
                    #64 é uma boa
                    latent_space_samples = torch.randn(self.amostras, 2)
                    generated_samples = self.generator(latent_space_samples)
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
                    plt.title(f"Bytes por segundo ao longo do tempo")
                    plt.savefig(os.path.join(samples_png, f"Epoca_{epoch}.png"))
                    plt.close()
                    print(f"Epoch: {epoch} Loss D.: {loss_discriminator} Loss G.: {loss_generator} n {n}")

                    # Desnormalizar Csv gerado
                    caminho_dataset_desmormalizar = 'Dataset_treinamento/10.0.2.2_to_10.0.4.4_normalizado_preenchido_com_zero.csv'
                    colunas = [0, 1]
                    arquivo_gerado_desnormalizado, diretorio_desnormalizado = Processamento.desnormalizar_datasets(samples_directory, arquivo_gerado, caminho_dataset_desmormalizar, colunas)
                    
                    # Plotar Csv gerado desnormalizado
                    Processamento.plot_csv_files(arquivo_gerado_desnormalizado, diretorio_desnormalizado)

                    # Passar Csv gerado para todas as escala e salvar
                    for escala_int in range(5):
                        caminho_datasets_salva = f'{diretorio_desnormalizado}/escala_{Processamento.calcular_escala(escala_int)}'
                        arquivo_gerado_desnormalizado_escalado, diretorio_desnormalizado_escalado = Processamento.escala(arquivo_gerado_desnormalizado, diretorio_desnormalizado, caminho_datasets_salva, escala_int)
                        Processamento.plot_csv_files(arquivo_gerado_desnormalizado_escalado, diretorio_desnormalizado_escalado)

                    print(f'Epoca {epoch} concluída')

# Exemplo de uso
if __name__ == "__main__":
    # Definir outros parâmetros
    torch.manual_seed(111)
    batch_size = 256
    num_epochs = 4100
    amostras = 44513

    gan = GAN(amostras)

    # Carregar o arquivo CSV em um DataFrame
    csv_file_path = 'Datasets_treinamento/10.0.2.2_to_10.0.4.4.csv'  # Substitua 'seu_arquivo.csv' pelo nome do seu arquivo
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

    gan.train(num_epochs, train_loader)