import torch
from torch import nn
import os
import math
import matplotlib.pyplot as plt
from torchvision.utils import save_image

import pandas as pd
import torch



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
batch_size = 64
num_epochs = 1100
amostras = 2702

# Carregar o arquivo CSV em um DataFrame
csv_file_path = 'Datasets_treinamento/escala_0.01_10.0.2.2_to_10.0.4.4_preenchido_com_zero_normalizado.csv'  # Substitua 'seu_arquivo.csv' pelo nome do seu arquivo
dataset = pd.read_csv(csv_file_path)

# Inicializar o tensor PyTorch
train_data_length = len(dataset)
train_data = torch.zeros((train_data_length, 2))
print(train_data_length)
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
            samples_directory = f"escala_0.01_10.0.2.2_to_10.0.4.4_preenchido_com_zero_normalizado_csv_A_{amostras}_B_{batch_size}_E_{num_epochs}"
            samples_png = f"escala_0.01_10.0.2.2_to_10.0.4.4_preenchido_com_zero_normalizado_png_A_{amostras}_B_{batch_size}_E_{num_epochs}"
            
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

            # Caminho completo para o arquivo CSV
            csv_file_path = os.path.join(samples_directory, f"Epoca_{epoch}.csv")

            # Salvar o DataFrame em um arquivo CSV
            df.to_csv(csv_file_path, index=False)

            if not os.path.exists(samples_png):
                os.makedirs(samples_png)
            plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
            plt.title(f"Bytes por segundo ao longo do tempo")
            plt.savefig(os.path.join(samples_png, f"Epoca_{epoch}.png"))
            plt.close()
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator} Loss G.: {loss_generator} n {n}")
            #print(f"Epoch: {epoch} Loss G.: {loss_generator}")

