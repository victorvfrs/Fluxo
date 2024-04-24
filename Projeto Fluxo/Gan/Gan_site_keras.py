import tensorflow as tf
import math
import matplotlib.pyplot as plt
import os

tf.random.set_seed(111)

# Create the dataset
train_data_length = 1024
theta = tf.random.uniform((train_data_length, 1), minval=0, maxval=2 * math.pi)
sin_values = tf.math.sin(theta)
train_data = tf.concat([theta, sin_values], axis=1)
train_labels = tf.zeros((train_data_length, 1))

# Plot data
plt.plot(train_data[:, 0], train_data[:, 1], ".")
plt.show()

# Prepare the DataLoader
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=train_data_length).batch(32)

# Define the models
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    
    def call(self, inputs):
        return self.model(inputs)

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(2)
        ])
    
    def call(self, inputs):
        return self.model(inputs)

generator = Generator()
discriminator = Discriminator()

# Optimizers
optimizer_discriminator = tf.keras.optimizers.Adam(learning_rate=0.001)
optimizer_generator = tf.keras.optimizers.Adam(learning_rate=0.001)

# Loss function
loss_function = tf.keras.losses.BinaryCrossentropy()

# Training loop
num_epochs = 300
for epoch in range(num_epochs):
    batch_count = 0
    for real_samples, _ in train_dataset:
        batch_size = real_samples.shape[0]

        # Train Discriminator
        with tf.GradientTape() as disc_tape:
            latent_space_samples = tf.random.normal((batch_size, 2))
            generated_samples = generator(latent_space_samples)
            real_samples_labels = tf.ones((batch_size, 1))
            generated_samples_labels = tf.zeros((batch_size, 1))
            
            real_output = discriminator(real_samples)
            generated_output = discriminator(generated_samples)
            
            disc_loss_real = loss_function(real_samples_labels, real_output)
            disc_loss_generated = loss_function(generated_samples_labels, generated_output)
            disc_loss = disc_loss_real + disc_loss_generated
        
        gradients_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        optimizer_discriminator.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

        # Train Generator
        with tf.GradientTape() as gen_tape:
            latent_space_samples = tf.random.normal((batch_size, 2))
            generated_samples = generator(latent_space_samples)
            discriminator_output = discriminator(generated_samples)
            gen_loss = loss_function(real_samples_labels, discriminator_output)
        
        gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
        optimizer_generator.apply_gradients(zip(gradients_gen, generator.trainable_variables))
        
        batch_count += 1
        print(f"Batch: {batch_count}/ Epoch: {epoch} Loss D.: {disc_loss.numpy()}, Loss G.: {gen_loss.numpy()}")

# Directory for saving samples
samples_directory = "samples_da_gan_tf"
os.makedirs(samples_directory, exist_ok=True)

# Generate and save 10 samples
for i in range(10):
    latent_space_samples = tf.random.normal((100, 2))
    generated_samples = generator(latent_space_samples)
    plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
    plt.title(f"Sample {i+1}")
    plt.savefig(os.path.join(samples_directory, f"sample_{i+1}.png"))
    plt.close()