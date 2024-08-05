import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.utils as utils

# Given functions to generate ellipses
def generate_ellipse(a, b):
    height, width = 64, 64
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    ellipse = ((x - center_x) ** 2 / a ** 2 + (y - center_y) ** 2 / b ** 2) <= 1
    return ellipse.astype(float)

def generate_soft_ellipse(a, b):
    height, width = 64, 64
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    ellipse = ((x - center_x) ** 2 / a ** 2 + (y - center_y) ** 2 / b ** 2) <= 1
    soft_ellipse = gaussian_filter(ellipse.astype(float), sigma=1)
    return soft_ellipse

def add_noise_to_ellipse(image, a, b, noise_level=0.2):
    noisy_image = image.copy()
    height, width = image.shape
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2

    # Create a mask for the ellipse and a bit around it, proportional to its size
    mask = ((x - center_x) ** 2 / (a * 1.5) ** 2 + (y - center_y) ** 2 / (b * 1.5) ** 2) <= 1
    
    # Apply a Gaussian blur to the mask to create a smooth transition
    blurred_mask = gaussian_filter(mask.astype(float), sigma=2)

    # Add Gaussian noise based on the blurred mask
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = np.clip(image + noise * blurred_mask, 0, 1)
    
    return noisy_image

def generate_random_ellipses(num_ellipses=50, noise_level=0.2):
    ellipses = []
    clean_ellipses = []
    for _ in range(num_ellipses):
        a, b = np.random.randint(4, 20, size=2)
        ellipse_image = generate_ellipse(a, b)
        noisy_ellipse = add_noise_to_ellipse(ellipse_image, a, b, noise_level)
        ellipses.append(noisy_ellipse.flatten())
        clean_ellipses.append(ellipse_image.flatten())
    return pd.DataFrame(ellipses), pd.DataFrame(clean_ellipses)

def generate_random_soft_ellipses(num_ellipses=50, noise_level=0.2):
    ellipses = []
    clean_ellipses = []
    for _ in range(num_ellipses):
        a, b = np.random.randint(4, 20, size=2)
        soft_ellipse_image = generate_soft_ellipse(a, b)
        noisy_ellipse = add_noise_to_ellipse(soft_ellipse_image, a, b, noise_level)
        ellipses.append(noisy_ellipse.flatten())
        clean_ellipses.append(soft_ellipse_image.flatten())
    return pd.DataFrame(ellipses), pd.DataFrame(clean_ellipses)

# Function to plot ellipses
def plot_ellipse_from_row(ax, row):
    image = row.values.reshape(64, 64)
    ax.imshow(image, cmap='gray')
    ax.axis('off')

def plot_ellipse_grid(df, grid_size=(3, 3)):
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(df):
            plot_ellipse_from_row(ax, df.iloc[i])
    plt.tight_layout()
    plt.savefig("ellipse_grid.png")

# Dataset class for the ellipses
class EllipseDataset(Dataset):
    def __init__(self, noisy_ellipses, clean_ellipses):
        self.noisy_ellipses = noisy_ellipses
        self.clean_ellipses = clean_ellipses
    
    def __len__(self):
        return len(self.noisy_ellipses)
    
    def __getitem__(self, idx):
        noisy_img = self.noisy_ellipses.iloc[idx].values.reshape(1, 64, 64)
        clean_img = self.clean_ellipses.iloc[idx].values.reshape(1, 64, 64)
        return torch.tensor(noisy_img, dtype=torch.float32), torch.tensor(clean_img, dtype=torch.float32)

# Define the Noise2Noise Model
class Noise2Noise(nn.Module):
    def __init__(self):
        super(Noise2Noise, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Training and Evaluation
def train_and_evaluate(noisy_ellipses, clean_ellipses, epochs=5, filename_prefix="ellipses"):
    dataset = EllipseDataset(noisy_ellipses, clean_ellipses)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Noise2Noise().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Train the model
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            noisy_inputs, clean_targets = data
            noisy_inputs, clean_targets = noisy_inputs.to(device), clean_targets.to(device)

            optimizer.zero_grad()
            outputs = net(noisy_inputs)
            loss = criterion(outputs, clean_targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 200:.4f}')
                running_loss = 0.0

    print('Finished Training')

    # Save the trained model
    torch.save(net.state_dict(), f"{filename_prefix}_model.pth")

    # Evaluate the model
    dataiter = iter(dataloader)
    noisy_images, clean_images = next(dataiter)
    noisy_images = noisy_images[:12]
    clean_images = clean_images[:12]

    net.eval()
    with torch.no_grad():
        denoised_images = net(noisy_images.to(device)).cpu()

    # Show the images
    def imshow(img):
        img = img.numpy()
        plt.imshow(np.transpose(img, (1, 2, 0)), cmap='gray')

    print('Noisy Images')
    imshow(utils.make_grid(noisy_images))

    print('Denoised Images')
    imshow(utils.make_grid(denoised_images))

    print('Clean Images')
    imshow(utils.make_grid(clean_images))

    # Save the result as a PNG file
    fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    axs[0].imshow(np.transpose(utils.make_grid(clean_images, nrow=4, padding=2).numpy(), (1, 2, 0)), cmap='gray')
    axs[0].set_title('Original')
    axs[1].imshow(np.transpose(utils.make_grid(noisy_images, nrow=4, padding=2).numpy(), (1, 2, 0)), cmap='gray')
    axs[1].set_title('Noisy')
    axs[2].imshow(np.transpose(utils.make_grid(denoised_images, nrow=4, padding=2).numpy(), (1, 2, 0)), cmap='gray')
    axs[2].set_title('Denoised')
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_denoising_result.png')

# Generate random ellipses and train the model
noisy_ellipses_df, clean_ellipses_df = generate_random_ellipses(num_ellipses=1000)
noisy_soft_ellipses_df, clean_soft_ellipses_df = generate_random_soft_ellipses(num_ellipses=1000)

print("Training on regular ellipses:")
train_and_evaluate(noisy_ellipses_df, clean_ellipses_df, filename_prefix="ellipses")

print("Training on soft ellipses:")
train_and_evaluate(noisy_soft_ellipses_df, clean_soft_ellipses_df, filename_prefix="soft_ellipses")