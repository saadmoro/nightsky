import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


#Given a width and a height, generate 
def generate_ellipse(a,b):
    height, width = 64,64
    y,x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width //2
    ellipse = ((x - center_x) ** 2 / a ** 2 + (y - center_y) ** 2 / b ** 2) <= 1
    return ellipse.astype(float)

def generate_soft_ellipse(a, b):
    height, width = 64, 64
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    ellipse = ((x - center_x) ** 2 / a ** 2 + (y - center_y) ** 2 / b ** 2) <= 1
    soft_ellipse = gaussian_filter(ellipse.astype(float), sigma=1)
    return soft_ellipse

def plot_ellipse_from_row(ax, row):
    image = row.values.reshape(64, 64)
    ax.imshow(image, cmap = 'gray')
    ax.axis('off')

def plot_ellipse_grid(df, grid_size = (3,3)):
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(10,10))
    for i, ax in enumerate(axes.flat):
        if i < len(df):
            plot_ellipse_from_row(ax, df.iloc[i])
    plt.tight_layout()
    plt.savefig("ellipse_grid.png")


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
    for _ in range(num_ellipses):
        a, b = np.random.randint(1, 15, size=2)
        ellipse_image = generate_ellipse(a, b)
        noisy_ellipse = add_noise_to_ellipse(ellipse_image, a, b, noise_level)
        ellipses.append(noisy_ellipse.flatten())
    return pd.DataFrame(ellipses)

def generate_random_soft_ellipses(num_ellipses=50, noise_level=0.2):
    ellipses = []
    for _ in range(num_ellipses):
        a, b = np.random.randint(1, 15, size=2)
        soft_ellipse_image = generate_soft_ellipse(a, b)
        #noisy_ellipse = add_noise_to_ellipse(soft_ellipse_image, a, b, noise_level)
        #ellipses.append(noisy_ellipse.flatten())
        ellipses.append(soft_ellipse_image.flatten())
    return pd.DataFrame(ellipses)

#Does the same as above, but using a DataFrame as an argument instead
def generate_noisy_ellipses(df, noise_level=0.2):
    ellipses = []
    for _, row in df.iterrows():
        a, b = row['a'], row['b']
        soft_ellipse_image = generate_soft_ellipse(a, b)
        noisy_ellipse = add_noise_to_ellipse(soft_ellipse_image, a, b, noise_level)
        ellipses.append(noisy_ellipse.flatten())
    return pd.DataFrame(ellipses)



###### Generate 50 random ellipses
np.random.seed(42)
ellipses_df = generate_random_soft_ellipses()

plot_ellipse_grid(ellipses_df.iloc[:9])