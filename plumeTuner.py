import numpy as np
import pandas as pd
import torch
from genEllipse import generate_ellipse, generate_noisy_ellipses
from ellipseNoise2Noise import Noise2Noise
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

class geneticPopulation:
    def __init__(self, pop_size=40, min=1, max=32):
        self.pop_size = pop_size
        self.max = max
        self.min = min
        self.training_epochs = 0

        init_a = np.random.uniform(self.min, self.max, pop_size)
        init_b = np.random.uniform(self.min, self.max, self.pop_size)

        self.population = pd.DataFrame({
            "a": init_a,
            "b": init_b
        })
        
        self.population["Fitness"] = 0

        # Initialize the Noise2Noise model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Noise2Noise().to(self.device)
        self.model.load_state_dict(torch.load('./soft_ellipses_model.pth'))  # Load your trained model
        self.model.eval()

        # Initialize global fitness bounds
        self.global_min_fitness = float('inf')
        self.global_max_fitness = float('-inf')

    def get_fitness(self, target):
        self.assess_fitness(self.population, target)

    def selection(self):
        self.population = self.population.sort_values(
            by='Fitness', ascending=True).reset_index(drop=True)

        #Visualize results at this stage
        self.plot_results(target)
        self.plot_population_evolution()

        parents = self.population.iloc[:len(self.population) // 2]

        children = []
        for i, row in parents.iterrows():
            parent1_a = row['a']
            parent1_b = row['b']
            while True:
                other_idx = np.random.randint(0, len(parents))
                if other_idx != i:
                    break
            parent2_a = parents.iloc[other_idx]['a']
            parent2_b = parents.iloc[other_idx]['b']

            alpha = np.random.rand()
            new_a = parent1_a + alpha * (parent2_a - parent1_a)
            new_b = parent1_b + alpha * (parent2_b - parent1_b)

            new_a += np.random.normal(0, 0.5)
            new_b += np.random.normal(0, 0.5)

            new_a = np.clip(new_a, 1, 32)
            new_b = np.clip(new_b, 1, 32)

            children.append({'a': new_a, 'b': new_b, 'Fitness': 0})

        self.population = pd.concat([parents, pd.DataFrame(children)], ignore_index=True)

    def evolve(self, target):
        self.get_fitness(target)
        self.selection()
        self.training_epochs += 1

    def assess_fitness(self, df, target_ellipse):
        print('Generating noisy ellipses for population of epoch ', self.training_epochs)

        #Given a dataframe (our population, generate associated noisy ellipses)
        ellipses_df = generate_noisy_ellipses(df)
        
        # Denoise the noisy ellipses using the trained model
        denoised_ellipses = []
        for index, row in ellipses_df.iterrows():
            noisy_ellipse = row.values.reshape(1, 1, 64, 64)
            noisy_tensor = torch.tensor(noisy_ellipse, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                denoised_tensor = self.model(noisy_tensor)
            denoised_ellipse = denoised_tensor.cpu().numpy().reshape(64, 64)
            denoised_ellipses.append(denoised_ellipse.flatten())
        
        denoised_ellipses_df = pd.DataFrame(denoised_ellipses)
        
        # Calculate fitness against the target
        fitness_scores = denoised_ellipses_df.apply(
            lambda row: calculate_fitness(row.values.reshape(64, 64), target_ellipse), axis=1)
        df['Fitness'] = fitness_scores
        self.denoised_ellipses_df = denoised_ellipses_df
        self.noisy_ellipses_df = ellipses_df

        # Update global fitness bounds
        self.global_min_fitness = min(self.global_min_fitness, df['Fitness'].min())
        self.global_max_fitness = max(self.global_max_fitness, df['Fitness'].max())

    def get_population(self):
        print(self.population)

    def plot_results(self, target):
        # Create a plot for the target and top/bottom 5 ellipses
        fig, axs = plt.subplots(5, 5, figsize=(20, 10))

        fig.suptitle(f"Most and Least Fit Population Members, Epoch {self.training_epochs}", fontsize=16)

        # Plot the target ellipse
        axs[0, 0].imshow(target, cmap='gray')
        axs[0, 0].set_title('Target')
        axs[0, 0].axis('off')

        # Leave empty spaces for alignment
        for i in range(1, 5):
            axs[0, i].axis('off')

        # Plot the top 5 most fit ellipses
        for i in range(5):
            # Noisy images
            noisy_img = self.noisy_ellipses_df.iloc[i].values.reshape(64, 64)
            axs[1, i].imshow(noisy_img, cmap='gray')
            axs[1, i].set_title(f'Top {i + 1} Noisy')
            axs[1, i].axis('off')
            
            # Denoised images
            denoised_img = self.denoised_ellipses_df.iloc[i].values.reshape(64, 64)
            axs[2, i].imshow(denoised_img, cmap='gray')
            axs[2, i].set_title(f'Top {i + 1} Denoised')
            axs[2, i].axis('off')

        # Plot the bottom 5 least fit ellipses
        for i in range(5):
            # Noisy images
            noisy_img = self.noisy_ellipses_df.iloc[-(i + 1)].values.reshape(64, 64)
            axs[3, i].imshow(noisy_img, cmap='gray')
            axs[3, i].set_title(f'Bottom {i + 1} Noisy')
            axs[3, i].axis('off')
            
            # Denoised images
            denoised_img = self.denoised_ellipses_df.iloc[-(i + 1)].values.reshape(64, 64)
            axs[4, i].imshow(denoised_img, cmap='gray')
            axs[4, i].set_title(f'Bottom {i + 1} Denoised')
            axs[4, i].axis('off')

        # Save the figure
        plt.tight_layout()
        plt.savefig(f'epoch_{self.training_epochs}.png')
        plt.close()

    def plot_population_evolution(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        sc = ax.scatter(self.population['a'], self.population['b'], c=self.population['Fitness'], cmap='plasma_r', vmin=0, vmax = 0.15)#self.global_min_fitness, vmax=self.global_max_fitness)
        ax.set_xlabel('a')
        ax.set_ylabel('b')
        ax.set_xlim(0, 32)
        ax.set_ylim(0, 32)
        ax.set_title(f'Genetic Algorithm Population Fitness, Epoch {self.training_epochs}', fontsize=20)
        ax.set_facecolor('black')
        cbar = fig.colorbar(sc, label='Loss')
        cbar.set_label('Loss')
        cbar.ax.set_yticklabels([f'{tick:.2f}' for tick in cbar.get_ticks()])
        cbar.ax.set_title(f'>=0.14', fontsize=10)
        plt.savefig(f'population_evolution_epoch_{self.training_epochs}.png')
        plt.close()

def calculate_fitness(ellipse, target):
    return np.mean(np.abs(ellipse - target))

if __name__ == "__main__":
    print("Running genetic plume tuner...")
    tuner = geneticPopulation()
    target = generate_ellipse(18, 4)
    for i in range(10):
        print(tuner.training_epochs)
        tuner.evolve(target=target)
        tuner.get_population()