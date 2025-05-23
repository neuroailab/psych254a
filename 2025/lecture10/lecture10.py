# %%
# Set up the environment with necessary imports
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm
from sklearn.utils import check_random_state
from sklearn.model_selection import ShuffleSplit
import copy

# Set plot style
sns.set_style("whitegrid")

# Mount drive (if using Google Colab)
from google.colab import drive

drive.mount("/content/drive/")
DATA_DIRECTORY = "/content/drive/MyDrive/psych254a_2025/data"

# set random seed
SEED = 111
np.random.seed(SEED)
torch.manual_seed(SEED)
check_random_state(SEED)

# %% [markdown]
# # Mixed Effects Models in Behavioral Science
#
# In this lecture, we will explore mixed effects models using PyTorch to analyze word acquisition data:
#
# 1. **Introduction to Mixed Effects Models**: Understanding when and why to use them
# 2. **Logistic Growth Models**: Implementing sigmoid functions for developmental trajectories
# 3. **Random Effects Logistic Models**: Adding word-specific parameters
# 4. **Model Comparison**: Evaluating the benefit of random effects
#
# We'll build on your existing knowledge of PyTorch models while demonstrating the flexibility it offers for implementing custom statistical models.

# %% [markdown]
# ## 1. Introduction to Mixed Effects Models
#
# Mixed effects models (also called multilevel or hierarchical models) are designed for data with nested or grouped structure. They include both fixed effects (population-level parameters) and random effects (group-specific parameters).
#
# The standard linear mixed model can be written as:
#
# $$y_{ij} = \beta_0 + \beta_1 x_{ij} + \ldots + \beta_p x_{ij}^{(p)} + b_{0i} + b_{1i} x_{ij} + \ldots + b_{qi} x_{ij}^{(q)} + \epsilon_{ij}$$
#
# Where:
# - $y_{ij}$ is the response for observation $j$ in group $i$
# - $\beta_0, \beta_1, \ldots, \beta_p$ are fixed effects (population parameters)
# - $b_{0i}, b_{1i}, \ldots, b_{qi}$ are random effects for group $i$ (deviations from population parameters)
# - $\epsilon_{ij}$ is the error term
#
# The random effects are typically assumed to follow a multivariate normal distribution:
#
# $$\mathbf{b}_i \sim {N}(\mathbf{0}, \mathbf{\Sigma})$$
#
# Where $\mathbf{\Sigma}$ is the variance-covariance matrix of the random effects.
#
# In this lecture, we'll extend this framework to a logistic growth model for word acquisition data. Logistic models are appropriate for developmental data where there is an S-shaped growth curve from 0% to 100%.
#
# The standard logistic function is:
#
# $$f(x) = \frac{L}{1 + e^{-k(x-x_0)}}$$
#
# Where:
# - $L$ is the maximum value (usually 1 for proportions)
# - $k$ is the growth rate (steepness of the curve)
# - $x_0$ is the midpoint (age at which 50% of children produce the word)
#
# By implementing this in PyTorch, we gain tremendous flexibility to customize our models while leveraging automatic differentiation for parameter estimation.

# %% [markdown]
# ## 2. Data Preparation and Exploration
#
# First, let's load and explore the Wordbank data, which contains the percentage of children producing various words at different ages.

# %%
# Load the wordbank_word data
wordbank_path = os.path.join(DATA_DIRECTORY, "wordbank_item_data.csv")
wordbank_data = pd.read_csv(wordbank_path)

# let's take 50 random words
wordbank_data = wordbank_data.sample(n=50, random_state=SEED)

# Explore the data
print(f"Shape of data: {wordbank_data.shape}")
print(f"Column names: {wordbank_data.columns[:10]}")
print("\nFirst few rows:")
wordbank_data.head()

# %%
# Data needs some cleaning and restructuring for modeling
# First, let's identify the age columns (those that start with numbers)
age_columns = [col for col in wordbank_data.columns if col[0].isdigit()]
print(f"Age columns: {age_columns}")

# Create a long-format dataframe with word, age, and production percentage
data_long = pd.melt(
    wordbank_data,
    id_vars=['item_definition', 'item_id'],
    value_vars=age_columns,
    var_name='age',
    value_name='production_pct'
)

# Convert age from string to numeric (remove 'mo' suffix)
data_long['age'] = data_long['age'].str.replace('mo', '').astype(int)

# Convert percentage to proportion (0-1 scale)
data_long['production_prop'] = data_long['production_pct'] / 100

# Filter out extreme values (optional)
data_long = data_long[(data_long['production_prop'] >= 0) & (data_long['production_prop'] <= 1)]

print(f"Shape of long-format data: {data_long.shape}")
print("\nFirst few rows of long-format data:")
data_long.head()

# %%
# Explore words and their acquisition curves
plt.figure(figsize=(12, 8))

# Select a few common words to visualize
random_word_indices = np.random.choice(len(data_long['item_definition'].unique()), 10, replace=False)
demonstration_words = data_long['item_definition'].unique()[random_word_indices]
for word in demonstration_words:
    word_data = data_long[data_long['item_definition'] == word]
    plt.plot(word_data['age'], word_data['production_prop'], 'o-', label=word)

plt.title("Word Acquisition Curves for Common Words")
plt.xlabel("Age (months)")
plt.ylabel("Proportion of Children Producing Word")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# As we can see from the exploration, different words follow different acquisition trajectories. Some words are learned earlier than others, and the rate of acquisition varies across words. This suggests that a mixed effects model with word-specific parameters would be appropriate.
#
# ## 3. Implementing a Logistic Growth Model
#
# Let's start by implementing a simple logistic growth model with fixed effects only. This will serve as our baseline model.

# %%
class UniversalProcedure:
    """A class to implement the universal procedure for model training and evaluation."""

    def __init__(
        self, cross_validator=None, evaluation_metrics={}, loss_func=None, optimizer=None
    ):

        self.cross_validator = cross_validator

        if not evaluation_metrics:
            self.evaluation_metrics = {
                'MSE': nn.MSELoss(),
                'R^2': lambda y, y_pred: 1 - torch.sum((y - y_pred)**2) / torch.sum((y - torch.mean(y))**2)
            }
        else:
            self.evaluation_metrics = evaluation_metrics

        if loss_func is None:
            self.loss_func = nn.MSELoss()
        else:
            self.loss_func = loss_func

        if optimizer is None:
            self.optimizer = optim.Adam
        else:
            self.optimizer = optimizer

    def train(
        self,
        model,  # The model to train
        X_train,  # Features
        y_train,  # Target values
        word_ids=None,  # Word IDs (for random effects)
        train_epochs=1000,  # Increase epochs
        lr=0.01,  # Learning rate
        reg_lambda=0.01  # Regularization strength
    ):
        # Initialize optimizer
        optimizer = self.optimizer(model.parameters(), lr=lr)
        loss_fn = self.loss_func
        
        # For tracking progress
        losses = []
        
        # Training loop
        for epoch in tqdm(range(train_epochs)):
            # Forward pass
            if word_ids is not None:
                y_pred = model(X_train, word_ids)
            else:
                y_pred = model(X_train)
            
            # Compute loss
            loss = loss_fn(y_pred, y_train)
            
            # Add regularization for random effects
            if hasattr(model, 'random_x0'):
                loss = loss + reg_lambda * torch.sum(model.random_x0**2)
            if hasattr(model, 'random_k'):
                loss = loss + reg_lambda * torch.sum(model.random_k**2)
            
            # Store loss
            losses.append(loss.item())
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Add early stopping based on loss plateau
            if epoch > 100 and epoch % 100 == 0:
                if abs(losses[-100] - losses[-1]) < 1e-4:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        return losses

# %%
class FixedEffectsLogisticModel(nn.Module):
    def __init__(self):
        """
        Initialize a logistic growth model with fixed effects only.
        """
        super(FixedEffectsLogisticModel, self).__init__()

        # Fixed effects (population parameters)
        self.fixed_L = nn.Parameter(torch.tensor(1.0))  # Maximum value (typically 1 for proportions)
        self.fixed_x0 = nn.Parameter(torch.tensor(18.0))  # Midpoint (age at 50% acquisition)
        self.fixed_k = nn.Parameter(torch.tensor(0.3))  # Growth rate

    def forward(self, x):
        """
        Forward pass implementing the logistic function.

        Args:
            x: Age values (months)

        Returns:
            Predicted proportion of children producing words at each age
        """
        # Apply sigmoid function with fixed parameters
        return self.fixed_L / (1 + torch.exp(-self.fixed_k * (x - self.fixed_x0)))

    def predict(self, x):
        """
        Make predictions (for compatibility with evaluation frameworks).
        """
        return self.forward(x)

# %%
# Prepare data for PyTorch modeling
# Create word-to-index mapping
unique_words = data_long['item_definition'].unique()
word_to_idx = {word: i for i, word in enumerate(unique_words)}
num_words = len(unique_words)

# Convert to tensors
X = torch.tensor(data_long['age'].values, dtype=torch.float32).reshape(-1, 1)
y = torch.tensor(data_long['production_prop'].values, dtype=torch.float32).reshape(-1, 1)
word_ids = torch.tensor([word_to_idx[word] for word in data_long['item_definition']], dtype=torch.long)

# Train the fixed effects logistic model
fixed_model = FixedEffectsLogisticModel()
procedure = UniversalProcedure()
fixed_losses = procedure.train(fixed_model, X, y, train_epochs=500, lr=0.05)

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(fixed_losses)
plt.title('Fixed Effects Logistic Model: Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.show()

# %% [markdown]
# ## 4. Random Effects Logistic Models
#
# Now, let's implement a mixed effects logistic model. We'll start with random midpoints (x0) only, allowing each word to have its own age of 50% acquisition, but sharing the growth rate parameter.

# %%
class RandomMidpointLogisticModel(nn.Module):
    def __init__(self, num_words):
        """
        Initialize a logistic growth model with random midpoints (x0).

        Args:
            num_words: Number of unique words
        """
        super(RandomMidpointLogisticModel, self).__init__()

        # Fixed effects (population parameters)
        self.fixed_L = nn.Parameter(torch.tensor(1.0))  # Maximum value
        self.fixed_x0 = nn.Parameter(torch.tensor(18.0))  # Population midpoint
        self.fixed_k = nn.Parameter(torch.tensor(0.3))  # Growth rate

        # Random effects (word-specific deviations)
        # Initialize with small random values instead of zeros
        self.random_x0 = nn.Parameter(torch.randn(num_words) * 0.1)  # Word-specific deviations from population midpoint

    def forward(self, x, word_ids):
        """
        Forward pass implementing the logistic function with random midpoints.

        Args:
            x: Age values (months)
            word_ids: Word indices for each observation

        Returns:
            Predicted proportion of children producing words at each age
        """
        # Get word-specific midpoints (fixed + random)
        word_x0 = self.fixed_x0 + self.random_x0[word_ids]

        # Apply sigmoid function with word-specific midpoints
        # Add a small epsilon for numerical stability
        return self.fixed_L / (1 + torch.exp(-self.fixed_k * (x - word_x0)))

    def predict(self, x, word_ids):
        """
        Make predictions (for compatibility with evaluation frameworks).
        """
        return self.forward(x, word_ids)

# %%
# Train the random midpoint logistic model
random_x0_model = RandomMidpointLogisticModel(num_words)
random_x0_losses = procedure.train(random_x0_model, X, y, word_ids, train_epochs=500, lr=0.05)

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(random_x0_losses)
plt.title('Random Midpoint Logistic Model: Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.show()

# %% [markdown]
# Now, let's implement a full mixed effects logistic model with both random midpoints (x0) and random growth rates (k). This allows each word to have its own age of 50% acquisition and its own learning rate.

# %%
class FullRandomEffectsLogisticModel(nn.Module):
    def __init__(self, num_words):
        """
        Initialize a logistic growth model with random midpoints (x0) and random growth rates (k).

        Args:
            num_words: Number of unique words
        """
        super(FullRandomEffectsLogisticModel, self).__init__()

        # Fixed effects (population parameters)
        self.fixed_L = nn.Parameter(torch.tensor(1.0))  # Maximum value
        self.fixed_x0 = nn.Parameter(torch.tensor(18.0))  # Population midpoint
        self.fixed_k = nn.Parameter(torch.tensor(0.3))  # Growth rate

        # Random effects (word-specific deviations)
        # Initialize with small random values
        self.random_x0 = nn.Parameter(torch.randn(num_words) * 0.1)  # Word-specific deviations from population midpoint
        self.random_k = nn.Parameter(torch.randn(num_words) * 0.01)  # Word-specific deviations from population growth rate

    def forward(self, x, word_ids):
        """
        Forward pass implementing the logistic function with random midpoints and growth rates.

        Args:
            x: Age values (months)
            word_ids: Word indices for each observation

        Returns:
            Predicted proportion of children producing words at each age
        """
        # Get word-specific parameters (fixed + random)
        word_x0 = self.fixed_x0 + self.random_x0[word_ids]
        word_k = self.fixed_k + self.random_k[word_ids]
        
        # Ensure k is positive by using softplus or abs
        word_k = torch.abs(word_k)  # Or use F.softplus(word_k)
        
        # Apply sigmoid function with word-specific parameters
        return self.fixed_L / (1 + torch.exp(-word_k * (x - word_x0)))

    def predict(self, x, word_ids):
        """
        Make predictions (for compatibility with evaluation frameworks).
        """
        return self.forward(x, word_ids)

# %%
# Train the full random effects logistic model
full_random_model = FullRandomEffectsLogisticModel(num_words)
full_random_losses = procedure.train(full_random_model, X, y, word_ids, train_epochs=500, lr=0.05)

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(full_random_losses)
plt.title('Full Random Effects Logistic Model: Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.show()

# %% [markdown]
# ## 5. Model Comparison and Visualization
#
# Let's compare the performance of our models and visualize the word-specific acquisition curves.

# %%
# Evaluate models
with torch.no_grad():
    # Fixed effects model
    y_pred_fixed = fixed_model(X)
    mse_fixed = nn.MSELoss()(y_pred_fixed, y)
    r2_fixed = 1 - torch.sum((y - y_pred_fixed)**2) / torch.sum((y - torch.mean(y))**2)

    # Random midpoint model
    y_pred_x0 = random_x0_model(X, word_ids)
    mse_x0 = nn.MSELoss()(y_pred_x0, y)
    r2_x0 = 1 - torch.sum((y - y_pred_x0)**2) / torch.sum((y - torch.mean(y))**2)

    # Full random effects model
    y_pred_full = full_random_model(X, word_ids)
    mse_full = nn.MSELoss()(y_pred_full, y)
    r2_full = 1 - torch.sum((y - y_pred_full)**2) / torch.sum((y - torch.mean(y))**2)

print("Model Performance Comparison:")
print(f"Fixed Effects Model:       MSE = {mse_fixed.item():.4f}, R² = {r2_fixed.item():.4f}")
print(f"Random Midpoint Model:     MSE = {mse_x0.item():.4f}, R² = {r2_x0.item():.4f}")
print(f"Full Random Effects Model: MSE = {mse_full.item():.4f}, R² = {r2_full.item():.4f}")

# %%
# Visualize fixed model predictions for selected words
def plot_word_predictions(word_list, model, is_random_effects=False):
    """
    Plot the actual data and model predictions for a list of words.
    
    Args:
        word_list: List of words to visualize
        model: The trained model
        is_random_effects: Whether the model has random effects (needs word IDs)
    """
    plt.figure(figsize=(15, 10))
    age_range = torch.linspace(8, 30, 100).reshape(-1, 1)
    
    for i, word in enumerate(word_list):
        plt.subplot(3, 3, i+1)
        
        # Get actual data for this word
        word_data = data_long[data_long['item_definition'] == word]
        plt.scatter(word_data['age'], word_data['production_prop'], color='blue', 
                   alpha=0.7, label='Observed Data')
        
        # Get model predictions
        if is_random_effects:
            word_idx = word_to_idx[word]
            word_id_tensor = torch.tensor([word_idx] * len(age_range))
            with torch.no_grad():
                predictions = model(age_range, word_id_tensor)
        else:
            with torch.no_grad():
                predictions = model(age_range)
        
        # Plot model predictions
        plt.plot(age_range, predictions, 'r-', linewidth=2, label='Model Prediction')
        
        # Add titles and labels
        plt.title(f'"{word}"')
        plt.xlabel('Age (months)')
        plt.ylabel('Proportion')
        plt.ylim(-0.05, 1.05)
        
        # Only add legend to the first subplot
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.show()

# %%
# Visualize the trained fixed effects model
plot_word_predictions(demonstration_words, fixed_model, is_random_effects=False)

# %% [markdown]
# As we can see, the fixed effects model uses the same curve for all words, which doesn't capture the variation between words. Some words are learned earlier than others, and some have different rates of acquisition. Let's extend our model to allow for word-specific midpoints.

# %%
# After training the random midpoint model, visualize its predictions
plot_word_predictions(demonstration_words, random_x0_model, is_random_effects=True)

# %% [markdown]
# The random midpoint model allows each word to have its own midpoint (x0), which better captures when words are learned. However, we can see that for some words, the steepness of the curve still doesn't match the data well. Let's extend to a full random effects model.

# %%
# After training the full random effects model, visualize its predictions
plot_word_predictions(demonstration_words, full_random_model, is_random_effects=True)

# %% [markdown]
# The full random effects model, with both word-specific midpoints and growth rates, provides the best fit to the data, capturing both when words are learned and how quickly they spread through the population.

# %%
# Compare all three models on the same plot for a few selected words
def compare_models(word_list, fixed_model, random_x0_model, full_random_model):
    """
    Compare predictions from all three models for selected words.
    """
    plt.figure(figsize=(15, 10))
    age_range = torch.linspace(8, 30, 100).reshape(-1, 1)
    
    for i, word in enumerate(word_list):
        plt.subplot(3, 2, i+1)
        
        # Get actual data
        word_data = data_long[data_long['item_definition'] == word]
        plt.scatter(word_data['age'], word_data['production_prop'], color='black', 
                   alpha=0.7, label='Observed Data')
        
        # Get word index for random effects models
        word_idx = word_to_idx[word]
        word_id_tensor = torch.tensor([word_idx] * len(age_range))
        
        # Get predictions from all models
        with torch.no_grad():
            fixed_pred = fixed_model(age_range)
            random_x0_pred = random_x0_model(age_range, word_id_tensor)
            full_random_pred = full_random_model(age_range, word_id_tensor)
        
        # Plot predictions
        plt.plot(age_range, fixed_pred, 'r-', linewidth=2, label='Fixed Effects')
        plt.plot(age_range, random_x0_pred, 'g-', linewidth=2, label='Random x0')
        plt.plot(age_range, full_random_pred, 'b-', linewidth=2, label='Random x0 & k')
        
        plt.title(f'"{word}"')
        plt.xlabel('Age (months)')
        plt.ylabel('Proportion')
        plt.ylim(-0.05, 1.05)
        
        # Only add legend to the first subplot
        if i == 0:
            plt.legend()
    
    plt.tight_layout()
    plt.show()

# Compare all three models for a subset of words
comparison_words = ['mommy', 'water', 'ball', 'computer', 'elephant', 'telephone']
compare_models(comparison_words, fixed_model, random_x0_model, full_random_model)

# %% [markdown]
# ## 6. Semantic Category Exploration
#
# Let's explore whether words from the same semantic category have similar acquisition patterns.

# %%
# Define some semantic categories (simplified example)
categories = {
    'animals': ['dog', 'cat', 'horse', 'cow', 'sheep', 'pig', 'fish', 'bird'],
    'food': ['banana', 'apple', 'cookie', 'juice', 'water', 'milk', 'bread', 'cheese'],
    'people': ['mommy', 'daddy', 'baby', 'grandma', 'grandpa', 'friend'],
    'household': ['phone', 'chair', 'table', 'bath', 'potty', 'blanket', 'bed', 'couch'],
    'toys': ['ball', 'book', 'toy', 'doll', 'teddy', 'balloon', 'car', 'block']
}

# Collect parameter values by category
category_data = []
with torch.no_grad():
    fixed_x0 = full_random_model.fixed_x0.item()
    fixed_k = full_random_model.fixed_k.item()
    random_x0 = full_random_model.random_x0.detach().numpy()
    random_k = full_random_model.random_k.detach().numpy()

for category, words in categories.items():
    for word in words:
        if word in word_to_idx:
            idx = word_to_idx[word]
            category_data.append({
                'word': word,
                'category': category,
                'x0': fixed_x0 + random_x0[idx],
                'k': fixed_k + random_k[idx]
            })

category_df = pd.DataFrame(category_data)

# Visualize parameters by category
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x='category', y='x0', data=category_df)
plt.title('Midpoint (x0) by Semantic Category')
plt.xlabel('Category')
plt.ylabel('Age at 50% Acquisition')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.boxplot(x='category', y='k', data=category_df)
plt.title('Growth Rate (k) by Semantic Category')
plt.xlabel('Category')
plt.ylabel('Growth Rate')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# %%
# Visualize acquisition curves by category
plt.figure(figsize=(15, 10))

for i, (category, words) in enumerate(categories.items()):
    plt.subplot(2, 3, i+1)

    # Plot acquisition curves for each word in the category
    for word in words:
        if word in word_to_idx:
            word_idx = word_to_idx[word]
            word_id_tensor = torch.tensor([word_idx] * len(age_range))

            with torch.no_grad():
                word_pred = full_random_model(age_range, word_id_tensor)

            plt.plot(age_range, word_pred, alpha=0.7, label=word if i == 0 else None)

    plt.title(f'Category: {category}')
    plt.xlabel('Age (months)')
    plt.ylabel('Proportion')
    plt.ylim(-0.05, 1.05)

    if i == 0:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Conclusion
#
# In this lecture, we've explored mixed effects models for logistic growth curves using PyTorch:
#
# 1. **Fixed Effects Logistic Model**: A baseline model with the same parameters for all words.
#
# 2. **Random Midpoint Model**: Allowing each word to have its own age of 50% acquisition.
#
# 3. **Full Random Effects Model**: Extending to word-specific growth rates as well.
#
# Our results show that:
#
# - Different words follow distinct developmental trajectories
# - Adding random effects significantly improves model fit
# - Words within semantic categories show some similarities in acquisition patterns
#
# The flexibility of PyTorch enabled us to implement these custom mixed effects models efficiently and extend them beyond what's typically available in standard statistical packages. By leveraging automatic differentiation and custom model architectures, we can create models that are tailored to the specific properties of our developmental data.
#
# This approach can be extended to more complex models with multiple predictors, interactions, and additional random effects as needed for specific research questions.

# %% [markdown]
# ## 8. Additional Exercises (optional)
#
# 1. **Regularization Exploration**: Experiment with different regularization strengths for the random effects and observe how they affect model fit and parameter estimates.
#
# 2. **Additional Random Effects**: Extend the model to include a random asymptote (L) parameter for each word.
#
# 3. **Category-Level Effects**: Implement a three-level model with effects at the word level nested within semantic categories.
#
# 4. **Model Validation**: Implement cross-validation to assess the generalization performance of the different models.
