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
# # Regression Models in Behavioral Science
#
# In this lecture, we will explore different regression techniques using PyTorch:
#
# 1. **Training and Evaluation Fundamentals**: Creating reusable functions
# 2. **Linear and Logistic Growth Models**: Comparing different functional forms
# 3. **Multivariate Regression**: Working with multiple predictors
# 4. **Multiple Response Variables**: Predicting multiple outcomes simultaneously
#
# We'll build on your existing knowledge of PyTorch models and optimization, focusing on applying these techniques to behavioral data.

# %% [markdown]
# ## 1. Training and Evaluation Fundamentals
#
# Before we dive into specific models, let's create reusable functions for training and evaluating models. This will help streamline our workflow for all subsequent exercises.
#
# ### Exercise 1.1: Implement a Training Function
#
# Create a `train_model` method within the UniversalProcedure class that trains a PyTorch model using gradient descent. Make sure to use the loss_func and optimizer defined within the class init!

# %%
class UniversalProcedure:
    """A class to implement the universal procedure for model training and evaluation."""

    def __init__(self, cross_validator,
                 evaluation_metrics=None, loss_func=None, optimizer=None):
        self.cross_validator = cross_validator

        if evaluation_metrics is None:
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


    def train(self, model, X_train, y_train, train_epochs, lr):

        # --- YOUR CODE HERE --- If you need a refresher: go to part 3 of the lecture 5 notebook!
        pass # delete this
        # set up optimizer and loss function from self

        # Track losses during training by appending to a list

        # Training loop

            # Forward pass

            # Compute loss


            # Backward pass and optimize

        # ----------------------

    def evaluate(self, model, x, y, train_epochs=500, lr=0.01):

        # Initialize results dictionary
        results = {}
        for name in self.evaluation_metrics.keys():
            results[f'splits_{name}'] = []

        # get default params from model
        original_state_dict = copy.deepcopy(model.state_dict())  # This is a reference so we can reset params later

        # state_dict lst
        state_dicts = []

        # Perform cross-validation
        for train_idx, test_idx in tqdm(self.cross_validator.split(x)):

            # --- YOUR CODE HERE ---
            # Split data
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # reset model params
            model.load_state_dict(original_state_dict)

            # Fit model
            self.train(model, x_train, y_train, train_epochs, lr)

            # Get predictions
            with torch.no_grad():
              y_test_pred = model(x_test)
            # ----------------------

            # Calculate metrics
            for name, metric_fn in self.evaluation_metrics.items():
                results[f'splits_{name}'].append(metric_fn(y_test, y_test_pred))

            state_dicts.append(copy.deepcopy(model.state_dict()))

        # Average metrics across folds
        for name in self.evaluation_metrics.keys():

            results[f'CV {name}'] = np.mean(results[f'splits_{name}'])
            results[f'CV {name} Std'] = np.std(results[f'splits_{name}'])

        return results, state_dicts


# %% [markdown]
# ### Solution 1.1: Training Function

# %%
class UniversalProcedure:
    """A class to implement the universal procedure for model training and evaluation."""

    def __init__(self,
                 cross_validator,
                 evaluation_metrics=None,
                 loss_func=None,
                 optimizer=None):

        self.cross_validator = cross_validator

        if evaluation_metrics is None:
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


    def train(self,
              model, #the instantiated but untrained pytorch model
              X_train, #the training data input
              y_train, #the training labels (the desired output)
              train_epochs, #how long you want to train for
              lr #the learning rate you want to use
              ):

        # --- YOUR CODE HERE ---
        # set up optimizer and loss function from self
        optimizer = self.optimizer(model.parameters(), lr=lr)
        loss_fn = self.loss_func
        # Track losses during training
        losses = []

        # Training loop
        for epoch in tqdm(range(train_epochs), leave=False):
            # Forward pass
            y_pred = model(X_train) #gets the prediction

            # Compute loss
            print(y_pred.shape, y_train.shape)
            loss = loss_fn(y_pred, y_train) #runs the loss function
            losses.append(loss.item()) #appends the loss for later tracking purposes

            # Backward pass and optimize
            optimizer.zero_grad() #... to make sure gradients don't accumulate
            loss.backward() #this actually calls the derivation calculation
            optimizer.step() #this actually applies the update

        return losses
        # ----------------------

    def evaluate(self, model, x, y, train_epochs=500, lr=0.01):

        # Initialize results dictionary
        results = {}
        for name in self.evaluation_metrics.keys():
            results[f'splits_{name}'] = []

        # get default params from model
        original_state_dict = copy.deepcopy(model.state_dict())  # This is a reference so we can reset params later

        # state_dict lst
        state_dicts = []

        # Perform cross-validation
        for train_idx, test_idx in tqdm(self.cross_validator.split(x)):

            # --- YOUR CODE HERE ---
            # Split data
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # reset model params
            model.load_state_dict(original_state_dict)

            # Fit model
            self.train(model, x_train, y_train, train_epochs, lr)

            # Get predictions
            with torch.no_grad():
              y_test_pred = model(x_test)
            # ----------------------

            # Calculate metrics
            for name, metric_fn in self.evaluation_metrics.items():
                results[f'splits_{name}'].append(metric_fn(y_test, y_test_pred))

            state_dicts.append(copy.deepcopy(model.state_dict()))

        # Average metrics across folds
        for name in self.evaluation_metrics.keys():

            results[f'CV {name}'] = np.mean(results[f'splits_{name}'])
            results[f'CV {name} Std'] = np.std(results[f'splits_{name}'])

        return results, state_dicts

# %% [markdown]
# ## 2. Linear and Logistic Growth Models
#
# Now, let's apply our training and evaluation functions to fit linear and logistic growth models to language acquisition data from Wordbank.
#
# First, let's load the Wordbank data:

# %%
# Load the Wordbank data
wordbank_data = pd.read_csv(os.path.join(DATA_DIRECTORY, "wordbank_bychild.csv"))
print("Wordbank data shape:", wordbank_data.shape)
print("\nFirst few rows:")
print(wordbank_data.head())

# %%
# Filter for rows with non-null production values
wordbank_filtered = wordbank_data.dropna(subset=["production"])

# Calculate summary statistics by language and age
production_averages = {}

# Get summary stats for each language
for language in wordbank_filtered["language"].unique():
    # Get data just for this language
    summary_data_by_child = wordbank_filtered[wordbank_filtered["language"] == language]

    # Skip languages with too little data
    if len(summary_data_by_child) < 10:
        continue

    # Produce summaries of the mean, std, and sample length, grouped by age bin
    prod_means = summary_data_by_child[["age", "production"]].groupby(["age"], as_index=False).mean()
    prod_stds = summary_data_by_child[["age", "production"]].groupby(["age"], as_index=False).std()
    prod_lens = summary_data_by_child[["age", "production"]].groupby(["age"], as_index=False).agg(len)

    # Get the independent variable
    ages = prod_means["age"]

    # Get the dependent variable mean
    means = prod_means["production"]

    # Get the dependent variable SEMs
    stds = prod_stds["production"]
    lens = prod_lens["production"]
    sems = stds / np.sqrt(lens)

    # Store computed things for future use
    production_averages[language] = (ages, means, sems)

# %% [markdown]
# ### Exercise 2.1: Linear Growth Model
#
# Implement a linear growth model as a PyTorch class, following the form $f(x) = ax + b$.

# %%
class LinearGrowthModel(nn.Module):
  ## YOUR CODE HERE ##
  pass # delete this
  # remember what the components of a pytorch class should be? if not, go to lecture 2 notebook!

# %% [markdown]
# ### Solution 2.1: Linear Growth Model

# %%
class LinearGrowthModel(nn.Module):
    def __init__(self):
        """
        Initialize the linear growth model with parameters a (slope) and b (intercept).
        """
        super(LinearGrowthModel, self).__init__()

        # Define parameters as nn.Parameter objects so PyTorch can track them
        self.a = nn.Parameter(torch.tensor(1.0))  # Slope
        self.b = nn.Parameter(torch.tensor(0.0))  # Intercept

    def forward(self, x):
        """
        Forward pass of the linear model.

        Args:
            x: Input tensor

        Returns:
            output: Model predictions
        """
        return self.a * x + self.b

# %% [markdown]
# ### Exercise 2.2: Logistic Growth Model
#
# Implement a logistic growth model as a PyTorch class, following the form:
#
# $$f(x) = \frac{c}{1 + e^{-b(x-a)}}$$
#
# Where:
# - a: Midpoint (age at which production is 50% of maximum)
# - b: Steepness (higher value = steeper curve)
# - c: Maximum value (asymptote)

# %%
class LogisticGrowthModel(nn.Module):
    ## YOUR CODE HERE ##
    pass # delete this
    # remember what the components of a pytorch class should be? if not, go to lecture 2 notebook!

# %% [markdown]
# ### Solution 2.2: Logistic Growth Model

# %%
class LogisticGrowthModel(nn.Module):
    def __init__(self):
        """
        Initialize the logistic growth model with parameters a, b, and c.
        """
        super(LogisticGrowthModel, self).__init__()

        # Define parameters as nn.Parameter objects
        self.a = nn.Parameter(torch.tensor(20.0))  # Midpoint (x value at 50% of maximum)
        self.b = nn.Parameter(torch.tensor(0.2))   # Steepness
        self.c = nn.Parameter(torch.tensor(500.0)) # Maximum value (asymptote)

    def forward(self, x):
        """
        Forward pass of the logistic model.

        Args:
            x: Input tensor

        Returns:
            output: Model predictions
        """
        return self.c / (1 + torch.exp(-self.b * (x - self.a)))

# %% [markdown]
# ### Exercise 2.3: Compare Linear and Logistic Models
#
# Train and evaluate both the linear and logistic growth models on language acquisition data for English.
# Compare their performance using your `train_model` and `eval_model` functions.

# %%
# Extract English data
language = "English (American)"
ages, means, sems = production_averages[language]

# Convert to PyTorch tensors
ages_tensor = torch.tensor(ages.values, dtype=torch.float32).reshape(-1, 1)
means_tensor = torch.tensor(means.values, dtype=torch.float32).reshape(-1, 1)

# Split data - using all data for this example since we have limited points
# In practice, you would use train/test split
X_train, y_train = ages_tensor, means_tensor

## YOUR CODE HERE ##

# create cross validator instance - if you don't remember how to do this, go to lecture 6 notebook and search for ShuffleSplit

# instantiate a UniversalProcedure instance

# create linear and logistic model instances

# Train and Evaluate models via the universal procedure, by calling the evaluate method

# print evaluation results - uncomment this
# print(f"Linear Model: CV R^2: {linear_results['CV R^2']} (Std={linear_results['CV R^2 Std']}); CV MSE: {linear_results['CV MSE']} (Std={linear_results['CV MSE Std']});")
# print(f"Logistic Model: CV R^2: {logistic_results['CV R^2']} (Std={logistic_results['CV R^2 Std']}); CV MSE: {logistic_results['CV MSE']} (Std={logistic_results['CV MSE Std']});")

# %% [markdown]
# ### Solution 2.3: Compare Linear and Logistic Models

# %%
# Extract English data
language = "English (American)"
ages, means, sems = production_averages[language]

# Convert to PyTorch tensors
ages_tensor = torch.tensor(ages.values, dtype=torch.float32).reshape(-1, 1)
means_tensor = torch.tensor(means.values, dtype=torch.float32).reshape(-1, 1)

# create cross validator instance
shufflesplit = ShuffleSplit(n_splits=10, test_size=0.2, random_state=SEED)

# instantiate a UniversalProcedure instance
up = UniversalProcedure(shufflesplit)

# create linear and logistic model instances
linear_model = LinearGrowthModel()
logistic_model = LogisticGrowthModel()

# Train and Evaluate models via the universal procedure
linear_results, state_dicts = up.evaluate(linear_model, ages_tensor, means_tensor)
logistic_results, state_dicts = up.evaluate(logistic_model, ages_tensor, means_tensor)

# print evaluation results
print(f"Linear Model: CV R^2: {linear_results['CV R^2']} (Std={linear_results['CV R^2 Std']}); CV MSE: {linear_results['CV MSE']} (Std={linear_results['CV MSE Std']});")
print(f"Logistic Model: CV R^2: {logistic_results['CV R^2']} (Std={logistic_results['CV R^2 Std']}); CV MSE {logistic_results['CV MSE']} (Std={logistic_results['CV MSE Std']});")

# %% [markdown]
# ### Exercise 2.4: Compare Models Across Languages
#
# Choose another language and repeat the comparison between linear and logistic growth models.
# Does the same model perform better across different languages?

# %%
# Choose another language and compare models
## YOUR CODE HERE ##
# you can simply copy the code above and change the language parameter

# %% [markdown]
# ### Solution 2.4: Compare Models Across Languages

# %%
# Extract English data
language = "Japanese"
ages, means, sems = production_averages[language]

# Convert to PyTorch tensors
ages_tensor = torch.tensor(ages.values, dtype=torch.float32).reshape(-1, 1)
means_tensor = torch.tensor(means.values, dtype=torch.float32).reshape(-1, 1)

# create cross validator instance
shufflesplit = ShuffleSplit(n_splits=10, test_size=0.2, random_state=SEED)

# instantiate a UniversalProcedure instance
up = UniversalProcedure(shufflesplit)

# create linear and logistic model instances
linear_model = LinearGrowthModel()
logistic_model = LogisticGrowthModel()

# Train and Evaluate models via the universal procedure
linear_results, state_dicts = up.evaluate(linear_model, ages_tensor, means_tensor)
logistic_results, state_dicts = up.evaluate(logistic_model, ages_tensor, means_tensor)

# print evaluation results
print(f"Linear Model: CV R^2: {linear_results['CV R^2']} (Std={linear_results['CV R^2 Std']}); CV MSE: {linear_results['CV MSE']} (Std={linear_results['CV MSE Std']});")
print(f"Logistic Model: CV R^2: {logistic_results['CV R^2']} (Std={logistic_results['CV R^2 Std']}); CV MSE {logistic_results['CV MSE']} (Std={logistic_results['CV MSE Std']});")

# %% [markdown]
# ## 3. Multivariate Regression
#
# We'll now move to multivariate regression, where we have multiple predictor variables.
#
# In multivariate regression, our model takes the form:
#
# $$\hat{y} = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b$$
#
# This can be expressed in matrix form as:
#
# $$\hat{y} = X \mathbf{w} + b$$
#
# Where:
# - $X$ is the design matrix with shape (n_samples, n_features)
# - $\mathbf{w}$ is the weight vector with shape (n_features, 1)
# - $b$ is the bias term
#
# Let's implement this using the SRO dataset.

# %%
# Load the SRO data
sro_datadir = os.path.join(DATA_DIRECTORY, "SRO")
health = pd.read_csv(os.path.join(sro_datadir, "health.csv"), index_col=0)
# add health before each health variable
health.columns = ["health_" + col for col in health.columns]
meaningful_vars = pd.read_csv(os.path.join(sro_datadir, "meaningful_variables_clean.csv"), index_col=0)

# Join the data
joined = health.join(meaningful_vars)

# show all columns
import sys
np.set_printoptions(threshold=sys.maxsize)
print("##############Health columns:#################\n")
all_health_features = [col for col in joined.columns if "health" in col]
print(all_health_features)

print("\n##############Survey columns:#################\n")
all_survey_features = [col for col in joined.columns if "survey" in col]
print(all_survey_features)

print("\n##############Task columns:#################\n")
all_task_features = [col for col in joined.columns if "task" in col]
print(all_task_features)

# %% [markdown]
# ### Exercise 3.1: Implement Multivariate Linear Regression
#
# Implement a multivariate linear regression model that takes multiple features as input.
# Use the design matrix approach, where the weights are represented as a vector.

# %%
class MultivariateLinearModel(nn.Module):
    def __init__(self, input_dim): # the __init__ method has to take in input_dim to know the size of the weight vector
      ## YOUR CODE HERE
      pass # delete this

# %% [markdown]
# ### Solution 3.1: Multivariate Linear Regression

# %%
class MultivariateLinearModel(nn.Module):
    def __init__(self, input_dim):
        """
        Initialize a multivariate linear regression model.

        Args:
            input_dim: Number of input features
        """
        super(MultivariateLinearModel, self).__init__()

        # Create weight matrix and bias vector as parameters
        self.W = nn.Parameter(torch.randn(input_dim, 1))
        self.b = nn.Parameter(torch.tensor(0.0))


    def forward(self, x):
        """
        Forward pass of the multivariate linear model.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            output: Model predictions
        """
        return x @ self.W + self.b

# %% [markdown]
# ### Exercise 3.2: Train and Evaluate Multivariate Model
#
# Use your training and evaluation functions to fit the multivariate model on SRO data.
# Predict "health_EverythingIsEffort" from multiple health-related features.

# %%
# Prepare the data for multivariate regression
# Select features and target
features = ["mindful_attention_awareness_survey.mindfulness",
            "ten_item_personality_survey.emotional_stability",
            "columbia_card_task_cold.loss_sensitivity",
            "probabilistic_selection.positive_learning_bias"]
target = "health_EverythingIsEffort"

# Drop rows with NaN values in any of these columns
clean_data = joined[features + [target]].dropna()

# Create the design matrix X and target vector y
X = clean_data[features].values
y = clean_data[target].values.reshape(-1, 1)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

## YOUR CODE HERE ##

# create cross validator instance

# instantiate the UniversalProcedure

# instantiate MultivariateLinearModel model instance

# Train and Evaluate models via the universal procedure

# # print evaluation results
# print(f"Linear Model: CV R^2: {multivariate_results['CV R^2']} (Std={multivariate_results['CV R^2 Std']})")

# # print avg feature values
# avg_W = torch.mean(torch.stack([state_dict["W"] for state_dict in state_dicts]), dim=0)
# print("\nAverage Weight per feature:\n")
# print("\n".join([": ".join([feature, str(weight)]) for feature, weight in zip(features, avg_W.squeeze().tolist())]))

# %% [markdown]
# ### Solution 3.2: Train and Evaluate Multivariate Model

# %%
# Prepare the data for multivariate regression
# Select features and target
features = ["mindful_attention_awareness_survey.mindfulness",
            "ten_item_personality_survey.emotional_stability",
            "columbia_card_task_cold.loss_sensitivity",
            "probabilistic_selection.positive_learning_bias"]
target = "health_EverythingIsEffort"

# Drop rows with NaN values in any of these columns
clean_data = joined[features + [target]].dropna()

# Create the design matrix X and target vector y
X = clean_data[features].values
y = clean_data[target].values.reshape(-1, 1)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# create cross validator instance
shufflesplit = ShuffleSplit(n_splits=10, test_size=0.2, random_state=SEED)

# instantiate the UniversalProcedure
up = UniversalProcedure(shufflesplit)

# instantiate MultivariateLinearModel model instance
multivariate_model = MultivariateLinearModel(len(features))

# Train and Evaluate models via the universal procedure
multivariate_results, state_dicts = up.evaluate(multivariate_model, X_tensor, y_tensor)

# print evaluation results
print(f"Multivariate model: CV R^2: {multivariate_results['CV R^2']} (Std={multivariate_results['CV R^2 Std']})")

# print avg feature values
avg_W = torch.mean(torch.stack([state_dict["W"] for state_dict in state_dicts]), dim=0)
print("\nAverage Weight per feature:\n")
print("\n".join([": ".join([feature, str(weight)]) for feature, weight in zip(features, avg_W.squeeze().tolist())]))

# %% [markdown]
# ## 4. Multiple Response Variables
#
# Finally, let's extend our model to handle multiple response variables. In this case, both our input X and output Y are matrices.
#
# The model takes the form:
#
# $$\hat{Y} = X W + B$$
#
# Where:
# - $X$ is the design matrix with shape (n_samples, n_features)
# - $W$ is the weight matrix with shape (n_features, n_targets)
# - $B$ is the bias matrix (or vector broadcast across samples)
#
# This generalizes the multivariate case to predict multiple outcomes simultaneously.

# %% [markdown]
# ### Exercise 4.1: Implement Multiple Response Model
#
# Implement a model that can predict multiple response variables simultaneously using standard matrix multiplication (not using `torch.linear`).

# %%
class MultipleResponseModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialize a model for multiple response variables.

        Args:
            input_dim: Number of input features
            output_dim: Number of output targets
        """
        ## YOUR CODE HERE ##
        pass # delete this

# %% [markdown]
#

# %% [markdown]
# ### Solution 4.1: Multiple Response Model

# %%
class MultipleResponseModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialize a model for multiple response variables.

        Args:
            input_dim: Number of input features
            output_dim: Number of output targets
        """
        super(MultipleResponseModel, self).__init__()

        # Create weight matrix and bias vector as parameters
        self.W = nn.Parameter(torch.randn(input_dim, output_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        """
        Forward pass using matrix multiplication.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            output: Model predictions of shape (batch_size, output_dim)
        """
        # Matrix multiplication: (batch_size, input_dim) × (input_dim, output_dim) = (batch_size, output_dim)
        return torch.matmul(x, self.W) + self.b

# %% [markdown]
# ### Exercise 4.2: Train and Evaluate Multiple Response Model
#
# Use your training and evaluation functions to predict multiple health outcomes simultaneously.

# %%
# Prepare the data for multiple response regression
# Select features and targets
features = ["mindful_attention_awareness_survey.mindfulness",
            "ten_item_personality_survey.emotional_stability",
            "columbia_card_task_cold.loss_sensitivity",
            "probabilistic_selection.positive_learning_bias"]
targets = ["health_EverythingIsEffort", "health_Depressed", "health_Nervous"]

# Drop rows with NaN values in any of these columns
clean_data = joined[features + targets].dropna()

# Create the design matrix X and target matrix Y
X = clean_data[features].values
Y = clean_data[targets].values

# Scale the features and targets
X_scaler = StandardScaler()
Y_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)
Y_scaled = Y_scaler.fit_transform(Y)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32)

## YOUR CODE HERE ##

# print evaluation results - uncomment below
# print(f"Multiple response model: CV R^2: {multiple_response_results['CV R^2']} (Std={multiple_response_results['CV R^2 Std']})")

# %% [markdown]
# ### Solution 4.2: Train and Evaluate Multiple Response Model

# %%
# Prepare the data for multiple response regression
# Select features and targets
features = ["mindful_attention_awareness_survey.mindfulness",
            "ten_item_personality_survey.emotional_stability",
            "columbia_card_task_cold.loss_sensitivity",
            "probabilistic_selection.positive_learning_bias"]
targets = ["health_EverythingIsEffort", "health_Depressed", "health_Nervous"]

# Drop rows with NaN values in any of these columns
clean_data = joined[features + targets].dropna()

# Create the design matrix X and target matrix Y
X = clean_data[features].values
Y = clean_data[targets].values

# Scale the features and targets
X_scaler = StandardScaler()
Y_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)
Y_scaled = Y_scaler.fit_transform(Y)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32)

# create cross validator instance
shufflesplit = ShuffleSplit(n_splits=10, test_size=0.2, random_state=SEED)

# instantiate the UniversalProcedure
up = UniversalProcedure(shufflesplit)

# instantiate MultivariateLinearModel model instance
multiple_response_model = MultipleResponseModel(len(features), len(targets))

# Train and Evaluate models via the universal procedure

multiple_response_results, state_dicts = up.evaluate(multiple_response_model, X_tensor, Y_tensor)

# print evaluation results
print(f"Multiple response model: CV R^2: {multiple_response_results['CV R^2']} (Std={multiple_response_results['CV R^2 Std']})")

# %% [markdown]
# ### Exercise 4.4: Run a model with all survey data versus all task data and get to the SRO paper's conclusions!
#
# This exercise intentionally has less scaffolding to let you get a sense of the full process of fitting this model!

# %%
targets = ["health_EverythingIsEffort", "health_Depressed", "health_Nervous", "health_Worthless", ]

# %%
# all survey data model
# Prepare the data for multiple response regression
# Select features and targets
features = all_survey_features

# Drop rows with NaN values in any of these columns
clean_data = joined[features + targets].dropna()

# Create the design matrix X and target matrix Y
X = clean_data[features].values
Y = clean_data[targets].values

# Scale the features and targets
X_scaler = StandardScaler()
Y_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)
Y_scaled = Y_scaler.fit_transform(Y)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32)

## YOUR CODE HERE ##

# create cross validator instance

# instantiate the UniversalProcedure

# instantiate MultivariateLinearModel model instance

# Train and Evaluate models via the universal procedure

# print evaluation results - uncomment below
# print(f"Multiple response model: CV R^2: {multiple_response_results['CV R^2']} (Std={multiple_response_results['CV R^2 Std']})")

# %%
# all task data model
# Prepare the data for multiple response regression
# Select features and targets
features = all_task_features
targets = ["health_EverythingIsEffort", "health_Depressed", "health_Nervous"]

# Drop rows with NaN values in any of these columns
clean_data = joined[features + targets].dropna()

# Create the design matrix X and target matrix Y
X = clean_data[features].values
Y = clean_data[targets].values

# Scale the features and targets
X_scaler = StandardScaler()
Y_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)
Y_scaled = Y_scaler.fit_transform(Y)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32)

## YOUR CODE HERE ##

# create cross validator instance

# instantiate the UniversalProcedure

# instantiate MultivariateLinearModel model instance

# Train and Evaluate models via the universal procedure

# print evaluation results - uncomment below
# print(f"Multiple response model: CV R^2: {multiple_response_results['CV R^2']} (Std={multiple_response_results['CV R^2 Std']})")

# %% [markdown]
# ### Solution 4.4: full survey and task models

# %%
# all survey data model
## YOUR CODE HERE ##
# Prepare the data for multiple response regression
# Select features and targets
features = all_survey_features
targets = ["health_EverythingIsEffort", "health_Depressed", "health_Nervous"]

# Drop rows with NaN values in any of these columns
clean_data = joined[features + targets].dropna()

# Create the design matrix X and target matrix Y
X = clean_data[features].values
Y = clean_data[targets].values

# Scale the features and targets
X_scaler = StandardScaler()
Y_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)
Y_scaled = Y_scaler.fit_transform(Y)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32)

# create cross validator instance
shufflesplit = ShuffleSplit(n_splits=10, test_size=0.2, random_state=SEED)

# instantiate the UniversalProcedure
up = UniversalProcedure(shufflesplit)

# instantiate MultivariateLinearModel model instance
multiple_response_model = MultipleResponseModel(len(features), len(targets))

# Train and Evaluate models via the universal procedure
multiple_response_results, state_dicts = up.evaluate(multiple_response_model, X_tensor, Y_tensor)

# print evaluation results
print(f"Multiple response model: CV R^2: {multiple_response_results['CV R^2']} (Std={multiple_response_results['CV R^2 Std']})")

# %%
# all task data model
## YOUR CODE HERE ##
# Prepare the data for multiple response regression
# Select features and targets
features = all_task_features
targets = ["health_EverythingIsEffort", "health_Depressed", "health_Nervous"]

# Drop rows with NaN values in any of these columns
clean_data = joined[features + targets].dropna()

# Create the design matrix X and target matrix Y
X = clean_data[features].values
Y = clean_data[targets].values

# Scale the features and targets
X_scaler = StandardScaler()
Y_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)
Y_scaled = Y_scaler.fit_transform(Y)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32)

# create cross validator instance
shufflesplit = ShuffleSplit(n_splits=10, test_size=0.2, random_state=SEED)

# instantiate the UniversalProcedure
up = UniversalProcedure(shufflesplit)

# instantiate MultivariateLinearModel model instance
multiple_response_model = MultipleResponseModel(len(features), len(targets))

# Train and Evaluate models via the universal procedure
multiple_response_results, state_dicts = up.evaluate(multiple_response_model, X_tensor, Y_tensor)

# print evaluation results
print(f"Multiple response model: CV R^2: {multiple_response_results['CV R^2']} (Std={multiple_response_results['CV R^2 Std']})")


# %% [markdown]
# ### Exercise 4.4: add regularization to improve your predictions
#
# We will get to why this works in future classes...
#
# Instructions:
#
# 1. Copy your Universal Procedure class here.
#
# 2. In the __init__ method, add a new argument called l1_regularization, if it set to true, set self.l1_regularization = True
#
# 3. In the train_model method, after computing the loss and before the gradient step, add an if clause for if self.l1_regularization
#
# 4. Inside the if clause, define a variable called l1_term, and set it equal to the sum of the absolute values of all model parameters (HINT: use model parameters(), torch.sum and torch.abs)
#
# 5. Again inside the if clause, set loss = loss + 0.01*l1_term
#
# 6. Rerun the chunk containing the updated universal procedure
#
# 7. Copy the two previous cells containing the models with all survey variables and all task variables to the code cells below, set l1_regularization = True when instantiating the Universal procedure class, and rerun them.
#
# 8. Do you see any differences? what changed?

# %%
# # copy universal procedure class here

# %%
# # copy all survey data model

# %%
# # copy all task data model

# %% [markdown]
# ### Solution 4.4:

# %%
class UniversalProcedure:
    """A class to implement the universal procedure for model training and evaluation."""

    def __init__(self, cross_validator,
                 evaluation_metrics=None, loss_func=None, optimizer=None, l1_regularization = True):
        self.cross_validator = cross_validator

        if evaluation_metrics is None:
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

        self.l1_regularization = l1_regularization


    def train(self, model, X_train, y_train, train_epochs, lr):

        # --- YOUR CODE HERE ---
        # set up optimizer and loss function from self
        optimizer = self.optimizer(model.parameters(), lr=lr)
        loss_fn = self.loss_func
        # Track losses during training
        losses = []

        # Training loop
        for epoch in tqdm(range(train_epochs), leave=False):
            # Forward pass
            y_pred = model(X_train)

            # Compute loss
            loss = loss_fn(y_pred, y_train)
            # add l1 reg if required
            if self.l1_regularization:
              l1_term = sum(p.abs().sum() for p in model.parameters())
              loss = loss + 0.01*l1_term

            losses.append(loss.item())

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return losses
        # ----------------------

    def evaluate(self, model, x, y, train_epochs=500, lr=0.01):

        # Initialize results dictionary
        results = {}
        for name in self.evaluation_metrics.keys():
            results[f'splits_{name}'] = []

        # get default params from model
        original_state_dict = copy.deepcopy(model.state_dict())  # This is a reference so we can reset params later

        # state_dict lst
        state_dicts = []

        # Perform cross-validation
        for train_idx, test_idx in tqdm(self.cross_validator.split(x)):

            # --- YOUR CODE HERE ---
            # Split data
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # reset model params
            model.load_state_dict(original_state_dict)

            # Fit model
            self.train(model, x_train, y_train, train_epochs, lr)

            # Get predictions
            with torch.no_grad():
              y_test_pred = model(x_test)
            # ----------------------

            # Calculate metrics
            for name, metric_fn in self.evaluation_metrics.items():
                results[f'splits_{name}'].append(metric_fn(y_test, y_test_pred))

            state_dicts.append(copy.deepcopy(model.state_dict()))

        # Average metrics across folds
        for name in self.evaluation_metrics.keys():

            results[f'CV {name}'] = np.mean(results[f'splits_{name}'])
            results[f'CV {name} Std'] = np.std(results[f'splits_{name}'])

        return results, state_dicts

# %%
# all survey data model
## YOUR CODE HERE ##
# Prepare the data for multiple response regression
# Select features and targets
features = all_survey_features
targets = ["health_EverythingIsEffort", "health_Depressed", "health_Nervous"]

# Drop rows with NaN values in any of these columns
clean_data = joined[features + targets].dropna()

# Create the design matrix X and target matrix Y
X = clean_data[features].values
Y = clean_data[targets].values

# Scale the features and targets
X_scaler = StandardScaler()
Y_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)
Y_scaled = Y_scaler.fit_transform(Y)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32)

# create cross validator instance
shufflesplit = ShuffleSplit(n_splits=10, test_size=0.2, random_state=SEED)

# instantiate the UniversalProcedure
up = UniversalProcedure(shufflesplit, l1_regularization=True)

# instantiate MultivariateLinearModel model instance
multiple_response_model = MultipleResponseModel(len(features), len(targets))

# Train and Evaluate models via the universal procedure
multiple_response_results, state_dicts = up.evaluate(multiple_response_model, X_tensor, Y_tensor)

# print evaluation results
print(f"Multiple response model: CV R^2: {multiple_response_results['CV R^2']} (Std={multiple_response_results['CV R^2 Std']})")

# %%
# all task data model
## YOUR CODE HERE ##
# Prepare the data for multiple response regression
# Select features and targets
features = all_task_features
targets = ["health_EverythingIsEffort", "health_Depressed", "health_Nervous"]

# Drop rows with NaN values in any of these columns
clean_data = joined[features + targets].dropna()

# Create the design matrix X and target matrix Y
X = clean_data[features].values
Y = clean_data[targets].values

# Scale the features and targets
X_scaler = StandardScaler()
Y_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)
Y_scaled = Y_scaler.fit_transform(Y)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32)

# create cross validator instance
shufflesplit = ShuffleSplit(n_splits=10, test_size=0.2, random_state=SEED)

# instantiate the UniversalProcedure
up = UniversalProcedure(shufflesplit, l1_regularization=True)

# instantiate MultivariateLinearModel model instance
multiple_response_model = MultipleResponseModel(len(features), len(targets))

# Train and Evaluate models via the universal procedure
multiple_response_results, state_dicts = up.evaluate(multiple_response_model, X_tensor, Y_tensor)

# print evaluation results
print(f"Multiple response model: CV R^2: {multiple_response_results['CV R^2']} (Std={multiple_response_results['CV R^2 Std']})")

# %% [markdown]
# ## Conclusion
#
# In this notebook, we explored various regression techniques using PyTorch:
#
# 1. **Training and Evaluation Functions**: We created reusable functions for model training and evaluation.
#
# 2. **Growth Models**: We compared linear and logistic growth models for language acquisition.
#
# 3. **Multivariate Regression**: We implemented regression with multiple predictors using a design matrix approach.
#
# 4. **Multiple Response Variables**: We extended our models to predict multiple outcomes simultaneously.
#
# These techniques form the foundation of regression analysis in behavioral science and demonstrate the flexibility of PyTorch for implementing custom statistical models.
