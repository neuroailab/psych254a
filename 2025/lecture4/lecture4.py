# %%
# Set up the environment with necessary imports
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx

# Set plot style
sns.set_style("whitegrid")

# %% [markdown]
# # Computation Graphs and Automatic Differentiation
#
# In this interactive lecture, we'll explore the fundamental concepts behind modern deep learning optimization:
#
# 1. **Computation Graphs**: How PyTorch represents operations
# 2. **Automatic Differentiation**: How PyTorch computes gradients
# 3. **Benefits of Autograd**: Handling complex functions
# 4. **Working with Autodiff**: Practical considerations
#
# Each section contains explanations and hands-on exercises to build your understanding.

# %% [markdown]
# ## 1. Computation Graphs
#
# ### 1.1 What is a Computation Graph?
#
# A computation graph represents mathematical operations as a directed graph:
# - **Nodes**: Variables (inputs, intermediate values, outputs)
# - **Edges**: Operations between variables
# - **Direction**: The flow of computation from inputs to outputs
#
# Let's visualize a computation graph for a simple function:
# $$f(x) = x^2 + 2x + 1$$

# %%
# Function to visualize a computation graph
def visualize_computation_graph(nodes, edges, edge_labels=None, node_labels=None):
    """
    Visualize a computation graph
    
    Parameters:
    - nodes: List of node names
    - edges: List of tuples (source, target) representing connections
    - edge_labels: Dictionary of edge labels {(source, target): operation}
    - node_labels: Dictionary of node labels {node: value}
    """
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42)  # positions for all nodes
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')
    
    # Draw edges with arrows
    nx.draw_networkx_edges(G, pos, width=2, arrowsize=20, arrowstyle='->')
    
    # Draw node labels
    if node_labels:
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12)
    else:
        nx.draw_networkx_labels(G, pos, font_size=12)
    
    # Draw edge labels
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# %%
# Create the computation graph for f(x) = x^2 + 2x + 1
x_value = 2.0
x = torch.tensor(x_value, requires_grad=True)

# Build the graph step by step
y1 = x * x           # x^2
y2 = 2 * x           # 2x
y3 = y1 + y2         # x^2 + 2x
y4 = y3 + 1          # x^2 + 2x + 1

# Define nodes and edges for visualization
nodes = ['x', 'y1', 'y2', 'y3', 'y4']
edges = [('x', 'y1'), ('x', 'y2'), ('y1', 'y3'), ('y2', 'y3'), ('y3', 'y4')]

# Define edge labels (operations)
edge_labels = {
    ('x', 'y1'): 'square',
    ('x', 'y2'): '* 2',
    ('y1', 'y3'): '+',
    ('y2', 'y3'): '+',
    ('y3', 'y4'): '+ 1'
}

# Define node labels with values
node_labels = {
    'x': f'x = {x.item()}',
    'y1': f'y1 = x² = {y1.item()}',
    'y2': f'y2 = 2x = {y2.item()}',
    'y3': f'y3 = x² + 2x = {y3.item()}',
    'y4': f'y4 = x² + 2x + 1 = {y4.item()}'
}

# Visualize the graph
visualize_computation_graph(nodes, edges, edge_labels, node_labels)

print(f"Function value at x = {x_value}: {y4.item()}")

# %% [markdown]
# ### 1.2 Exercise: Build Your Own Computation Graph
#
# Now it's your turn to build a computation graph for the function:
# $$g(x) = 3x^2 - 4x + 2$$
#
# 1. Create the variables and operations needed
# 2. Define the nodes, edges, and labels for visualization
# 3. Visualize the graph using the helper function

# %%
# Your solution here

# %% [markdown]
# ### Solution 1.2

# %%
# Create the computation graph for g(x) = 3x^2 - 4x + 2
x_value = 2.0
x = torch.tensor(x_value, requires_grad=True)

# Build the graph step by step
x_squared = x * x        # x^2
term1 = 3 * x_squared    # 3x^2
term2 = 4 * x            # 4x
term3 = term1 - term2    # 3x^2 - 4x
g = term3 + 2            # 3x^2 - 4x + 2

# Define nodes and edges for visualization
nodes = ['x', 'x_squared', 'term1', 'term2', 'term3', 'g']
edges = [
    ('x', 'x_squared'), 
    ('x_squared', 'term1'), 
    ('x', 'term2'), 
    ('term1', 'term3'), 
    ('term2', 'term3'), 
    ('term3', 'g')
]

# Define edge labels
edge_labels = {
    ('x', 'x_squared'): 'square',
    ('x_squared', 'term1'): '* 3',
    ('x', 'term2'): '* 4',
    ('term1', 'term3'): '-',
    ('term2', 'term3'): '-',
    ('term3', 'g'): '+ 2'
}

# Define node labels with values
node_labels = {
    'x': f'x = {x.item()}',
    'x_squared': f'x² = {x_squared.item()}',
    'term1': f'3x² = {term1.item()}',
    'term2': f'4x = {term2.item()}',
    'term3': f'3x² - 4x = {term3.item()}',
    'g': f'g = 3x² - 4x + 2 = {g.item()}'
}

# Visualize the graph
visualize_computation_graph(nodes, edges, edge_labels, node_labels)

print(f"Function value at x = {x_value}: {g.item()}")

# %% [markdown]
# ## 2. Automatic Differentiation
#
# ### 2.1 Understanding Autograd
#
# PyTorch's automatic differentiation (autograd) allows us to compute derivatives automatically. This works by:
#
# 1. **Building a dynamic computation graph** as operations are performed
# 2. **Recording operations** on variables that have `requires_grad=True`
# 3. **Applying the chain rule** to compute gradients during backward pass
#
# The **chain rule** is a fundamental concept in calculus that allows us to compute derivatives of composite functions:
#
# For a composite function $f(g(x))$, the derivative is:
# $$\frac{d}{dx}f(g(x)) = \frac{df}{dg} \cdot \frac{dg}{dx}$$
#
# For example, if $y = f(x) = x^2 + 2x + 1$, we can break it down:
# - $y_1 = x^2$, so $\frac{dy_1}{dx} = 2x$
# - $y_2 = 2x$, so $\frac{dy_2}{dx} = 2$
# - $y_3 = y_1 + y_2$, so $\frac{dy_3}{dy_1} = 1$ and $\frac{dy_3}{dy_2} = 1$
# - $y_4 = y_3 + 1$, so $\frac{dy_4}{dy_3} = 1$
#
# Using the chain rule:
# $$\frac{dy_4}{dx} = \frac{dy_4}{dy_3} \cdot \left(\frac{dy_3}{dy_1} \cdot \frac{dy_1}{dx} + \frac{dy_3}{dy_2} \cdot \frac{dy_2}{dx}\right) = 1 \cdot (1 \cdot 2x + 1 \cdot 2) = 2x + 2$$
#
# PyTorch automates this process by traversing the computation graph backward from the output to the inputs.

# %%
# Let's compute the gradient of our function f(x) = x^2 + 2x + 1
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 2*x + 1

# Compute the gradient
y.backward()

# Access the gradient
print(f"Function: f(x) = x^2 + 2x + 1")
print(f"Value at x = 2: {y.item()}")
print(f"Gradient at x = 2: {x.grad.item()}")
print(f"Expected gradient (2x + 2): {2*2 + 2}")

# %% [markdown]
# ### 2.2 Gradients in Multiple Dimensions
#
# For functions with multiple inputs, autograd computes partial derivatives with respect to each input.
#
# Let's look at a 2D function: $f(x, y) = x^2 + y^2$

# %%
# Create a function to visualize 2D functions and their gradients
def visualize_function_and_gradient_2d(f, x_range, y_range, point=None, num_points=20):
    """
    Visualize a 2D function and its gradient
    
    Parameters:
    - f: Function that takes x, y and returns z
    - x_range: Tuple (x_min, x_max)
    - y_range: Tuple (y_min, y_max)
    - point: Optional tuple (x0, y0, z0) to mark a specific point
    - num_points: Number of points in each dimension for the grid
    """
    # Create a grid of points
    x_vals = np.linspace(x_range[0], x_range[1], num_points)
    y_vals = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Compute function values
    Z = np.zeros_like(X)
    grad_x = np.zeros_like(X)
    grad_y = np.zeros_like(X)
    
    # Compute function values and gradients using autograd
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_tensor = torch.tensor(X[i, j], requires_grad=True)
            y_tensor = torch.tensor(Y[i, j], requires_grad=True)
            
            # Compute function value
            z = f(x_tensor, y_tensor)
            Z[i, j] = z.item()
            
            # Compute gradients
            z.backward()
            grad_x[i, j] = x_tensor.grad.item()
            grad_y[i, j] = y_tensor.grad.item()
            
            # Reset gradients for next iteration
            x_tensor.grad.zero_()
            y_tensor.grad.zero_()
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 5))
    
    # 3D surface plot
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8)
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax1.set_zlabel('$f(x,y)$')
    ax1.set_title('3D Surface Plot')
    
    # Mark the specific point if provided
    if point:
        x0, y0, z0 = point
        ax1.scatter([x0], [y0], [z0], color='red', s=50)
    
    # Contour plot with gradient field
    ax2 = fig.add_subplot(1, 3, 2)
    contour = ax2.contourf(X, Y, Z, 20, cmap=cm.coolwarm)
    ax2.quiver(X, Y, grad_x, grad_y, color='black', scale=50)
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$y$')
    ax2.set_title('Contour Plot with Gradient Field')
    plt.colorbar(contour, ax=ax2)
    
    # Mark the specific point if provided
    if point:
        x0, y0, _ = point
        ax2.plot(x0, y0, 'ro', markersize=10)
        # Get the gradient at the specific point
        x_tensor = torch.tensor(x0, requires_grad=True)
        y_tensor = torch.tensor(y0, requires_grad=True)
        z = f(x_tensor, y_tensor)
        z.backward()
        grad_x0 = x_tensor.grad.item()
        grad_y0 = y_tensor.grad.item()
        ax2.quiver(x0, y0, grad_x0, grad_y0, color='red', scale=10)
        ax2.text(x0+0.1, y0+0.1, f'∇f({x0},{y0}) = [{grad_x0:.2f}, {grad_y0:.2f}]', fontsize=10)
    
    # Gradient magnitude plot
    ax3 = fig.add_subplot(1, 3, 3)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    contour2 = ax3.contourf(X, Y, grad_magnitude, 20, cmap='viridis')
    ax3.set_xlabel('$x$')
    ax3.set_ylabel('$y$')
    ax3.set_title('Gradient Magnitude')
    plt.colorbar(contour2, ax=ax3)
    
    plt.tight_layout()
    plt.show()

# %%
# Define a simple 2D function
def f_2d(x, y):
    return x**2 + y**2

# Point to highlight
x0, y0 = 2.0, 3.0
z0 = f_2d(x0, y0)

# Visualize
visualize_function_and_gradient_2d(f_2d, (-4, 4), (-4, 4), point=(x0, y0, z0))

# %% [markdown]
# ### 2.3 Exercise: Compute and Visualize Gradients
#
# Now it's your turn! For the function $h(x, y) = x^2 - y^2$:
#
# 1. Define the function
# 2. Compute the gradient at the point (1, 2) using PyTorch's autograd
# 3. Visualize the function and its gradient using our helper function
# 4. Compare the autograd result with the analytical gradient

# %%
# Your solution here

# %% [markdown]
# ### Solution 2.3

# %%
# 1. Define the function
h = lambda x, y: x**2 - y**2

# 2. Compute the gradient at (1, 2) using PyTorch's autograd
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)

z = x**2 - y**2

z.backward()

print(f"Function value h(1, 2) = {z.item()}")
print(f"∂h/∂x at (1, 2) using autograd: {x.grad.item()}")
print(f"∂h/∂y at (1, 2) using autograd: {y.grad.item()}")

# Analytical gradient
print(f"Analytical ∂h/∂x at (1, 2): {2*1}")
print(f"Analytical ∂h/∂y at (1, 2): {-2*2}")

# 3. Visualize the function and its gradient
x0, y0 = 1.0, 2.0
z0 = h(x0, y0)

visualize_function_and_gradient_2d(h, (-3, 3), (-3, 3), point=(x0, y0, z0))

# %% [markdown]
# ## 3. The Benefit of Autograd: Computing Gradients for Complex Functions
#
# Autograd's real power is its ability to handle complex functions where calculating derivatives by hand would be difficult.

# %% [markdown]
# ### 3.1 Example: Complex Function
#
# Let's look at a more complex function: $f(x, y) = \sin(x^2) \cdot \cos(y) + e^{x \cdot y}$

# %%
# Define the complex function
def f_complex(x, y):
    return torch.sin(x**2) * torch.cos(y) + torch.exp(x * y)

# Compute gradients at (1, 0.5)
x0, y0 = 1.0, 0.5
z0 = f_complex(torch.tensor(x0), torch.tensor(y0)).item()

# Visualize
visualize_function_and_gradient_2d(f_complex, (-2, 2), (-2, 2), point=(x0, y0, z0))

# %% [markdown]
# ### 3.2 Exercise: Your Complex Function
#
# Now it's your turn to experiment with autograd on a function of your choosing!
#
# 1. Create your own complex function (you can check its 3D shape at https://www.desmos.com/3d)
# 2. Compute its gradient at a point of your choice using PyTorch's autograd
# 3. Visualize your function and its gradient
#
# Be creative! Try combinations of trigonometric functions, exponentials, polynomials, etc.

# %%
# Your solution here

# %% [markdown]
# ### Solution 3.2 (Example)

# %%
# 1. Create a complex function
# Let's try: f(x, y) = sin(x*y) * cos(x-y) + log(1 + x^2 + y^2)

# Define the function
def my_complex_f(x, y):
    return torch.sin(x*y) * torch.cos(x-y) + torch.log(1 + x**2 + y**2)

# Choose a point
x0, y0 = 0.8, 1.2
z0 = my_complex_f(torch.tensor(x0), torch.tensor(y0)).item()

# Compute gradients and visualize
visualize_function_and_gradient_2d(my_complex_f, (-2, 2), (-2, 2), point=(x0, y0, z0))

# %% [markdown]
# ## 4. Working with Autodiff
#
# ### 4.1 Computing Gradients for Different Inputs
#
# In deep learning, we often use the same function with different inputs. Let's see how to compute gradients
# for the same function at different input points:

# %%
# Function: f(x, y) = x^2 * y + y^3

def f_demo(x, y):
    return x**2 * y + y**3

# First point: (2, 1)
x1 = torch.tensor(2.0, requires_grad=True)
y1 = torch.tensor(1.0, requires_grad=True)

f1 = f_demo(x1, y1)
f1.backward()

print("Point (2, 1):")
print(f"Function value: {f1.item()}")
print(f"∂f/∂x: {x1.grad.item()}")
print(f"∂f/∂y: {y1.grad.item()}")

# To compute gradients at a different point, we need to zero out previous gradients
x1.grad.zero_()
y1.grad.zero_()

# Alternative: create new tensors for the second point
x2 = torch.tensor(1.0, requires_grad=True)
y2 = torch.tensor(2.0, requires_grad=True)

f2 = f_demo(x2, y2)
f2.backward()

print("\nPoint (1, 2):")
print(f"Function value: {f2.item()}")
print(f"∂f/∂x: {x2.grad.item()}")
print(f"∂f/∂y: {y2.grad.item()}")

# Visualize both points on the same graph
fig = plt.figure(figsize=(12, 6))

# Create the contour plot
x_vals = np.linspace(-2, 3, 30)
y_vals = np.linspace(-2, 3, 30)
X, Y = np.meshgrid(x_vals, y_vals)

# Compute function values
Z = np.zeros_like(X)
grad_x = np.zeros_like(X)
grad_y = np.zeros_like(X)

# Compute function values and gradients using autograd
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x_tensor = torch.tensor(X[i, j], requires_grad=True)
        y_tensor = torch.tensor(Y[i, j], requires_grad=True)
        
        # Compute function value
        z = f_demo(x_tensor, y_tensor)
        Z[i, j] = z.item()
        
        # Compute gradients
        z.backward()
        grad_x[i, j] = x_tensor.grad.item()
        grad_y[i, j] = y_tensor.grad.item()
        
        # Reset gradients for next iteration
        x_tensor.grad.zero_()
        y_tensor.grad.zero_()

# Plot
contour = plt.contourf(X, Y, Z, 20, cmap=cm.coolwarm)
plt.quiver(X, Y, grad_x, grad_y, color='black', scale=300)

# Mark points and their gradients
plt.plot(2, 1, 'ro', markersize=10)
# Compute gradients at (2, 1)
x_tensor = torch.tensor(2.0, requires_grad=True)
y_tensor = torch.tensor(1.0, requires_grad=True)
z = f_demo(x_tensor, y_tensor)
z.backward()
gx1, gy1 = x_tensor.grad.item(), y_tensor.grad.item()
plt.quiver(2, 1, gx1, gy1, color='red', scale=50)
plt.text(2.1, 1.1, f'∇f(2,1) = [{gx1}, {gy1}]', fontsize=10)

plt.plot(1, 2, 'go', markersize=10)
# Compute gradients at (1, 2)
x_tensor = torch.tensor(1.0, requires_grad=True)
y_tensor = torch.tensor(2.0, requires_grad=True)
z = f_demo(x_tensor, y_tensor)
z.backward()
gx2, gy2 = x_tensor.grad.item(), y_tensor.grad.item()
plt.quiver(1, 2, gx2, gy2, color='green', scale=50)
plt.text(1.1, 2.1, f'∇f(1,2) = [{gx2}, {gy2}]', fontsize=10)

plt.colorbar(contour)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Function $f(x,y) = x^2y + y^3$ with Gradients at Different Points')
plt.grid(True)
plt.show()

# %% [markdown]
# ### 4.2 Using requires_grad=False
#
# Sometimes, we don't need to compute gradients for all inputs. We can control this with `requires_grad=False`:

# %%
# Create tensors with different requires_grad settings
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=False)  # No gradients for y

# Compute function
f = x**2 * y

# Backpropagate
f.backward()

print(f"Function f = x^2 * y at (2, 3): {f.item()}")
print(f"∂f/∂x: {x.grad.item()}")
print(f"∂f/∂y: {'Not computed (requires_grad=False)'}")

# We can also temporarily disable gradient tracking with torch.no_grad()
with torch.no_grad():
    # No operations here will track gradients
    z = x * y
    print(f"\nComputed z = x * y = {z.item()} without tracking gradients")
    print(f"z.requires_grad: {z.requires_grad}")

# %% [markdown]
# ### 4.3 Exercise: Computing and Comparing Gradients
#
# In this exercise, you'll explore how gradients change for different input values:
#
# 1. Pick a function of your choice (e.g., $f(x, y) = \sin(x + y) * e^{xy}$)
# 2. Compute its gradients at three different points
# 3. Visualize how the gradient field changes across the input space
# 4. Identify regions where the gradient is large or small

# %%
# Your solution here

# %% [markdown]
# ### Solution 4.3

# %%
# 1. Choose a function
# f(x, y) = sin(x + y) * e^(xy)

# Define the function
def f_exercise(x, y):
    return torch.sin(x + y) * torch.exp(x * y)

# 2. Compute gradients at three different points
def compute_gradient_at_point(f, x_val, y_val):
    x = torch.tensor(x_val, requires_grad=True)
    y = torch.tensor(y_val, requires_grad=True)
    
    z = f(x, y)
    z.backward()
    
    return {
        'point': (x_val, y_val),
        'f_value': z.item(),
        'grad_x': x.grad.item(),
        'grad_y': y.grad.item()
    }

# Compute at three points
point1 = compute_gradient_at_point(f_exercise, 0.0, 0.0)
point2 = compute_gradient_at_point(f_exercise, 1.0, 1.0)
point3 = compute_gradient_at_point(f_exercise, -1.0, 1.0)

# Print results
for i, point in enumerate([point1, point2, point3], 1):
    print(f"Point {i}: ({point['point'][0]}, {point['point'][1]})")
    print(f"  Function value: {point['f_value']:.4f}")
    print(f"  ∂f/∂x: {point['grad_x']:.4f}")
    print(f"  ∂f/∂y: {point['grad_y']:.4f}")
    print()

# 3. Visualize gradients across input space
# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Create a grid of points
x_vals = np.linspace(-2, 2, 30)
y_vals = np.linspace(-2, 2, 30)
X, Y = np.meshgrid(x_vals, y_vals)

# Compute function values and gradients
Z = np.zeros_like(X)
grad_x = np.zeros_like(X)
grad_y = np.zeros_like(X)
grad_magnitude = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x_tensor = torch.tensor(X[i, j], requires_grad=True)
        y_tensor = torch.tensor(Y[i, j], requires_grad=True)
        
        # Compute function value
        z = f_exercise(x_tensor, y_tensor)
        Z[i, j] = z.item()
        
        # Compute gradients
        z.backward()
        grad_x[i, j] = x_tensor.grad.item()
        grad_y[i, j] = y_tensor.grad.item()
        grad_magnitude[i, j] = np.sqrt(x_tensor.grad.item()**2 + y_tensor.grad.item()**2)
        
        # Reset gradients for next iteration
        x_tensor.grad.zero_()
        y_tensor.grad.zero_()

# Contour plot with gradient field
contour = ax1.contourf(X, Y, Z, 20, cmap=cm.coolwarm)
ax1.quiver(X, Y, grad_x, grad_y, color='black', scale=50)

# Mark the three points
for point, color, marker in zip([point1, point2, point3], ['red', 'green', 'blue'], ['o', 's', '^']):
    x_val, y_val = point['point']
    ax1.plot(x_val, y_val, color=color, marker=marker, markersize=10)
    ax1.quiver(x_val, y_val, point['grad_x'], point['grad_y'], color=color, scale=20)
    ax1.text(x_val+0.1, y_val+0.1, f'∇f{point["point"]}', color=color, fontsize=10)

fig.colorbar(contour, ax=ax1)
ax1.set_xlabel('$x$')
ax1.set_ylabel('$y$')
ax1.set_title('Function $f(x,y) = \sin(x+y) \cdot e^{xy}$ with Gradient Field')

# 4. Plot gradient magnitude to identify regions of large/small gradients
contour2 = ax2.contourf(X, Y, grad_magnitude, 20, cmap='viridis')
fig.colorbar(contour2, ax=ax2)
ax2.set_xlabel('$x$')
ax2.set_ylabel('$y$')
ax2.set_title('Gradient Magnitude $|\\nabla f(x,y)|$')

# Mark regions of interest
# Find location of maximum gradient in our grid
i_max, j_max = np.unravel_index(grad_magnitude.argmax(), grad_magnitude.shape)
x_max, y_max = X[i_max, j_max], Y[i_max, j_max]
ax2.plot(x_max, y_max, 'r*', markersize=15)
ax2.text(x_max+0.1, y_max+0.1, 'Max gradient', color='red', fontsize=12)

# Find a location with small gradient (near origin)
ax2.plot(0, 0, 'bo', markersize=10)
ax2.text(0.1, 0.1, 'Small gradient', color='blue', fontsize=12)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Conclusion
# 
# In this lecture, we've explored:
# 
# 1. **Computation Graphs**: How PyTorch represents mathematical operations as a network
# 
# 2. **Automatic Differentiation**: How PyTorch computes gradients by traversing the computation graph and applying the chain rule
# 
# 3. **Benefits of Autograd for Complex Functions**: How autograd helps with complicated derivatives
# 
# 4. **Working with Autodiff**: Practical considerations for gradient computation
# 
# These concepts form the foundation of deep learning optimization. When we train neural networks, we're essentially:
# 
# 1. Building a computation graph (the forward pass)
# 2. Computing gradients (the backward pass)
# 3. Updating parameters based on these gradients (optimization)
# 
# Understanding these concepts will help you debug and improve your neural network implementations!