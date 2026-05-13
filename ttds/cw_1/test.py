import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Set up grid
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Common mean
mu = [0, 0]

# Define different covariance matrices
cov1 = [[1, 0],     # No correlation, equal variance
        [0, 1]]

cov2 = [[2, 0],     # No correlation, different variances (stretched)
        [0, 0.5]]

cov3 = [[1, 0.8],   # Positive correlation
        [0.8, 1]]

cov4 = [[1, -0.8],  # Negative correlation
        [-0.8, 1]]

covs = [cov1, cov2, cov3, cov4]
titles = [
    'Independent (σ²=1, ρ=0)',
    'Uncorrelated, unequal variance',
    'Positive correlation (ρ=0.8)',
    'Negative correlation (ρ=-0.8)'
]

# Plot
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.ravel()

for i, (cov, title) in enumerate(zip(covs, titles)):
    rv = multivariate_normal(mu, cov)
    Z = rv.pdf(pos)
    axes[i].contour(X, Y, Z, levels=10, cmap='viridis')
    axes[i].set_title(title)
    axes[i].set_aspect('equal')
    axes[i].grid(True)

plt.tight_layout()
plt.show()



