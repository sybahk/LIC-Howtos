from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm

# 1. Normal Distribution Parameters
mu = 3.7  # Mean
sigma = 1.5  # Standard Deviation
sample_size = 500000  # Sample size

# 2. Generate Latent Representation y
np.random.seed(42)
y = np.random.normal(loc=mu, scale=sigma, size=sample_size)

# 3. Calculate Transformed Distributions

# Distribution 1: Centered Quantization (Discrete)
# y_hat = round(y - mu) + mu
y_hat = np.round(y - mu) + mu

# Distribution 2: Adding Uniform Noise (Continuous)
# eta ~ U(-1/2, 1/2)
eta = np.random.uniform(low=-0.5, high=0.5, size=sample_size)
y_tilde = y + eta

# 4. Calculate Density for Discrete Spike Plot (y_hat)

# --- Distribution 1: y_hat ---
counts_y_hat = Counter(y_hat)
total_y_hat = len(y_hat)
x_y_hat = sorted(counts_y_hat.keys())
density_y_hat = [counts_y_hat[i] / total_y_hat for i in x_y_hat]


# 5. Visualization (Matplotlib Spike Plot vs. Histogram)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Define common y-limit for comparison
max_density_spike = max(density_y_hat)
max_density_pdf = norm.pdf(mu, loc=mu, scale=sigma)  # Original PDF peak as reference
y_max_limit = max(max_density_pdf, max_density_spike) * 1.1

# Define common x-range
x_min = min(y.min(), y_hat.min(), y_tilde.min()) - 1
x_max = max(y.max(), y_hat.max(), y_tilde.max()) + 1

# ----------------------------------------------------
# Subplot 1: y_hat = round(y - mu) + mu (Discrete Spike Plot)
# ----------------------------------------------------
ax1.vlines(
    x_y_hat, 0, density_y_hat, colors="C0", lw=5, alpha=0.8, label=r"$\hat{y}$ Density"
)
ax1.scatter(x_y_hat, density_y_hat, color="C0", zorder=5)
# Title with % formatting and no \mathbf
ax1.set_title(
    r"Distribution of Quantized Latent Variable $\hat{y}$ ($\mu=%.1f$)" % mu,
    fontsize=12,
)
ax1.set_xlabel(r"$\hat{y} = \text{round}(y - \mu) + \mu$ Value")
ax1.set_ylabel("Probability Density (PMF)")
ax1.grid(axis="y", alpha=0.5)
ax1.set_xlim(x_min, x_max)
ax1.set_ylim(0, y_max_limit)
# X-ticks는 이산 값
ax1.set_xticks(x_y_hat)

# ----------------------------------------------------
# Subplot 2: y_tilde = y + eta (Continuous Distribution - Histogram)
# ----------------------------------------------------
# KDE 대신 bin size가 작은 histogram 사용
sns.histplot(y_tilde, ax=ax2, color="purple", bins=200, stat="density", alpha=0.5)
# Title with % formatting and no \mathbf
ax2.set_title(
    r"Distribution of Noisy Latent Variable $\tilde{y}$ ($\mu=%.1f$)" % mu, fontsize=12
)
ax2.set_xlabel(r"$\tilde{y} = y + \eta, \eta \sim U(-1/2, 1/2)$ Value")
ax2.set_ylabel("Probability Density")
ax2.grid(axis="y", alpha=0.5)
ax2.set_xlim(x_min, x_max)
ax2.set_ylim(0, y_max_limit)
# X-ticks는 연속 분포의 정수 범위
min_int = np.floor(x_min).astype(int)
max_int = np.ceil(x_max).astype(int)
ax2.set_xticks(np.arange(min_int, max_int + 1, 1))

# ----------------------------------------------------
# Overall Layout Adjustment
# ----------------------------------------------------
plt.subplots_adjust(top=0.85, wspace=0.25)

# Super Title with % formatting and no \mathbf
fig.suptitle(
    r"Comparison of Quantized Latent Variable $\hat{y}$ and Noisy Latent Variable $\tilde{y}$ ($\mu=%.1f, \sigma=%.1f$)"
    % (mu, sigma),
    fontsize=16,
)
plt.show()
