from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# 1. Normal Distribution Parameters
mu = 3.7  # Mean (비정수 값)
sigma = 1.5  # Standard Deviation
sample_size = 100000  # Sample size

# 2. Generate Latent Representation y
# y ~ N(mu, sigma^2)
np.random.seed(42)
y = np.random.normal(loc=mu, scale=sigma, size=sample_size)

# 3. Calculate Transformed Distributions
y_rounded = np.round(y)
y_mu_rounded = np.round(y - mu)
# 새로 추가된 분포: round(y - mu) + mu
y_mu_rounded_plus_mu = y_mu_rounded + mu

# 4. Calculate Density for Spike Plots (PMF)

# --- Distribution 1: round(y) ---
counts_y = Counter(y_rounded)
total_y = len(y_rounded)
x_y = sorted(counts_y.keys())
density_y = [counts_y[i] / total_y for i in x_y]

# --- Distribution 2: round(y - mu) ---
counts_ymu = Counter(y_mu_rounded)
total_ymu = len(y_mu_rounded)
x_ymu = sorted(counts_ymu.keys())
density_ymu = [counts_ymu[i] / total_ymu for i in x_ymu]

# --- Distribution 3: round(y - mu) + mu ---
counts_ymu_pmu = Counter(y_mu_rounded_plus_mu)
total_ymu_pmu = len(y_mu_rounded_plus_mu)
# 이 분포는 무거운 계산을 피하기 위해 x-ticks를 따로 설정하지 않고,
# float 값을 그대로 유지하여 spike plot을 그립니다.
x_ymu_pmu = sorted(counts_ymu_pmu.keys())
density_ymu_pmu = [counts_ymu_pmu[i] / total_ymu_pmu for i in x_ymu_pmu]


# 5. Visualization (4 Subplots)
fig, axes = plt.subplots(1, 4, figsize=(18, 6))
ax0, ax1, ax2, ax3 = axes

# Define limits and ranges
x_range = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
# y_max_limit 계산 (모든 분포의 최대값 포함)
max_density_spike = max(max(density_y), max(density_ymu))
max_density_pdf = norm.pdf(mu, loc=mu, scale=sigma)
y_max_limit = max(max_density_pdf, max_density_spike) * 1.1

# ----------------------------------------------------
# Subplot 0: Original Distribution (Continuous PDF)
# ----------------------------------------------------
pdf_values = norm.pdf(x_range, loc=mu, scale=sigma)
ax0.plot(
    x_range,
    pdf_values,
    color="green",
    lw=2,
    label=r"$N(\mu_{\hat{y}}, \sigma_{\hat{y}}^2)$ PDF",
)
ax0.axvline(mu, color="gray", linestyle="--", alpha=0.7, label=r"Mean $\mu_{\hat{y}}$")
ax0.set_title(r"Original Latent Distribution $y$", fontsize=12)
ax0.set_xlabel(r"Latent Value $y$")
ax0.set_ylabel("Probability Density")
ax0.grid(axis="y", alpha=0.5)
ax0.set_ylim(0, y_max_limit)
# 정수 x-ticks 설정
min_int = np.floor(min(x_range)).astype(int)
max_int = np.ceil(max(x_range)).astype(int)
ax0.set_xticks(np.arange(min_int, max_int + 1, 1))
ax0.legend()

# ----------------------------------------------------
# Subplot 1: round(y) (Discrete Spike Plot)
# ----------------------------------------------------
ax1.vlines(x_y, 0, density_y, colors="C0", lw=5, alpha=0.8)
ax1.scatter(x_y, density_y, color="C0", zorder=5)
ax1.set_title(r"Distribution of $\text{round}(y)$", fontsize=12)
ax1.set_xlabel(r"$\text{round}(y)$ Value")
ax1.set_ylabel("Probability Density (PMF)")
ax1.grid(axis="y", alpha=0.5)
ax1.set_xlim(min(x_y) - 1, max(x_y) + 1)
ax1.set_ylim(0, y_max_limit)
# 모든 관찰된 정수 값을 x-tick으로 설정
ax1.set_xticks(x_y)
ax1.plot(
    x_range,
    pdf_values,
    color="green",
    lw=2,
)
# ----------------------------------------------------
# Subplot 2: round(y - mu) (Discrete Spike Plot)
# ----------------------------------------------------
ax2.vlines(x_ymu, 0, density_ymu, colors="C1", lw=5, alpha=0.8)
ax2.scatter(x_ymu, density_ymu, color="C1", zorder=5)
ax2.set_title(r"Distribution of $\text{round}(y - \mu_{\hat{y}})$", fontsize=12)
ax2.set_xlabel(r"$\text{round}(y - \mu_{\hat{y}})$ Value")
ax2.set_ylabel("Probability Density (PMF)")
ax2.grid(axis="y", alpha=0.5)
ax2.set_xlim(min(x_ymu) - 1, max(x_ymu) + 1)
ax2.set_ylim(0, y_max_limit)
# 모든 관찰된 정수 값을 x-tick으로 설정
ax2.set_xticks(x_ymu)
zero_centered_pdf = norm.pdf(
    np.linspace(min(x_ymu) - 1, max(x_ymu) + 1, 500), loc=0.0, scale=sigma
)
ax2.plot(
    np.linspace(min(x_ymu) - 1, max(x_ymu) + 1, 500),
    zero_centered_pdf,
    color="green",
    lw=2,
)
# ----------------------------------------------------
# Subplot 3: round(y - mu) + mu (Discrete Spike Plot, NEW)
# ----------------------------------------------------
ax3.vlines(x_ymu_pmu, 0, density_ymu_pmu, colors="C2", lw=5, alpha=0.8)
ax3.scatter(x_ymu_pmu, density_ymu_pmu, color="C2", zorder=5)
ax3.set_title(
    r"Distribution of $\text{round}(y - \mu_{\hat{y}}) + \mu_{\hat{y}}$",
    fontsize=12,
)
ax3.set_xlabel(r"$\text{round}(y - \mu_{\hat{y}}) + \mu_{\hat{y}}$ Value")
ax3.set_ylabel("Probability Density (PMF)")
ax3.grid(axis="y", alpha=0.5)
ax3.set_xlim(min(x_ymu_pmu) - 1, max(x_ymu_pmu) + 1)
ax3.set_ylim(0, y_max_limit)
# x-ticks는 float이므로, 원래 round(y)의 정수 x-ticks와 동일한 범위 내의 정수만 표시
# (시각적 비교를 위해 정수 눈금만 표시하고 싶다면)
min_x_pmu_int = np.floor(min(x_ymu_pmu)).astype(int)
max_x_pmu_int = np.ceil(max(x_ymu_pmu)).astype(int)
ax3.set_xticks(np.arange(min_x_pmu_int, max_x_pmu_int + 1, 1))
ax3.plot(
    x_range,
    pdf_values,
    color="green",
    lw=2,
)
# ----------------------------------------------------
# Overall Layout Adjustment
# ----------------------------------------------------
# top: 제목 공간 확보, wspace: 플롯 간 간격 조정
plt.subplots_adjust(top=0.85, wspace=0.3)

# Super Title (전체 제목) 설정
fig.suptitle(
    "Comparison of Latent and Quantized Distributions ($\mu_{\hat{y}}={3.7}$, $\sigma_{\hat{y}}={1.5}$)",
    fontsize=16,
)
plt.show()
