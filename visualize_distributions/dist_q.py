import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 1. 정규분포 모수 설정
mu = 3.7  # 평균 (Mean)
sigma = 1.5  # 표준편차 (Standard Deviation)
sample_size = 500000  # 큰 표본 크기를 사용하여 분포를 부드럽게 표현

# 2. 잠재표현 y 생성
np.random.seed(42)
y = np.random.normal(loc=mu, scale=sigma, size=sample_size)

# 3. 두 가지 양자화 예측값 y_hat 계산
# ----------------------------------------------------
# Method 1: Centered Quantization (오차 분석에 유리)
# y_hat_1 = round(y - mu) + mu
y_hat_1 = np.round(y - mu) + mu

# Method 2: Direct Quantization (가장 일반적인 반올림)
# y_hat_2 = round(y)
y_hat_2 = np.round(y)

# 4. 양자화 오차(Quantization Error) 계산
# Error e = y - y_hat
error_1 = y - y_hat_1
error_2 = y - y_hat_2

# 5. 시각화 (Matplotlib 및 Seaborn Histogram)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 전체 제목 설정
fig.suptitle(
    r"Distribution of Quantization Error $e = y - \hat{y}$ ($\mu=3.7, \sigma=1.5$)",
    fontsize=16,
)

# ----------------------------------------------------
# Subplot 1: Centered Quantization Error
# ----------------------------------------------------
sns.histplot(
    error_1, bins=100, kde=True, stat="density", color="darkorange", alpha=0.7, ax=ax1
)
ax1.set_title(r"Error: $e_1 = y - (\text{round}(y - \mu) + \mu)$", fontsize=12)
ax1.set_xlabel(r"Quantization Error $e_1$ Value")
ax1.set_ylabel("Probability Density")
ax1.axvline(-0.5, color="r", linestyle="--", alpha=0.7, label=r"Boundary (-0.5)")
ax1.axvline(0.5, color="r", linestyle="--", alpha=0.7, label=r"Boundary (0.5)")
ax1.set_xlim(-0.6, 0.6)  # 오차 구간 강조
ax1.grid(axis="y", alpha=0.5)
ax1.legend()

# ----------------------------------------------------
# Subplot 2: Direct Quantization Error
# ----------------------------------------------------
sns.histplot(
    error_2, bins=100, kde=True, stat="density", color="purple", alpha=0.7, ax=ax2
)
ax2.set_title(r"Error: $e_2 = y - \text{round}(y)$", fontsize=12)
ax2.set_xlabel(r"Quantization Error $e_2$ Value")
ax2.set_ylabel("Probability Density")
ax2.axvline(-0.5, color="r", linestyle="--", alpha=0.7)
ax2.axvline(0.5, color="r", linestyle="--", alpha=0.7)
ax2.set_xlim(-0.6, 0.6)  # 오차 구간 강조
ax2.grid(axis="y", alpha=0.5)

# 레이아웃 조정
plt.subplots_adjust(top=0.85, wspace=0.3)
plt.show()
