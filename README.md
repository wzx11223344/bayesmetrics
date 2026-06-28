# BayesianEconometrics — 从零实现的贝叶斯计量经济学引擎

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![NumPy](https://img.shields.io/badge/NumPy-✓-013243.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-✓-8CAAE6.svg)](https://scipy.org/)

**纯 Python + NumPy/SciPy 实现的贝叶斯计量经济学工具包**。

从底层算法构建了 4 种 **MCMC 采样器**和 4 个**贝叶斯计量模型**，包含完整的**收敛诊断体系**。

---

## 🧠 为什么这很硬核

贝叶斯推断是计量经济学的前沿领域。本项目**从零实现了 Stan/PyMC 的核心算法**：

| 组件 | 算法 | 与 Stan 等价物 |
|------|------|:---:|
| **NUTS** | No-U-Turn Sampler | Stan 默认采样器 |
| **HMC** | Hamiltonian Monte Carlo | Stan 备选采样器 |
| **Gibbs** | 全条件分布采样 | BUGS/JAGS 核心 |
| **M-H** | 自适应 Metropolis-Hastings | 经典 MCMC |
| **R-hat** | Gelman-Rubin 收敛诊断 | Stan 输出指标 |
| **ESS** | 有效样本量 (Geyer 方法) | Stan 输出指标 |
| **Minnesota** | BVAR 先验 | 央行标准工具 |

## 📦 安装

```bash
pip install bayesmetrics
```

## 🚀 快速开始

### 贝叶斯线性回归

```python
from bayesmetrics import BayesianLinearRegression
import numpy as np

X = np.random.randn(200, 2)
y = 2 + 1.5*X[:,0] - 0.8*X[:,1] + np.random.randn(200)*1.5

blr = BayesianLinearRegression()
blr.fit(X, y)
print(blr.summary())
# 输出后验均值、标准差、HDI 区间
```

### 贝叶斯 Logit（MH 采样）

```python
from bayesmetrics import BayesianLogit

blogit = BayesianLogit()
blogit.fit(X, y_binary, n_iter=5000, n_burnin=1000)
print(blogit.summary())  # 含边际效应
```

### 贝叶斯 VAR（Minnesota 先验）

```python
from bayesmetrics import BayesianVAR

bvar = BayesianVAR(lags=2)
bvar.fit(macro_data, lambda1=0.2)
print(bvar.summary())

# 脉冲响应函数
irf = bvar.impulse_response(shock_var=0, horizon=20)
```

### NUTS 采样器（直接使用）

```python
from bayesmetrics import NUTS

def log_post(theta):
    return -0.5 * np.sum(theta**2)  # 标准正态后验

def grad_log_post(theta):
    return -theta

nuts = NUTS(log_post, grad_log_post, np.zeros(3))
samples = nuts.sample(n_iter=2000, n_burnin=500)
```

### 收敛诊断

```python
from bayesmetrics import gelman_rubin, effective_sample_size
from bayesmetrics.diagnostics import print_summary

print_summary(samples, var_names=["alpha", "beta", "gamma"])
```

## 📚 模块参考

| 模块 | 内容 | 技术要点 |
|------|------|---------|
| `samplers` | MH, Gibbs, HMC, NUTS | Leapfrog 积分、二叉树递归、自适应步长 (Dual Averaging) |
| `models` | BLR, Logit, Probit, VAR | 共轭先验、Minnesota 先验、IRF、Cholesky 分解 |
| `diagnostics` | R-hat, ESS, ACF | 链内/链间方差分解、Geyer 单调序列截断 |
| `priors` | Normal, IG, Wishart, MVN | 对数密度、共轭关系 |

## 🎯 采样器对比

| 特性 | MH | Gibbs | HMC | NUTS |
|------|:--:|:-----:|:---:|:----:|
| 需梯度 | ❌ | ❌ | ✅ | ✅ |
| 自适应 | 提案协方差 | — | — | 步长 (Dual Avg) |
| 高维效率 | 低 | 中 | 高 | 极高 |
| 调参难度 | 中 | 低 | 高 | 极低 |
| 理论收敛 | 几何 | 几何 | 几何 | 几何 |

## 📖 理论背景

- Hoffman & Gelman (2014). *The No-U-Turn Sampler*. JMLR 15:1593-1623.
- Neal (2011). *MCMC using Hamiltonian dynamics*. Handbook of MCMC.
- Koop (2003). *Bayesian Econometrics*. Wiley.
- Litterman (1986). *Forecasting with Bayesian Vector Autoregressions*.
- Gelman et al. (2013). *Bayesian Data Analysis*. CRC Press.
- Betancourt (2017). *A Conceptual Introduction to Hamiltonian Monte Carlo*.

## 📄 许可证

MIT License © 2024 wzx11223344
