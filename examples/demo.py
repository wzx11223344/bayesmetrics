"""
贝叶斯计量经济学综合示例
========================

演示各模型和采样器的使用:
1. 贝叶斯线性回归 (共轭先验 + Gibbs)
2. 贝叶斯 Logit (MH 采样)
3. 贝叶斯 VAR (Minnesota 先验 + Gibbs)
4. HMC vs MH 采样效率对比
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

np.random.seed(42)

# ==================== 1. 贝叶斯线性回归 ====================
print("=" * 60)
print("1. 贝叶斯线性回归 (共轭先验 + Gibbs 采样)")
print("=" * 60)

from bayesmetrics import BayesianLinearRegression

n = 200
X = np.random.randn(n, 2)
true_beta = np.array([2.0, 1.5, -0.8])  # [const, x1, x2]
y = true_beta[0] + true_beta[1] * X[:, 0] + true_beta[2] * X[:, 1] + np.random.randn(n) * 1.5

blr = BayesianLinearRegression()
blr.fit(X, y, n_iter=3000, n_burnin=500, var_names=["edu", "exp"], progress=False)
print(blr.summary())

# ==================== 2. 贝叶斯 Logit ====================
print("\n" + "=" * 60)
print("2. 贝叶斯 Logit (MH 采样)")
print("=" * 60)

from bayesmetrics import BayesianLogit

X_b = np.random.randn(500, 2)
eta = 0.5 + 1.8 * X_b[:, 0] - 1.2 * X_b[:, 1]
prob = 1 / (1 + np.exp(-eta))
y_b = (np.random.rand(500) < prob).astype(float)

blogit = BayesianLogit()
blogit.fit(X_b, y_b, n_iter=5000, n_burnin=1000, var_names=["income", "education"], progress=False)
print(blogit.summary())

# ==================== 3. 贝叶斯 VAR ====================
print("\n" + "=" * 60)
print("3. 贝叶斯 VAR(2) (Minnesota 先验 + Gibbs)")
print("=" * 60)

from bayesmetrics import BayesianVAR

# 生成 AR(1) 型 VAR 数据 (强平稳)
T = 200
M = 3
data = np.zeros((T, M))
data[0] = np.random.randn(M) * 0.3
for t in range(1, T):
    data[t] = 0.6 * data[t-1] + np.random.randn(M) * 0.5

bvar = BayesianVAR(lags=1)
bvar.fit(data, n_iter=2000, n_burnin=500, lambda1=0.05, progress=False)
print(bvar.summary())

# 脉冲响应函数
irf = bvar.impulse_response(shock_var=0, horizon=8)
irf_mean = np.mean(irf, axis=0)
irf_upper = np.percentile(irf, 95, axis=0)
irf_lower = np.percentile(irf, 5, axis=0)

h_last = min(7, irf_mean.shape[1] - 1)
print(f"变量1 冲击的脉冲响应 (后验均值):")
print(f"  h=0:  {irf_mean[0, 0]:.3f}  h=4:  {irf_mean[0, 4]:.3f}  h=7:  {irf_mean[0, h_last]:.3f}")

# ==================== 4. 采样器诊断 ====================
print("\n" + "=" * 60)
print("4. MCMC 收敛诊断")
print("=" * 60)

from bayesmetrics import gelman_rubin, effective_sample_size
from bayesmetrics.diagnostics import print_summary

# 用 3 条链的 BLR 采样做诊断
X_multi = np.column_stack([np.ones(100), np.random.randn(100, 2)])
y_multi = 2 + 1.5 * X_multi[:, 1] - 0.8 * X_multi[:, 2] + np.random.randn(100) * 0.5

blr2 = BayesianLinearRegression()
blr2.fit(X_multi[:, 1:], y_multi, n_iter=2000, n_burnin=200, progress=False)

print_summary(blr2.beta_samples, var_names=["const", "x1", "x2"])
