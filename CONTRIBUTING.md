# 贡献指南

感谢你考虑为 BayesianEconometrics 做出贡献！贝叶斯统计是一个需要严谨数学基础的研究领域，请确保你的贡献满足方法论正确性要求。

## 行为准则

本项目遵守 [Contributor Covenant](CODE_OF_CONDUCT.md) 行为准则。

## 贡献范围

- **新采样器**：Slice Sampling、Riemannian HMC (RMHMC)、SGHMC
- **新模型**：贝叶斯 Probit 扩展、层次模型、贝叶斯状态空间模型
- **收敛诊断**：改进的 R-hat (Vehtari et al. 2021)、ESS 批量方法
- **文档与测试**：理论推导注释、基准测试、数值验证

## 开发环境

```bash
git clone https://github.com/wzx11223344/bayesmetrics.git
cd bayesmetrics
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -e .
```

## 代码规范

### MCMC 采样器必须实现

```python
class NewSampler:
    def __init__(self, log_post, grad_log_post=None, init_params=None, **kwargs):
        """
        Parameters
        ----------
        log_post : callable
            对数后验密度函数，签名: f(theta) -> float
        grad_log_post : callable, optional
            对数后验密度梯度，签名: f(theta) -> np.ndarray
        init_params : np.ndarray
            参数初始值, shape (dim,)
        """
        ...

    def sample(self, n_iter=1000, n_burnin=500, n_chains=1) -> np.ndarray:
        """
        Run MCMC sampling.

        Returns
        -------
        samples : np.ndarray
            shape (n_iter - n_burnin, dim) or (n_chains, n_iter - n_burnin, dim)
        """
        ...
```

### 收敛诊断规范

- R-hat 计算需严格遵循 Gelman-Rubin 链间/链内方差分解公式
- ESS 使用 Geyer 单调序列截断方法，不得使用朴素的 `N / (1 + 2*sum_acf)`
- 所有诊断函数返回包含 `value` 和 `message` 的 dict

## 测试要求

```bash
# 运行全部测试
python -m pytest tests/ -v

# 对已知后验分布进行 KS 检验
python -m pytest tests/test_samplers.py -k "test_posterior_distribution"
```

每个采样器至少通过以下验证：
1. 2D 标准正态分布的 KS 检验 (p > 0.01)
2. 自相关衰减检验 (lag-50 ACF < 0.3)
3. R-hat < 1.1 (3 chains)

## 问题反馈

- 安全漏洞: 直接联系 `3521257027@QQ.com`
- 一般问题: [GitHub Issues](https://github.com/wzx11223344/bayesmetrics/issues)
