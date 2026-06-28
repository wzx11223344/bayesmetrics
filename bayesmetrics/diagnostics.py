"""
MCMC 收敛诊断模块
================

实现了 MCMC 链收敛诊断的核心统计量:

1. Gelman-Rubin R-hat 统计量
   - 多链方差分解: 链内方差 vs 链间方差
   - R-hat < 1.1 通常认为链已收敛

2. Effective Sample Size (ESS)
   - 考虑自相关的有效样本量
   - ESS = N / (1 + 2 Σ ρ_k), ρ_k 为滞后 k 自相关

3. Autocorrelation
   - 自相关函数 (ACF)
   - 帮助判断链的混合速度

4. Summary Diagnostics
   - 一站式报告: 均值, 标准差, 分位数, R-hat, ESS

理论参考:
    - Gelman & Rubin (1992) Statistical Science
    - Vehtari et al. (2021) Bayesian Analysis (rank-normalized R-hat & ESS)
"""

import numpy as np
from scipy import stats as scipy_stats
from typing import Dict, Any


def _split_chains(samples: np.ndarray) -> np.ndarray:
    """
    将每条链对半拆分（用于 R-hat 的 split-R-hat）。

    参数
    ----
    samples : np.ndarray, shape (n_chains, n_iter, d) or (n_iter, d)
        后验样本。

    返回
    ----
    np.ndarray, shape (2*n_chains, n_iter/2, d)
    """
    if samples.ndim == 2:
        samples = samples[np.newaxis, :, :]
    n_chains, n_iter, d = samples.shape
    half = n_iter // 2
    first_half = samples[:, :half, :]
    second_half = samples[:, half:2*half, :]
    return np.vstack([first_half, second_half])


def gelman_rubin(samples: np.ndarray) -> np.ndarray:
    """
    Gelman-Rubin R-hat 收敛诊断。

    对每个参数维度分别计算 R-hat:
        R-hat = sqrt((N-1)/N + B/(N*W))

    其中 B 为链间方差, W 为链内方差。

    参数
    ----
    samples : np.ndarray, shape (n_chains, n_iter, d) or (n_iter, d)

    返回
    ----
    np.ndarray, shape (d,) — 每个参数的 R-hat 值。
    """
    if samples.ndim == 2:
        samples = samples[np.newaxis, :, :]

    n_chains, n_iter, d = samples.shape
    if n_chains < 2:
        return np.full(d, np.nan)

    # 每条链的均值
    chain_means = np.mean(samples, axis=1)  # (n_chains, d)

    # 链内方差
    chain_vars = np.var(samples, axis=1, ddof=1)  # (n_chains, d)
    W = np.mean(chain_vars, axis=0)  # (d,)

    # 链间方差
    grand_mean = np.mean(chain_means, axis=0)  # (d,)
    B = n_iter * np.var(chain_means, axis=0, ddof=1)  # (d,)

    # 边际后验方差估计
    var_hat = (n_iter - 1) / n_iter * W + B / n_iter

    r_hat = np.sqrt(var_hat / W)
    r_hat = np.nan_to_num(r_hat, nan=np.inf, posinf=np.inf)
    return r_hat


def autocorrelation(samples: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """
    计算每个参数维度的自相关函数 (ACF)。

    参数
    ----
    samples : np.ndarray, shape (n_iter, d)
    max_lag : int

    返回
    ----
    np.ndarray, shape (max_lag + 1, d)
    """
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    n, d = samples.shape
    samples_centered = samples - np.mean(samples, axis=0)

    acf = np.zeros((max_lag + 1, d))
    acf[0] = 1.0

    var = np.var(samples, axis=0, ddof=1)
    for lag in range(1, max_lag + 1):
        acf[lag] = np.sum(samples_centered[:-lag] * samples_centered[lag:], axis=0) / (n * var)

    return acf


def effective_sample_size(samples: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """
    计算有效样本量 (ESS)。

    ESS = N / (1 + 2 Σ_{k=1}^{max_lag} (1 - k/N) ρ_k)

    使用 Geyer (1992) 的初始单调序列估计器自动截断。

    参数
    ----
    samples : np.ndarray, shape (n_iter, d)
    max_lag : int

    返回
    ----
    np.ndarray, shape (d,)
    """
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    n, d = samples.shape

    # 计算自相关
    acf = autocorrelation(samples, min(max_lag, n // 4))

    # Geyer 的初始单调序列: 当连续 pair 的和变负时截断
    ess = np.zeros(d)
    for j in range(d):
        tau = 1.0
        for k in range(1, len(acf) - 1, 2):
            pair_sum = acf[k, j] + acf[k + 1, j]
            if pair_sum < 0:
                break
            tau += 2 * pair_sum
        ess[j] = n / max(tau, 1.0)

    return ess


def summary_diagnostics(samples: np.ndarray,
                        var_names: list = None,
                        alpha: float = 0.05) -> Dict[str, Any]:
    """
    生成 MCMC 收敛诊断全面报告。

    参数
    ----
    samples : np.ndarray, shape (n_chains, n_iter, d) or (n_iter, d)
    var_names : list of str, optional
    alpha : float, 置信水平

    返回
    ----
    dict with:
        - 'mean': 后验均值
        - 'sd': 后验标准差
        - 'hdi_lower', 'hdi_upper': 最高密度区间 (近似)
        - 'r_hat': Gelman-Rubin 统计量
        - 'ess': 有效样本量
        - 'ess_per_sec': 每迭代的有效样本率 (链混合效率指标)
    """
    if samples.ndim == 2:
        samples = samples[np.newaxis, :, :]

    n_chains, n_iter, d = samples.shape

    # 合并所有链
    flat_samples = samples.reshape(-1, d)

    # 后验统计
    mean = np.mean(flat_samples, axis=0)
    sd = np.std(flat_samples, axis=0, ddof=1)
    q_lower = np.percentile(flat_samples, 100 * alpha / 2, axis=0)
    q_upper = np.percentile(flat_samples, 100 * (1 - alpha / 2), axis=0)

    # 收敛诊断
    r_hat = gelman_rubin(samples)

    # ESS (对每条链分别计算, 取最小值)
    ess_values = np.zeros((n_chains, d))
    for c in range(n_chains):
        ess_values[c] = effective_sample_size(samples[c])

    ess = np.min(ess_values, axis=0)  # 保守估计

    if var_names is None:
        var_names = [f"theta[{i}]" for i in range(d)]

    table = []
    for i in range(d):
        rhat_status = "✓" if r_hat[i] < 1.1 else ("⚠" if r_hat[i] < 1.2 else "✗")
        row = {
            "parameter": var_names[i],
            "mean": float(mean[i]),
            "sd": float(sd[i]),
            f"hdi_{alpha:.0%}_lower": float(q_lower[i]),
            f"hdi_{alpha:.0%}_upper": float(q_upper[i]),
            "r_hat": float(r_hat[i]),
            "ess": float(ess[i]),
            "rhat_status": rhat_status,
        }
        table.append(row)

    return {
        "table": table,
        "n_chains": n_chains,
        "n_iter": n_iter,
        "total_draws": n_chains * n_iter,
    }


def print_summary(samples: np.ndarray, var_names: list = None):
    """以表格形式打印收敛诊断报告。"""
    result = summary_diagnostics(samples, var_names)

    print("=" * 72)
    print("MCMC 收敛诊断报告")
    print("=" * 72)
    print(f"链数: {result['n_chains']}  迭代/链: {result['n_iter']}"
          f"  总样本: {result['total_draws']}")
    print("-" * 72)
    header = (f"{'Parameter':>16s}  {'Mean':>10s}  {'SD':>10s}  "
              f"{'R-hat':>8s}  {'ESS':>8s}  {'Status':>6s}")
    print(header)
    print("-" * 72)

    for row in result["table"]:
        print(f"{row['parameter']:>16s}  {row['mean']:>10.4f}  "
              f"{row['sd']:>10.4f}  {row['r_hat']:>8.3f}  "
              f"{row['ess']:>8.1f}  {row['rhat_status']:>6s}")

    print("=" * 72)
    print("R-hat < 1.1: 链已收敛  |  ESS > 400: 有效样本充足")
