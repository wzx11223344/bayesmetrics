"""
先验分布模块
===========

实现了贝叶斯推断中常用的先验分布及其对数概率密度函数。

支持:
    - NormalPrior: 一元正态先验
    - InverseGammaPrior: 逆伽马先验 (方差的共轭先验)
    - WishartPrior: Wishart 先验 (协方差矩阵的共轭先验)
    - MVNPrior: 多元正态先验
    - log_posterior: 通用对数后验计算
"""

import numpy as np
from scipy.special import gammaln
from numpy.linalg import inv, det, slogdet


class NormalPrior:
    """一元正态先验 N(mu, sigma^2)。"""

    def __init__(self, mu: float = 0.0, sigma: float = 10.0):
        """
        参数
        ----
        mu : float, 均值
        sigma : float, 标准差 (> 0)
        """
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.mu = mu
        self.sigma = sigma
        self.sigma2 = sigma ** 2

    def logpdf(self, x):
        """对数概率密度。"""
        if np.ndim(x) == 0:
            return -0.5 * np.log(2 * np.pi * self.sigma2) - 0.5 * (x - self.mu) ** 2 / self.sigma2
        return -0.5 * np.log(2 * np.pi * self.sigma2) - 0.5 * np.sum((x - self.mu) ** 2) / self.sigma2

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def sample(self, size=1):
        return np.random.normal(self.mu, self.sigma, size=size)


class InverseGammaPrior:
    """逆伽马先验 IG(alpha, beta)。方差 sigma^2 的常用共轭先验。"""

    def __init__(self, alpha: float = 2.0, beta: float = 1.0):
        """
        参数
        ----
        alpha : float, 形状参数 (> 0)
        beta  : float, 尺度参数 (> 0)
        """
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive")
        self.alpha = alpha
        self.beta = beta

    def logpdf(self, x):
        """对数概率密度。x > 0。"""
        x = np.asarray(x, dtype=float)
        if np.any(x <= 0):
            return -np.inf
        return (self.alpha * np.log(self.beta) - gammaln(self.alpha)
                - (self.alpha + 1) * np.log(x) - self.beta / x)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def sample(self, size=1):
        return 1.0 / np.random.gamma(self.alpha, 1.0 / self.beta, size=size)


class MVNPrior:
    """多元正态先验 N(mu, Sigma)。回归系数向量的常用先验。"""

    def __init__(self, mu=None, Sigma=None, k: int = 1):
        """
        参数
        ----
        mu : np.ndarray, shape (k,) or None, 均值向量
        Sigma : np.ndarray, shape (k,k) or None, 协方差矩阵
        k : int, 维度（当 mu 和 Sigma 为 None 时使用）
        """
        if mu is not None:
            self.k = len(mu)
            self.mu = np.asarray(mu, dtype=float)
        else:
            self.k = k
            self.mu = np.zeros(k)

        if Sigma is not None:
            self.Sigma = np.asarray(Sigma, dtype=float)
        else:
            # 默认使用单位矩阵 × 100 (弱信息先验)
            self.Sigma = 100.0 * np.eye(self.k)

        self.Sigma_inv = inv(self.Sigma)
        _, self.logdet_Sigma = slogdet(self.Sigma)

    def logpdf(self, x):
        """对数概率密度。"""
        x = np.asarray(x, dtype=float)
        diff = x - self.mu
        return -0.5 * (self.k * np.log(2 * np.pi) + self.logdet_Sigma
                       + diff @ self.Sigma_inv @ diff)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def sample(self, size=1):
        return np.random.multivariate_normal(self.mu, self.Sigma, size=size)


class WishartPrior:
    """Wishart 先验 W(nu, V)。协方差矩阵的共轭先验。"""

    def __init__(self, nu: int, V=None, k: int = 1):
        """
        参数
        ----
        nu : int, 自由度 (>= k)
        V : np.ndarray, shape (k,k) or None, 尺度矩阵
        k : int, 维度
        """
        self.nu = nu
        if V is not None:
            self.V = np.asarray(V, dtype=float)
            self.k = self.V.shape[0]
        else:
            self.k = k
            self.V = np.eye(k)
        self.V_inv = inv(self.V)
        _, self.logdet_V = slogdet(self.V)

    def logpdf(self, X):
        """对数概率密度。X 为 k×k 正定矩阵。"""
        X = np.asarray(X, dtype=float)
        k = X.shape[0]
        if np.any(np.linalg.eigvalsh(X) <= 0):
            return -np.inf

        _, logdet_X = slogdet(X)
        trace_term = np.trace(self.V_inv @ X)

        result = (-0.5 * self.nu * k * np.log(2)
                  - 0.5 * k * (k - 1) / 4 * np.log(np.pi)
                  - np.sum(gammaln(0.5 * (self.nu - np.arange(k)))))
        result += 0.5 * (self.nu - k - 1) * logdet_X
        result += 0.5 * self.nu * self.logdet_V
        result -= 0.5 * trace_term
        return result

    def pdf(self, X):
        return np.exp(self.logpdf(X))


def log_posterior(log_likelihood_fn, prior_list, theta, *args):
    """
    计算非标准化对数后验: log p(theta|y) ∝ log p(y|theta) + Σ log p(theta_j)

    参数
    ----
    log_likelihood_fn : callable
        对数似然函数 log p(y|theta, *args)。
    prior_list : list of Prior objects
        先验分布列表，每个先验对应 theta 的一个分量。
    theta : np.ndarray
        参数向量。
    *args : 传递给 log_likelihood_fn 的额外参数。

    返回
    ----
    float
    """
    lp = sum(p.logpdf(t) for p, t in zip(prior_list, theta))
    return log_likelihood_fn(theta, *args) + lp
