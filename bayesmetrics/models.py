"""
贝叶斯模型模块
==============

实现了常用计量经济学模型的贝叶斯版本:

1. Bayesian Linear Regression
   - 共轭先验 (Normal-InverseGamma)
   - Gibbs 采样精确后验
   - 支持先验超参数设定

2. Bayesian Logit / Probit
   - 二元选择模型的贝叶斯估计
   - MH 或 HMC/NUTS 采样
   - 边际效应后验分布

3. Bayesian VAR
   - 向量自回归模型的贝叶斯估计
   - Minnesota 先验 (Litterman, 1986)
   - 脉冲响应函数 (IRF) 后验分布

理论参考:
    - Koop (2003) Bayesian Econometrics
    - Litterman (1986) Forecasting with Bayesian VARs
    - Gelman et al. (2013) Bayesian Data Analysis
"""

import numpy as np
from scipy import stats as scipy_stats
from numpy.linalg import inv, det
from scipy.special import expit as logistic
from typing import Optional, Tuple, Dict, Any
from .samplers import MetropolisHastings, GibbsSampler
from .priors import NormalPrior, InverseGammaPrior, MVNPrior


# ============================================================
# 1. Bayesian Linear Regression
# ============================================================

class BayesianLinearRegression:
    """
    贝叶斯线性回归: y = Xβ + ε, ε ~ N(0, σ²)

    使用 Normal-InverseGamma 共轭先验:
        β | σ² ~ N(β₀, σ² * V₀)
        σ² ~ IG(a₀, b₀)

    后验分布有解析形式，通过 Gibbs 采样获得后验样本。

    使用
    ----
    blr = BayesianLinearRegression()
    blr.fit(X, y, n_iter=5000, n_burnin=1000)
    print(blr.summary())
    """

    def __init__(self):
        self.beta_samples = None
        self.sigma2_samples = None
        self.var_names = None
        self.n_obs = 0
        self.k = 0

    def fit(self, X: np.ndarray, y: np.ndarray,
            beta_prior_mean: Optional[np.ndarray] = None,
            beta_prior_precision: float = 0.01,
            sigma2_a0: float = 2.0,
            sigma2_b0: float = 1.0,
            n_iter: int = 5000, n_burnin: int = 1000,
            var_names: Optional[list] = None,
            progress: bool = True) -> "BayesianLinearRegression":
        """
        y = Xβ + ε 的贝叶斯估计。

        共轭先验:
            β | σ² ~ N(β₀, σ² * P₀⁻¹),  P₀ = beta_prior_precision * I
            σ² ~ IG(a₀, b₀)

        参数
        ----
        X : np.ndarray, shape (n, k)
        y : np.ndarray, shape (n,)
        beta_prior_mean : np.ndarray or None, β₀
        beta_prior_precision : float, 先验精度 (小值 = 弱先验)
        sigma2_a0 : float, IG 形状
        sigma2_b0 : float, IG 尺度
        n_iter, n_burnin : int

        返回
        ----
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).flatten()
        n, k = X.shape

        self.n_obs = n
        self.k = k

        if var_names is None:
            self.var_names = ["const"] + [f"x{i}" for i in range(k - 1)]
        else:
            self.var_names = var_names

        # 添加截距项
        if not np.allclose(X[:, 0], 1.0):
            X = np.column_stack([np.ones(n), X])
            k += 1
            self.var_names = ["const"] + (["x" + str(i) for i in range(k - 1)] if var_names is None else var_names)

        # 先验参数
        if beta_prior_mean is None:
            beta_prior_mean = np.zeros(k)
        beta_prior_mean = np.asarray(beta_prior_mean, dtype=float)

        P0 = beta_prior_precision * np.eye(k)

        # X'X 和 X'y (后验充分统计量)
        XtX = X.T @ X
        Xty = X.T @ y

        # 后验精度
        P_n = XtX + P0

        # Gibbs 采样
        beta = np.zeros(k)
        sigma2 = 1.0
        beta_samples = []
        sigma2_samples = []

        for i in range(n_burnin + n_iter):
            # 1. β | σ² ~ N(β_n, σ² * P_n⁻¹)
            beta_n = inv(P_n) @ (Xty + P0 @ beta_prior_mean)
            beta_cov = sigma2 * inv(P_n)
            beta = np.random.multivariate_normal(beta_n, beta_cov)

            # 2. σ² | β ~ IG(a_n, b_n)
            residuals = y - X @ beta
            a_n = sigma2_a0 + n / 2
            b_n = sigma2_b0 + 0.5 * np.sum(residuals ** 2)
            sigma2 = 1.0 / np.random.gamma(a_n, 1.0 / b_n)

            if i >= n_burnin:
                beta_samples.append(beta.copy())
                sigma2_samples.append(sigma2)

            if progress and (i + 1) % ((n_burnin + n_iter) // 5) == 0:
                print(f"  BLR Gibbs: {100 * (i + 1) / (n_burnin + n_iter):.0f}%")

        self.beta_samples = np.array(beta_samples)
        self.sigma2_samples = np.array(sigma2_samples)
        self.XtX = XtX
        self.Xty = Xty

        return self

    def summary(self, alpha: float = 0.05) -> str:
        """生成贝叶斯回归结果汇总。"""
        if self.beta_samples is None:
            raise RuntimeError("请先调用 fit() 方法")

        lines = []
        lines.append("=" * 72)
        lines.append("贝叶斯线性回归 (共轭先验 + Gibbs 采样)")
        lines.append("=" * 72)
        lines.append(f"观测数: {self.n_obs:>6d}         变量数: {self.k:>6d}")
        lines.append(f"采样数: {len(self.beta_samples):>6d}")
        lines.append("-" * 72)

        header = (f"{'Variable':>16s}  {'Post.Mean':>10s}  {'Post.SD':>10s}  "
                  f"{'HDI Low':>10s}  {'HDI High':>10s}")
        lines.append(header)
        lines.append("-" * 72)

        for i in range(self.k):
            name = self.var_names[i] if i < len(self.var_names) else f"x{i}"
            beta_i = self.beta_samples[:, i]
            mean = float(np.mean(beta_i))
            sd = float(np.std(beta_i))
            hdi_low = float(np.percentile(beta_i, 100 * alpha / 2))
            hdi_high = float(np.percentile(beta_i, 100 * (1 - alpha / 2)))

            # 显著性星号 (基于 HDI 是否跨越 0)
            sig = "***" if hdi_low * hdi_high > 0 else "   "
            lines.append(f"{name:>16s}  {mean:>10.4f}  {sd:>10.4f}  "
                         f"{hdi_low:>10.4f}  {hdi_high:>10.4f} {sig}")

        lines.append("-" * 72)
        sigma2_i = self.sigma2_samples
        lines.append(f"  σ² (误差方差): mean={np.mean(sigma2_i):.4f}, "
                     f"sd={np.std(sigma2_i):.4f}")
        lines.append("*** HDI 不跨越 0 (相当于频率主义 p<0.01)")
        lines.append("=" * 72)
        return "\n".join(lines)

    def predict(self, X_new: np.ndarray, return_ci: bool = True,
                alpha: float = 0.05) -> Tuple[np.ndarray, ...]:
        """后验预测分布。"""
        X_new = np.asarray(X_new, dtype=float)
        if X_new.shape[1] == self.k - 1:
            X_new = np.column_stack([np.ones(len(X_new)), X_new])

        y_pred_samples = np.array([X_new @ beta + np.sqrt(s2) * np.random.randn(len(X_new))
                                    for beta, s2 in zip(self.beta_samples[-1000:],
                                                        self.sigma2_samples[-1000:])])

        y_mean = np.mean(y_pred_samples, axis=0)
        if not return_ci:
            return y_mean

        y_lower = np.percentile(y_pred_samples, 100 * alpha / 2, axis=0)
        y_upper = np.percentile(y_pred_samples, 100 * (1 - alpha / 2), axis=0)
        return y_mean, y_lower, y_upper


# ============================================================
# 2. Bayesian Logit / Probit
# ============================================================

class BayesianLogit:
    """
    贝叶斯 Logit 模型。

    使用 MH 采样器 (或可以将 log_target 传入 HMC/NUTS) 估计:
        P(y=1 | x) = logistic(xβ)

    先验: β ~ N(0, τ² I), 默认 τ = 10 (弱信息)

    使用
    ----
    blogit = BayesianLogit()
    result = blogit.fit(X, y, n_iter=10000, n_burnin=2000)
    print(result.summary())
    """

    def __init__(self):
        self.beta_samples = None
        self.var_names = None
        self.mfx_samples = None
        self.n_obs = 0
        self.k = 0

    def _add_intercept(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if not np.allclose(X[:, 0], 1.0):
            X = np.column_stack([np.ones(len(X)), X])
        return X

    def fit(self, X: np.ndarray, y: np.ndarray,
            prior_sd: float = 10.0,
            n_iter: int = 10000, n_burnin: int = 2000,
            n_chains: int = 1,
            var_names: Optional[list] = None,
            progress: bool = True) -> "BayesianLogit":
        """
        贝叶斯 Logit 估计 (使用 MH 采样器)。

        对数后验:
            log p(β|y,X) ∝ Σ [y_i log(p_i) + (1-y_i) log(1-p_i)] - 1/(2τ²) β'β
        """
        X = self._add_intercept(X)
        y = np.asarray(y, dtype=float).flatten()
        n, k = X.shape
        self.n_obs = n
        self.k = k

        if var_names is None:
            self.var_names = ["const"] + [f"x{i}" for i in range(k - 1)]
        else:
            self.var_names = var_names

        tau2 = prior_sd ** 2

        def log_posterior(beta):
            eta = X @ beta
            # 对数似然 (避免溢出)
            eta_clipped = np.clip(eta, -100, 100)
            log_lik = np.sum(y * eta_clipped - np.log(1 + np.exp(eta_clipped)))
            # 对数先验 N(0, τ²)
            log_prior = -0.5 * np.sum(beta ** 2) / tau2
            return log_lik + log_prior

        initial_beta = np.zeros(k)

        if progress:
            print("运行 MH 采样器...")

        mh = MetropolisHastings(log_posterior, initial_beta, step_size=0.2)
        chains = mh.sample(n_iter=n_iter, n_burnin=n_burnin,
                           n_chains=n_chains, adaptive=True,
                           adapt_interval=500, progress=progress)

        self.beta_samples = chains[0]
        if progress:
            acc = mh.acceptance_rate
            print(f"  MH acceptance rate: {acc:.3f}")

        # 计算边际效应 (AME)
        self._compute_ame(X)

        return self

    def _compute_ame(self, X):
        """计算平均边际效应后验分布。"""
        n_samples = len(self.beta_samples)
        self.mfx_samples = np.zeros((n_samples, self.k))

        for s in range(min(n_samples, 1000)):
            beta = self.beta_samples[s]
            eta = X @ beta
            mu = logistic(eta)
            mu_bar = np.mean(mu)
            scale = mu_bar * (1 - mu_bar)
            self.mfx_samples[s] = beta * scale

    def summary(self, alpha: float = 0.05) -> str:
        if self.beta_samples is None:
            raise RuntimeError("请先调用 fit() 方法")

        lines = [
            "=" * 72,
            "贝叶斯 Logit 模型 (MH 采样)",
            "=" * 72,
            f"观测数: {self.n_obs:>6d}         变量数: {self.k:>6d}",
            f"采样数: {len(self.beta_samples):>6d}",
            "-" * 72,
            f"{'Variable':>16s}  {'Post.Mean':>10s}  {'Post.SD':>10s}  {'HDI Low':>10s}  {'HDI High':>10s}  {'AME':>10s}",
            "-" * 72,
        ]

        for i in range(self.k):
            name = self.var_names[i] if i < len(self.var_names) else f"x{i}"
            beta_i = self.beta_samples[:, i]
            mfx_i = self.mfx_samples[:, i]
            mean = float(np.mean(beta_i))
            sd = float(np.std(beta_i))
            ame = float(np.mean(mfx_i))
            hdi_low = float(np.percentile(beta_i, 100 * alpha / 2))
            hdi_high = float(np.percentile(beta_i, 100 * (1 - alpha / 2)))
            sig = "***" if hdi_low * hdi_high > 0 else "   "
            lines.append(f"{name:>16s}  {mean:>10.4f}  {sd:>10.4f}  "
                         f"{hdi_low:>10.4f}  {hdi_high:>10.4f}  {ame:>10.4f} {sig}")

        lines.append("=" * 72)
        lines.append("AME = 平均边际效应  |  *** HDI 不跨越 0")
        return "\n".join(lines)


class BayesianProbit(BayesianLogit):
    """
    贝叶斯 Probit 模型。

    与 Logit 类似，但使用概率链接函数:
        P(y=1 | x) = Φ(xβ)

    通过 MH 采样器估计。
    """

    def fit(self, X: np.ndarray, y: np.ndarray,
            prior_sd: float = 10.0,
            n_iter: int = 10000, n_burnin: int = 2000,
            n_chains: int = 1,
            var_names: Optional[list] = None,
            progress: bool = True) -> "BayesianProbit":
        from scipy.stats import norm

        X = self._add_intercept(X)
        y = np.asarray(y, dtype=float).flatten()
        n, k = X.shape
        self.n_obs = n
        self.k = k

        if var_names is None:
            self.var_names = ["const"] + [f"x{i}" for i in range(k - 1)]
        else:
            self.var_names = var_names

        tau2 = prior_sd ** 2

        eps = 1e-15

        def log_posterior(beta):
            eta = X @ beta
            p = np.clip(norm.cdf(eta), eps, 1 - eps)
            log_lik = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
            log_prior = -0.5 * np.sum(beta ** 2) / tau2
            return log_lik + log_prior

        mh = MetropolisHastings(log_posterior, np.zeros(k), step_size=0.2)
        chains = mh.sample(n_iter=n_iter, n_burnin=n_burnin,
                           n_chains=n_chains, adaptive=True,
                           adapt_interval=500, progress=progress)
        self.beta_samples = chains[0]
        self._compute_ame(X)
        return self


# ============================================================
# 3. Bayesian VAR
# ============================================================

class BayesianVAR:
    """
    贝叶斯向量自回归模型 (BVAR)。

    使用 Minnesota (Litterman) 先验, 通过 Gibbs 采样估计。

    模型: y_t = c + A₁ y_{t-1} + ... + A_p y_{t-p} + ε_t,  ε_t ~ N(0, Σ)

    Minnesota 先验假设:
        - 每个变量服从 AR(1) 过程
        - 较远滞后项的影响递减
        - 交叉变量滞后项系数趋近于 0

    使用
    ----
    bvar = BayesianVAR(lags=2)
    bvar.fit(data)  # data shape (T, M)
    irf_samples = bvar.impulse_response(horizon=20)
    """

    def __init__(self, lags: int = 2):
        """
        参数
        ----
        lags : int, 滞后阶数 p
        """
        self.p = lags
        self.coef_samples = None  # (A₁, ..., A_p, c) 的后验样本
        self.sigma_samples = None
        self.M = 0  # 变量数
        self.T_eff = 0  # 有效样本量

    def fit(self, data: np.ndarray,
            lambda1: float = 0.2, lambda2: float = 0.5,
            lambda3: float = 1.0, lambda4: float = 100.0,
            n_iter: int = 5000, n_burnin: int = 1000,
            progress: bool = True) -> "BayesianVAR":
        """
        使用 Minnesota 先验 + Gibbs 采样估计 BVAR。

        参数
        ----
        data : np.ndarray, shape (T, M)
        lambda1 : float, 整体紧缩超参数 (越小, 先验越紧)
        lambda2 : float, 交叉变量权重 (标准设定 0.5)
        lambda3 : float, 滞后衰减率 (1 = 线性衰减, 2 = 调和衰减)
        lambda4 : float, 常数项紧缩 (大值 = 弱先验)
        """
        data = np.asarray(data, dtype=float)
        T, M = data.shape
        self.M = M
        self.T_eff = T - self.p

        # 构建 Y 和 X (VAR(p) 的 SUR 形式)
        Y = data[self.p:]  # (T-p, M)
        X = np.zeros((self.T_eff, M * self.p + 1))
        for lag in range(1, self.p + 1):
            X[:, (lag - 1) * M:lag * M] = data[self.p - lag: T - lag]
        X[:, -1] = 1.0  # 截距项

        # ---- Minnesota 先验超参数 ----
        # 对每个变量计算 AR(1) 系数方差的先验尺度
        sigma_sq = np.var(data, axis=0, ddof=1)  # 各变量的无条件方差

        # 先验均值: 全为零 (平稳性先验，适用于差分后数据)
        # 如需随机游走先验，设为: prior_mean[i] = 1.0 for i in range(M)
        prior_mean = np.zeros(M * self.p + 1)

        # 先验精度矩阵
        prior_precision = np.zeros((M * self.p + 1, M * self.p + 1))

        for i in range(M):
            for lag in range(1, self.p + 1):
                for j in range(M):
                    idx_i = (lag - 1) * M + i
                    idx_j = (lag - 1) * M + j
                    if i == j:
                        # 自回归系数: 先验方差 = (λ₁ / lag^{λ₃})²
                        prior_precision[idx_i, idx_i] = (lag ** lambda3 / lambda1) ** 2
                    else:
                        # 交叉变量系数: 先验方差 = (λ₁ λ₂ σ_i / lag^{λ₃} σ_j)²
                        ratio = (lambda1 * lambda2 * np.sqrt(sigma_sq[i])
                                 / (lag ** lambda3 * np.sqrt(sigma_sq[j]))) if sigma_sq[j] > 0 else lambda1 * lambda2
                        prior_precision[idx_i, idx_j] = 1.0 / max(ratio ** 2, 1e-10)

        # 截距项先验 (弱)
        const_idx = M * self.p
        prior_precision[const_idx, const_idx] = 1.0 / (lambda4 * np.mean(sigma_sq))

        # ---- Gibbs 采样 ----
        # Σ 的先验: Inverse Wishart, 自由度 = M + 2, 尺度 = I
        Sigma = np.eye(M)
        nu0 = M + 2
        S0 = np.eye(M)

        coef_samples = []
        sigma_samples = []

        for it in range(n_burnin + n_iter):
            # 1. vec(B) | Σ ~ N(vec(B̂), Σ ⊗ (X'X + P₀)⁻¹)
            XtX = X.T @ X
            P_post = XtX + prior_precision
            P_post_inv = inv(P_post)

            # 后验均值
            vec_mu = P_post_inv @ (X.T @ Y + prior_precision @ prior_mean.reshape(-1, 1))
            vec_mu = vec_mu.flatten()  # (M*(Mp+1),)
            B_hat = vec_mu.reshape(M * self.p + 1, M).T  # (M, Mp+1)

            # 从 vec(B) 的多元正态后验采样
            # vec(B) | Sigma, Y ~ N(vec_mu, Sigma ⊗ P_post_inv)
            # 先采样 z ~ N(0, I), 然后 vec(B) = vec_mu + (L_Sigma ⊗ L_P) z
            try:
                L_P = np.linalg.cholesky(P_post_inv)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(P_post_inv)
                eigvals = np.maximum(eigvals, 1e-10)
                L_P = eigvecs @ np.diag(np.sqrt(eigvals))

            try:
                L_Sigma = np.linalg.cholesky(Sigma)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(Sigma)
                eigvals = np.maximum(eigvals, 1e-10)
                L_Sigma = eigvecs @ np.diag(np.sqrt(eigvals))

            # Kronecker 采样: reshape(L_P @ randn(P) @ L_Sigma.T)
            Z = np.random.randn(M * self.p + 1, M)
            B_draw_raw = L_P @ Z @ L_Sigma.T + vec_mu.reshape(M * self.p + 1, M)
            B_draw = B_draw_raw.T  # (M, Mp+1)

            # 2. Σ | B ~ IW(nu_n, S_n)
            residuals = Y - X @ B_draw.T
            S_n = S0 + residuals.T @ residuals
            nu_n = nu0 + self.T_eff

            # 从 IW 采样
            try:
                Sigma = inv(scipy_stats.wishart.rvs(nu_n, inv(S_n)))
            except:
                Sigma = np.eye(M)

            if it >= n_burnin:
                coef_samples.append(B_draw.copy())
                sigma_samples.append(Sigma.copy())

            if progress and (it + 1) % ((n_burnin + n_iter) // 5) == 0:
                print(f"  BVAR Gibbs: {100 * (it + 1) / (n_burnin + n_iter):.0f}%")

        self.coef_samples = np.array(coef_samples)
        self.sigma_samples = np.array(sigma_samples)

        return self

    def impulse_response(self, shock_var: int = 0,
                         horizon: int = 20,
                         shock_sign: bool = True) -> np.ndarray:
        """
        计算脉冲响应函数 (IRF) 的后验分布。

        对后验中的每个系数样本, 计算 Cholesky 分解下的 IRF。

        参数
        ----
        shock_var : int, 冲击变量的索引
        horizon : int, 脉冲响应期数
        shock_sign : bool, 正向冲击 (True) 或负向

        返回
        ----
        np.ndarray, shape (n_samples, M, horizon)
        """
        if self.coef_samples is None:
            raise RuntimeError("请先调用 fit() 方法")

        M = self.M
        p = self.p
        n_samples = len(self.coef_samples)

        irf_samples = np.zeros((n_samples, M, horizon))

        for s in range(n_samples):
            B = self.coef_samples[s]
            Sigma = self.sigma_samples[s]

            # Cholesky 分解获取冲击
            chol_Sigma = np.linalg.cholesky(Sigma)

            # 冲击向量 (1 个标准差冲击)
            shock = chol_Sigma[:, shock_var] * (1.0 if shock_sign else -1.0)

            # 构建 companion 矩阵
            companion = np.zeros((M * p, M * p))
            companion[:M, :M * p] = B[:, :M * p]  # A₁ ... A_p
            if p > 1:
                companion[M:, :M * (p - 1)] = np.eye(M * (p - 1))

            # 脉冲响应迭代
            state = np.zeros(M * p)
            state[:M] = shock

            for h in range(horizon):
                irf_samples[s, :, h] = state[:M]
                state = companion @ state

        return irf_samples

    def summary(self, alpha: float = 0.05) -> str:
        """BVAR 估计概要。"""
        if self.coef_samples is None:
            raise RuntimeError("请先调用 fit() 方法")

        n_samples = len(self.coef_samples)
        lines = [
            "=" * 72,
            f"贝叶斯 VAR({self.p}) (Minnesota 先验 + Gibbs 采样)",
            "=" * 72,
            f"变量数: {self.M}  有效样本: {self.T_eff}  后验采样数: {n_samples}",
            "=" * 72,
        ]

        # 报告第一滞后矩阵 (A₁) 的后验均值
        lines.append("A₁ (第一滞后矩阵) 后验均值:")
        lines.append("-" * 36)
        header = "       " + " ".join(f"{'Eq'+str(i+1):>8s}" for i in range(self.M))
        lines.append(header)
        lines.append("-" * 36)

        A1 = np.mean(self.coef_samples[:, :, :self.M], axis=0)  # (M, M)
        for i in range(self.M):
            row = f"Var{i+1}  " + " ".join(f"{A1[i, j]:>8.4f}" for j in range(self.M))
            lines.append(row)

        lines.append("=" * 72)
        return "\n".join(lines)
