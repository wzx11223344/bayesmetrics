"""
MCMC 采样器模块
==============

从零实现了四种马尔可夫链蒙特卡洛 (MCMC) 采样算法:

1. Metropolis-Hastings (MH)
   - 经典 MH 算法，支持自适应 proposal 调优
   - 多链并行采样

2. Gibbs Sampler
   - 全条件分布逐分量采样
   - 适用于共轭模型

3. Hamiltonian Monte Carlo (HMC)
   - 利用梯度信息的 Hamiltonian 动力学采样
   - Leapfrog 积分器 + 动量翻新
   - 比 MH 效率高 10-100 倍

4. No-U-Turn Sampler (NUTS)
   - Hoffman & Gelman (2014) 的自适应 HMC
   - 自动调整 Leapfrog 步数
   - Stan/PyMC 使用的核心算法

理论参考:
    - Neal (2011) MCMC using Hamiltonian dynamics
    - Hoffman & Gelman (2014) The No-U-Turn Sampler
    - Betancourt (2017) A Conceptual Introduction to HMC
"""

import numpy as np
from scipy import stats as scipy_stats
from typing import Callable, Optional, Tuple


# ============================================================
# 1. Metropolis-Hastings
# ============================================================

class MetropolisHastings:
    """
    Metropolis-Hastings 采样器。

    支持自适应 proposal 协方差调优 (Haario et al., 2001)。

    使用
    ----
    mh = MetropolisHastings(log_target, initial_theta)
    samples = mh.sample(n_iter=10000, n_burnin=2000)
    """

    def __init__(self, log_target: Callable, initial_theta: np.ndarray,
                 proposal_cov: Optional[np.ndarray] = None,
                 step_size: float = 0.1):
        """
        参数
        ----
        log_target : callable
            非标准化的对数目标密度函数: f(theta) -> float。
        initial_theta : np.ndarray, shape (d,)
            初始参数值。
        proposal_cov : np.ndarray, shape (d,d) or None
            proposal 协方差矩阵。None 时自动使用 step_size * I。
        step_size : float
            proposal 缩放因子。
        """
        self.log_target = log_target
        self.theta = np.asarray(initial_theta, dtype=float).copy()
        self.d = len(self.theta)

        if proposal_cov is None:
            self.proposal_cov = (step_size ** 2) * np.eye(self.d)
        else:
            self.proposal_cov = np.asarray(proposal_cov, dtype=float)
        self.step_size = step_size

        self._current_logp = self.log_target(self.theta)
        self._n_accepted = 0

    def sample(self, n_iter: int = 10000, n_burnin: int = 2000,
               thin: int = 1, n_chains: int = 1,
               adaptive: bool = True, adapt_interval: int = 100,
               progress: bool = True) -> np.ndarray:
        """
        运行 MH 采样。

        参数
        ----
        n_iter : int, 总迭代次数
        n_burnin : int, warm-up 迭代次数
        thin : int, 稀释间隔
        n_chains : int, 并行链数
        adaptive : bool, 是否使用自适应 proposal
        adapt_interval : int, 自适应调优间隔
        progress : bool, 是否打印进度

        返回
        ----
        np.ndarray, shape (n_chains, (n_iter - n_burnin) / thin, d)
        """
        chains = []
        accept_rates = []

        for chain_id in range(n_chains):
            theta = self.theta.copy() + 0.01 * np.random.randn(self.d)
            logp = self.log_target(theta)
            chain_samples = []
            n_accept = 0
            total = n_burnin + n_iter

            # 自适应调优的累积统计量
            s_mu = np.zeros(self.d)
            s_cov = np.zeros((self.d, self.d))
            cov = self.proposal_cov.copy()

            for i in range(total):
                # proposal: theta* ~ N(theta, cov)
                proposal = np.random.multivariate_normal(theta, cov)
                logp_proposal = self.log_target(proposal)

                # Metropolis 接受率
                log_alpha = logp_proposal - logp
                if log_alpha > 0 or np.log(np.random.rand()) < log_alpha:
                    theta = proposal
                    logp = logp_proposal
                    n_accept += 1

                # 存储 (burn-in 之后)
                if i >= n_burnin and (i - n_burnin) % thin == 0:
                    chain_samples.append(theta.copy())

                # 自适应调优: 每 adapt_interval 步更新 proposal 协方差
                if adaptive and i >= n_burnin and len(chain_samples) >= adapt_interval and i % adapt_interval == 0:
                    recent = np.array(chain_samples[-adapt_interval:])
                    if recent.shape[0] >= self.d:
                        sigma_hat = np.cov(recent.T)
                        if sigma_hat.ndim == 0:
                            sigma_hat = np.array([[sigma_hat]])
                        # 确保正定性
                        sigma_hat += 0.01 * np.eye(self.d)
                        cov = 2.38 ** 2 / self.d * sigma_hat

            chains.append(np.array(chain_samples))
            accept_rates.append(n_accept / total)

            if progress and n_chains > 1:
                print(f"  Chain {chain_id + 1}: acceptance rate = {accept_rates[-1]:.3f}")

        self.acceptance_rate = np.mean(accept_rates)
        if progress:
            print(f"  Average acceptance rate: {self.acceptance_rate:.3f}")
            if 0.2 < self.acceptance_rate < 0.5:
                print(f"  ✓ Acceptance rate in optimal range [0.2, 0.5]")
            else:
                print(f"  ⚠ Consider adjusting step_size or proposal_cov")

        result = np.array(chains)
        if result.ndim == 2:
            result = result[np.newaxis, :, :]
        return result


# ============================================================
# 2. Gibbs Sampler
# ============================================================

class GibbsSampler:
    """
    Gibbs 采样器。

    对每个参数分量依次从全条件分布采样: θ_j^(t+1) ~ p(θ_j | θ_{-j}^(t), y)

    使用
    ----
    gibbs = GibbsSampler(conditional_samplers, initial_theta)
    samples = gibbs.sample(n_iter=10000, n_burnin=2000)
    """

    def __init__(self, conditional_samplers: list, initial_theta: np.ndarray):
        """
        参数
        ----
        conditional_samplers : list of callable
            每个分量的全条件采样函数: sampler_j(theta_current, j) -> new_theta_j。
        initial_theta : np.ndarray
        """
        self.samplers = conditional_samplers
        self.theta = np.asarray(initial_theta, dtype=float).copy()
        self.d = len(self.theta)

    def sample(self, n_iter: int = 10000, n_burnin: int = 2000,
               thin: int = 1, progress: bool = True) -> np.ndarray:
        """
        运行 Gibbs 采样。

        返回
        ----
        np.ndarray, shape ((n_iter - n_burnin) / thin, d)
        """
        theta = self.theta.copy()
        samples = []
        total = n_burnin + n_iter

        for i in range(total):
            # 依次更新每个分量
            for j in range(self.d):
                theta[j] = self.samplers[j](theta, j)

            if i >= n_burnin and (i - n_burnin) % thin == 0:
                samples.append(theta.copy())

            if progress and (i + 1) % (total // 10) == 0:
                print(f"  Gibbs: {100 * (i + 1) / total:.0f}%")

        return np.array(samples)


# ============================================================
# 3. Hamiltonian Monte Carlo (HMC)
# ============================================================

class HamiltonianMC:
    """
    Hamiltonian Monte Carlo 采样器。

    通过模拟物理系统的 Hamiltonian 动力学进行高效采样:
        H(θ, p) = -log p(θ) + 1/2 p^T M^{-1} p

    使用 Leapfrog 积分器在相空间中移动，然后 Metropolis 接受/拒绝。

    使用
    ----
    hmc = HamiltonianMC(log_target, grad_log_target, initial_theta)
    samples = hmc.sample(n_iter=5000, n_burnin=1000)
    """

    def __init__(self, log_target: Callable, grad_log_target: Callable,
                 initial_theta: np.ndarray, step_size: float = 0.01,
                 n_leapfrog: int = 20, mass_matrix: Optional[np.ndarray] = None):
        """
        参数
        ----
        log_target : callable
            非标准化对数后验: f(theta) -> float。
        grad_log_target : callable
            对数后验的梯度: g(theta) -> np.ndarray, shape (d,)。
        initial_theta : np.ndarray
        step_size : float, Leapfrog 步长 ε
        n_leapfrog : int, Leapfrog 步数 L
        mass_matrix : np.ndarray or None, 质量矩阵 M。None 时用单位矩阵。
        """
        self.log_target = log_target
        self.grad_log_target = grad_log_target
        self.theta = np.asarray(initial_theta, dtype=float).copy()
        self.d = len(self.theta)
        self.epsilon = step_size
        self.L = n_leapfrog

        if mass_matrix is None:
            self.M = np.eye(self.d)
            self.M_inv = np.eye(self.d)
        else:
            self.M = np.asarray(mass_matrix, dtype=float)
            self.M_inv = np.linalg.inv(self.M)

    def _leapfrog(self, theta: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Leapfrog 积分器: 在相空间中前进 L 步。"""
        theta = theta.copy()
        p = p.copy()

        # 半步动量更新
        grad = self.grad_log_target(theta)
        p = p + 0.5 * self.epsilon * grad

        # L-1 步全步更新
        for _ in range(self.L - 1):
            theta = theta + self.epsilon * (self.M_inv @ p)
            grad = self.grad_log_target(theta)
            p = p + self.epsilon * grad

        # 最后一步位置 + 半步动量
        theta = theta + self.epsilon * (self.M_inv @ p)
        grad = self.grad_log_target(theta)
        p = p + 0.5 * self.epsilon * grad

        return theta, -p  # 动量取反 (可逆性)

    def _kinetic_energy(self, p: np.ndarray) -> float:
        """动能: K(p) = 1/2 p^T M^{-1} p"""
        return 0.5 * (p @ self.M_inv @ p)

    def sample(self, n_iter: int = 5000, n_burnin: int = 1000,
               thin: int = 1, progress: bool = True) -> np.ndarray:
        """
        运行 HMC 采样。

        返回
        ----
        np.ndarray, shape ((n_iter - n_burnin) / thin, d)
        """
        theta = self.theta.copy()
        samples = []
        n_accept = 0
        total = n_burnin + n_iter

        for i in range(total):
            # 从标准正态分布中采样动量
            p = np.random.multivariate_normal(np.zeros(self.d), self.M)

            # 记录初始状态
            theta_old = theta.copy()
            H_old = -self.log_target(theta_old) + self._kinetic_energy(p)

            # Leapfrog 积分
            theta_new, p_new = self._leapfrog(theta_old, p)

            # Metropolis 接受
            H_new = -self.log_target(theta_new) + self._kinetic_energy(p_new)
            log_alpha = H_old - H_new

            if log_alpha > 0 or np.log(np.random.rand()) < log_alpha:
                theta = theta_new
                n_accept += 1

            if i >= n_burnin and (i - n_burnin) % thin == 0:
                samples.append(theta.copy())

            if progress and (i + 1) % (total // 5) == 0:
                print(f"  HMC: {100 * (i + 1) / total:.0f}%")

        self.acceptance_rate = n_accept / total
        if progress:
            print(f"  HMC acceptance rate: {self.acceptance_rate:.3f}")
            if 0.6 < self.acceptance_rate < 0.9:
                print(f"  ✓ Optimal HMC acceptance rate [0.6, 0.9]")

        return np.array(samples)


# ============================================================
# 4. No-U-Turn Sampler (NUTS)
# ============================================================

class NUTS:
    """
    No-U-Turn Sampler (Hoffman & Gelman, 2014).

    自适应 HMC 变体，通过递归构建二叉树自动确定最优 Leapfrog 步数。
    这是 Stan 和 PyMC 使用的核心采样算法。

    参考: Hoffman & Gelman (2014), JMLR 15:1593-1623

    使用
    ----
    nuts = NUTS(log_target, grad_log_target, initial_theta)
    samples = nuts.sample(n_iter=5000, n_burnin=1000)
    """

    def __init__(self, log_target: Callable, grad_log_target: Callable,
                 initial_theta: np.ndarray, step_size: float = 0.01):
        """
        参数
        ----
        log_target, grad_log_target : callable
            对数后验及其梯度。
        initial_theta : np.ndarray
        step_size : float, 初始步长 (会被自动调整)
        """
        self.log_target = log_target
        self.grad_log_target = grad_log_target
        self.theta = np.asarray(initial_theta, dtype=float).copy()
        self.d = len(self.theta)
        self.epsilon = step_size

        # 质量矩阵和其逆 (初始为单位矩阵)
        self.M = np.eye(self.d)
        self.M_inv = np.eye(self.d)
        self.mu = np.log(10 * self.epsilon)  # log epsilon 的均值

        # 自适应参数
        self._t = 0
        self._gamma = 0.05
        self._t0 = 10
        self._kappa = 0.75
        self._log_epsilon_bar = np.log(self.epsilon)

    def _leapfrog_step(self, theta: np.ndarray, p: np.ndarray,
                       direction: int) -> Tuple[np.ndarray, np.ndarray]:
        """单步 Leapfrog, direction = +1 或 -1。"""
        p = p.copy()
        theta = theta.copy()

        p = p + 0.5 * direction * self.epsilon * self.grad_log_target(theta)
        theta = theta + direction * self.epsilon * (self.M_inv @ p)
        p = p + 0.5 * direction * self.epsilon * self.grad_log_target(theta)

        return theta, p

    def _build_tree(self, theta: np.ndarray, p: np.ndarray,
                    u: float, direction: int, depth: int,
                    theta0: np.ndarray, p0: np.ndarray):
        """
        递归构建二叉树（NUTS 核心）。

        返回
        ----
        theta_minus, theta_plus, p_minus, p_plus,
        theta_prime, n_prime, s_prime, alpha, n_alpha
        """
        if depth == 0:
            # 基础情形: 单步 Leapfrog
            theta_prime, p_prime = self._leapfrog_step(theta, p, direction)

            log_H = -self.log_target(theta_prime) + 0.5 * (p_prime @ self.M_inv @ p_prime)
            n_prime = int(np.log(u) <= log_H)  # 1 if accepted, 0 otherwise
            s_prime = int(np.log(u) < 500 + log_H)  # U-turn check 通过

            return (theta_prime, theta_prime, p_prime, p_prime,
                    theta_prime, n_prime, s_prime, min(1, np.exp(log_H)), 1)

        else:
            # 递归构建左右子树
            (theta_minus1, theta_plus1, p_minus1, p_plus1,
             theta_prime1, n_prime1, s_prime1, alpha1, n_alpha1) = \
                self._build_tree(theta, p, u, direction, depth - 1,
                                 theta0, p0)

            if s_prime1 == 0:
                return (theta_minus1, theta_plus1, p_minus1, p_plus1,
                        theta_prime1, n_prime1, 0, alpha1, n_alpha1)

            if direction == -1:
                theta, p = theta_minus1, p_minus1
            else:
                theta, p = theta_plus1, p_plus1

            (theta_minus2, theta_plus2, p_minus2, p_plus2,
             theta_prime2, n_prime2, s_prime2, alpha2, n_alpha2) = \
                self._build_tree(theta, p, u, direction, depth - 1,
                                 theta0, p0)

            # 以概率 n_prime2 / (n_prime1 + n_prime2) 接受第二子树的提议
            n_total = n_prime1 + n_prime2
            if n_total > 0 and np.random.rand() < n_prime2 / n_total:
                theta_prime1 = theta_prime2

            alpha_new = alpha1 + alpha2
            n_alpha_new = n_alpha1 + n_alpha2

            # U-turn 条件检查
            if direction == -1:
                theta_minus = theta_minus2
                d_theta = theta_plus1 - theta_minus
            else:
                theta_plus = theta_plus2
                d_theta = theta_plus - theta_minus1

            # U-turn: 检查梯度方向是否反转
            if s_prime2 == 1:
                # 简化版 U-turn 检查: dot product of position diff and momentum
                p_avg = (p_plus1 + p_minus1) / 2 if direction == -1 else (p_plus2 + p_minus2) / 2
                s_prime = int(d_theta @ p_avg > 0)
            else:
                s_prime = 0

            return (theta_minus1 if direction == -1 else theta_minus2,
                    theta_plus2 if direction == 1 else theta_plus1,
                    p_minus1 if direction == -1 else p_minus2,
                    p_plus2 if direction == 1 else p_plus1,
                    theta_prime1, n_total, s_prime, alpha_new, n_alpha_new)

    def _adapt_step_size(self, alpha: float):
        """自适应调整步长 (Dual Averaging, Nesterov, 2009)。"""
        self._t += 1
        accept_stat = alpha

        # HMC 的最优接受率约为 0.651 (Beskos et al., 2013)
        delta = 0.651
        eta = self._t ** (-self._kappa)
        self._log_epsilon_bar = (eta * (delta - accept_stat)
                                 + (1 - eta) * self._log_epsilon_bar)

        log_epsilon = self.mu - np.sqrt(self._t) / self._gamma * self._log_epsilon_bar
        self.epsilon = np.exp(log_epsilon)

    def sample(self, n_iter: int = 5000, n_burnin: int = 1000,
               max_tree_depth: int = 10, thin: int = 1,
               progress: bool = True) -> np.ndarray:
        """
        运行 NUTS 采样。

        返回
        ----
        np.ndarray, shape ((n_iter - n_burnin) / thin, d)
        """
        theta = self.theta.copy()
        samples = []
        total = n_burnin + n_iter

        for i in range(total):
            # 重新采样动量
            p0 = np.random.multivariate_normal(np.zeros(self.d), self.M)

            # 切片变量
            log_u = (self.log_target(theta)
                     - 0.5 * (p0 @ self.M_inv @ p0)
                     + np.log(np.random.rand()))

            # 初始化二叉树
            theta_minus = theta.copy()
            theta_plus = theta.copy()
            p_minus = p0.copy()
            p_plus = p0.copy()
            n = 1  # 接受的提议数
            s = 1  # 是否继续

            # 递归构建二叉树
            for depth in range(max_tree_depth):
                # 随机选择方向
                direction = 1 if np.random.rand() < 0.5 else -1

                if direction == -1:
                    (theta_minus, _, p_minus, _, theta_prime,
                     n_prime, s_prime, alpha, n_alpha) = \
                        self._build_tree(theta_minus, p_minus, log_u,
                                         direction, depth, theta, p0)
                else:
                    (_, theta_plus, _, p_plus, theta_prime,
                     n_prime, s_prime, alpha, n_alpha) = \
                        self._build_tree(theta_plus, p_plus, log_u,
                                         direction, depth, theta, p0)

                if s_prime == 0:
                    break

                # 以概率 min(1, n_prime / n) 接受提议
                if np.random.rand() < min(1, n_prime / n):
                    theta = theta_prime

                n += n_prime

                # U-turn 检查
                d_theta = theta_plus - theta_minus
                s = int(s_prime and (d_theta @ p_minus >= 0) and (d_theta @ p_plus >= 0))

                if s == 0:
                    break

            # 自适应步长 (仅在 warm-up 阶段)
            if i < n_burnin:
                self._adapt_step_size(alpha / max(n_alpha, 1))

            if i >= n_burnin and (i - n_burnin) % thin == 0:
                samples.append(theta.copy())

            if progress and (i + 1) % (total // 5) == 0:
                print(f"  NUTS: {100 * (i + 1) / total:.0f}%, ε={self.epsilon:.4f}")

        if progress:
            print(f"  NUTS final step size: ε = {self.epsilon:.4f}")

        return np.array(samples)
