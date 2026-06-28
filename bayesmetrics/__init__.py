"""
BayesianEconometrics — 从零实现的贝叶斯计量经济学引擎
====================================================

纯 Python + NumPy/SciPy 实现的 MCMC 贝叶斯推断工具包。
所有采样器均从底层算法构建，包含完整的收敛诊断体系。

核心组件:
    - MCMC 采样器: Metropolis-Hastings, Gibbs, HMC, NUTS
    - 贝叶斯模型: 线性回归, Logit/Probit, VAR
    - 收敛诊断: R-hat, ESS, autocorrelation, trace
    - 先验分布: Normal, InverseGamma, Wishart, MVN

理论参考:
    - Gelman et al. (2013) Bayesian Data Analysis
    - Hoffman & Gelman (2014) NUTS
    - Koop (2003) Bayesian Econometrics
"""

from .samplers import (
    MetropolisHastings,
    GibbsSampler,
    HamiltonianMC,
    NUTS,
)

from .diagnostics import (
    gelman_rubin,
    effective_sample_size,
    autocorrelation,
    summary_diagnostics,
)

from .models import (
    BayesianLinearRegression,
    BayesianLogit,
    BayesianProbit,
    BayesianVAR,
)

from .priors import (
    NormalPrior,
    InverseGammaPrior,
    WishartPrior,
    MVNPrior,
    log_posterior,
)

__version__ = "1.0.0"
__author__ = "wzx11223344"
