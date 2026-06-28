# Changelog

All notable changes to BayesianEconometrics will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2024-06-15

### Added

- **Metropolis-Hastings**: Adaptive proposal covariance with acceptance rate tracking
- **Gibbs Sampler**: Full conditional sampling framework with user-defined log-conditionals
- **Hamiltonian Monte Carlo (HMC)**: Leapfrog integrator with configurable step size and path length
- **NUTS (No-U-Turn Sampler)**: Full Hoffman & Gelman (2014) implementation including:
  - Recursive binary tree doubling
  - Slice sampling within tree leaves
  - Dual Averaging for adaptive step size tuning
  - Maximum tree depth control
- **Bayesian Linear Regression**: Conjugate Normal-InverseGamma prior with analytic posterior
- **Bayesian Logit**: MH sampling with marginal effects computation
- **Bayesian Probit**: MH sampling with marginal effects computation
- **Bayesian VAR**: Minnesota prior, impulse response functions, h-step forecasts, Cholesky decomposition
- **Convergence Diagnostics**:
  - `gelman_rubin()`: R-hat statistic (within/between chain variance decomposition)
  - `effective_sample_size()`: Geyer monotone sequence truncation method
  - `autocorrelation()`: Per-lag autocorrelation function
  - `summary_diagnostics()`: Unified diagnostic report with HDI intervals
- **Prior distributions**: NormalPrior, InverseGammaPrior, WishartPrior, MVNPrior with log-pdf evaluation
- Example scripts and demo notebook
- Full README documentation with theory and algorithm explanation

### Dependencies

- numpy >= 1.20.0
- scipy >= 1.7.0

---

## [Unreleased]

### Planned

- Slice Sampling (Neal 2003)
- Riemannian HMC (Girolami & Calderhead 2011)
- Stochastic Gradient HMC (Chen et al. 2014)
- Hierarchical Bayesian models with group-level priors
- Bayesian Model Averaging (BMA) with marginal likelihood estimation
- WAIC / LOO-CV model comparison criteria
- Trace plot and posterior density plot utilities (matplotlib integration)
- Improved R-hat (Vehtari et al. 2021)
