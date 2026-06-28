from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bayesmetrics",
    version="1.0.0",
    author="wzx11223344",
    author_email="3521257027@QQ.com",
    description="从零实现的贝叶斯计量经济学引擎 — MCMC采样器(NUTS/HMC/Gibbs/MH)+贝叶斯模型(线性回归/Logit/VAR)+收敛诊断",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wzx11223344/bayesmetrics",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=["numpy>=1.20.0", "scipy>=1.7.0"],
    keywords="bayesian econometrics mcmc nuts hmc gibbs var",
)
