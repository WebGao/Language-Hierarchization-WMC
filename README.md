# Language Hierarchization and Human Working Memory Limits

This repository contains the source code, statistical data, and theoretical derivations for the manuscript:  
**"Language Hierarchization Provides the Optimal Solution to Human Working Memory Limits"** (Submitted to *Nature*).

## Overview
This study explores the relationship between the hierarchical structure of human language and the constraints of working memory. By analyzing multiple corpora (Classics, Alice, and Child Spoken Language), we demonstrate that language hierarchization serves as an optimal strategy to maximize information transfer within human cognitive limits.

## Key Contents
- `src/`: Core scripts for corpus processing and hierarchical analysis.
- `stats/`: Detailed statistical data across different languages (as seen in Table S1 of the Supplementary Information).
- `mle_derivation/`: Implementation of the Maximum Likelihood Estimation (MLE) for working memory capacity ($\theta$).
- `data/`: Processed metadata from the Classics, Alice, and Child Spoken Language corpora.

## Theoretical Appendix
The repository also includes the mathematical framework for:
- Deriving the likelihood function $L(u|\theta)$.
- Solving the MLE for $\theta_{MLE} = \bar{U}$.
- Validation of the optimal solution against empirical linguistic data.

## Installation & Usage
(请在此处补充您的运行环境，例如：)
```bash
git clone [https://github.com/您的用户名/Language-Hierarchization-WMC.git](https://github.com/您的用户名/Language-Hierarchization-WMC.git)
cd Language-Hierarchization-WMC
# 运行主要的统计脚本
python main_analysis.py
