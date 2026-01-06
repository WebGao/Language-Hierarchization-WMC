# Language Hierarchization and Human Working Memory Limits

This repository contains the source code for the manuscript:

**"Language Hierarchization Provides the Optimal Solution to Human Working Memory Limits"**

## 📂 Repository Structure

The repository is organized into three main modules, each corresponding to a specific corpus analyzed in the study. Each module contains its own data processing and analysis scripts.

```text
Language-Hierarchization-WMC
├── OpenNodes_Alice               # Multilingual analysis of "Alice's Adventures in Wonderland"
│   ├── data/                     # Processed text data for 8 languages
│   └── main.py                   # Main analysis script for Alice corpus
├── OpenNodes_Child               # Analysis of the Child Spoken Language Corpus (Ages 3-8)
│   ├── data/                     # Longitudinal spoken language data
│   └── main.py                   # Main analysis script for developmental data
└── OpenNodes_Natural_Language    # Analysis of the Classics Corpus (English)
    ├── data/                     # Large-scale natural language dataset
    └── main.py                   # Main analysis script for adult natural language

```

## 📊 Datasets

1. **OpenNodes_Natural_Language (Classics Corpus):** A large-scale English corpus consisting of approximately 34,995 sentences, used to validate the hierarchization theory in adult natural language.
2. **OpenNodes_Alice (Alice Corpus):** A multilingual dataset covering 8 languages (English, Chinese, French, German, Russian, Japanese, Italian, and Spanish) to test the cross-linguistic universality of the optimal solution.
3. **OpenNodes_Child (Child Spoken Language):** Developmental data categorized by age (3-8 years old) to examine how the hierarchization strategy evolves alongside working memory capacity.

## 🚀 Getting Started

### Prerequisites

* Python 3.8 or higher
* Required packages: `numpy`, `pandas`, `scipy`, `matplotlib`, `stanza` (Installation: `pip install numpy pandas scipy matplotlib stanza`)

### Running the Analysis

Each module is self-contained. You can replicate the results presented in the paper by running the `main.py` script within each directory.

For example, to analyze the Alice Corpus:

```bash
cd OpenNodes_Alice
python main.py

```

To analyze the Child Spoken Language data:

```bash
cd OpenNodes_Child
python main.py

```

<!-- ## ⚙️ Core Methodology

The code implements:

* **Maximum Likelihood Estimation (MLE):** Calculating the working memory capacity () from linguistic structures.
* **Entropy Calculation:** Measuring the information transfer efficiency under different processing mechanisms (Linear vs. Hierarchical).
* **Optimization Validation:** Demonstrating how language hierarchization minimizes cognitive load while maximizing information density. -->

<!-- ## 📜 Citation

If you use this code or data in your research, please cite:

> *Chen, L., Gao, W., Wu, J., Wu, J., & Friederici, A. D. (2026). Language Hierarchization Provides the Optimal Solution to Human Working Memory Limits. Nature (under review).* -->

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## ✉️ Contact

For questions regarding the code or data, please contact the corresponding authors:

* Prof. Dr. Luyao Chen (harry-luyao.chen@polyu.edu.hk)
* Dr. Weibo Gao (weibogao@mail.ustc.edu.cn)
