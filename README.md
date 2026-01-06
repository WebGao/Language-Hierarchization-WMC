# Language Hierarchization and Human Working Memory Limits

This repository provides the computational framework and statistical analysis for the manuscript:

**"Language Hierarchization Provides the Optimal Solution to Human Working Memory Limits"**

---

## 📢 Availability Notice

**Data Availability:** To comply with institutional policies and protect pending research, the `data/` directories currently contain placeholder files. **Full datasets will be made publicly available upon the formal publication of the manuscript.** **Code Availability:** The core analytical scripts (`main.py`) are provided. However, they require the datasets to execute fully. Reviewers or researchers requesting early access for validation may contact the authors.

---

## 📂 Repository Structure

The repository is organized into three main modules, each corresponding to a specific corpus analyzed in the study:

```text
Language-Hierarchization-WMC
├── OpenNodes_Alice               # Multilingual analysis (8 languages)
│   ├── data (Alice Corpus)/                     # [To be released upon publication]
│   └── main.py                   # Script for cross-linguistic optimization analysis
├── OpenNodes_Child               # Developmental trajectory analysis (Ages 3-8)
│   ├── data (Child Spoken Language Corpus)/                     # [To be released upon publication]
│   └── main.py                   # Script for developmental evolution analysis
└── OpenNodes_Natural_Language    # Adult natural language validation
    ├── data (Classics Corpus)/                     # [To be released upon publication]
    └── main.py                   # Script for large-scale corpus analysis

```

---

## 📊 Dataset Descriptions

1. **OpenNodes_Natural_Language (Classics Corpus):** A large-scale English corpus (~34,995 sentences) used to validate the hierarchization theory in stable adult language systems.
2. **OpenNodes_Alice (Alice Corpus):** A parallel multilingual dataset covering 8 languages (English, Chinese, French, German, Russian, Japanese, Italian, and Spanish) to test the cross-linguistic universality of the proposed optimal solution.
3. **OpenNodes_Child (Child Spoken Language):** Longitudinal developmental data categorized by age (3–8 years) to examine how hierarchization strategies emerge alongside growing working memory capacity.

---

## 🚀 Getting Started

### Prerequisites

* **Python:** 3.8 or higher
* **Dependencies:** `numpy`, `pandas`, `scipy`, `matplotlib`, `stanza`
* **Installation:**
```bash
pip install numpy pandas scipy matplotlib stanza

```



### Running the Analysis

Each module is self-contained. Once the datasets are released, you can replicate the results by running the `main.py` script within its respective directory.

```bash
# Example: Running the Alice Corpus analysis
cd OpenNodes_Alice
python main.py

```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](https://opensource.org/licenses/MIT) file for details.

## ✉️ Contact

For questions regarding the methodology, code, or data access, please contact:

* **Prof. Dr. Luyao Chen** (harry-luyao.chen@polyu.edu.hk)
* **Dr. Weibo Gao** (weibogao@mail.ustc.edu.cn)
