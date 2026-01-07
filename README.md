# Language Hierarchization and Human Working Memory Limits

Source code and analytical framework for the for the manuscript:

**Language Hierarchization Provides the Optimal Solution to Human Working Memory Limits** > Luyao Chen, Weibo Gao, Junjie Wu, Jinshan Wu, and Angela D. Friederici > [[Read the Preprint on arXiv](https://arxiv.org/abs/2601.02740)]

## 👥 Authors

- Luyao Chen $^{1,2}$ [[Homepage](https://research.polyu.edu.hk/en/persons/luyao-chen/)] [[Google Scholar](https://scholar.google.com/citations?user=lFNU4bsAAAAJ)] (Co-corresponding Author)

- Weibo Gao $^3$ [[Homepage](http://home.ustc.edu.cn/~weibogao/)] [[Google Scholar](https://scholar.google.com/citations?user=k19RS74AAAAJ)]

- Junjie Wu $^{4,5,6,7}$

- Jinshan Wu $^{8}$ (Co-corresponding Author)

- Angela D. Friederici $^{2}$ [[Homepage](https://www.mpg.de/390351/human-cognitive-and-brain-sciences-friederici)] [[Google Scholar](https://scholar.google.com/citations?user=9-9aZewAAAAJ)] (Co-corresponding Author & Senior Author)

## 🏫 Affiliations

1. Department of Language Science and Technology, Hong Kong Polytechnic University, Hong Kong, China.

2. Department of Neuropsychology, Max Planck Institute for Human Cognitive and Brain Sciences, Leipzig, Germany.

3. State Key Laboratory of Cognitive Intelligence, University of Science and Technology of China, Hefei, China.

4. Academy of Psychology and Behavior, Tianjin Normal University, Tianjin, China.

5. Faculty of Psychology, Tianjin Normal University, Tianjin, China.

6. Department of Linguistics, Macquarie University, Sydney, Australia.

7. Tianjin Key Laboratory of Student Mental Health and Intelligence Assessment, China.

8. School of Systems Science, Beijing Normal University, Beijing, China.

---

## 📢 Availability Notice

**Data Availability:** To comply with institutional policies and protect pending research, the `data/` directories currently contain placeholder files. **Full datasets will be made publicly available upon the formal publication of the manuscript.**

**Code Availability:** The core analytical scripts (`main.py`) are provided. However, they require the datasets to execute fully. Reviewers or researchers requesting early access for validation may contact the authors.

---

## 📂 Repository Structure

The repository is organized into three main modules, each corresponding to a specific corpus analyzed in the study:

```text
Language-Hierarchization-WMC
├── OpenNodes_Alice                                     # Multilingual analysis (8 languages)
│   ├── data (Alice Corpus)/                            # [To be released upon publication]
│   └── main.py                                         # Script for cross-linguistic optimization analysis
├── OpenNodes_Child                                     # Developmental trajectory analysis (Ages 3-8)
│   ├── data (Child Spoken Language Corpus)/            # [To be released upon publication]
│   └── main.py                                         # Script for developmental evolution analysis
├── OpenNodes_Natural_Language                          # Adult natural language validation
│   ├── data (Classics Corpus)/                         # [To be released upon publication]
│   └── main.py                                         # Script for large-scale corpus analysis
└── Simulation                                          # Simulation of Non-natural Languages
    └── Simulation.ipynb                                # Script for simulation
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

## 📜 Citation

If you use this code or data in your research, please cite:

```
@misc{chen2026language,
      title={Language Hierarchization Provides the Optimal Solution to Human Working Memory Limits}, 
      author={Luyao Chen and Weibo Gao and Junjie Wu and Jinshan Wu and Angela D. Friederici},
      year={2026},
      eprint={2601.02740},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.02740}, 
}
```
---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](https://opensource.org/licenses/MIT) file for details.

## ✉️ Contact

For questions regarding the methodology, code, or data access, please contact:

* **Prof. Dr. Luyao Chen** (harry-luyao.chen@polyu.edu.hk)
* **Dr. Weibo Gao** (weibogao@mail.ustc.edu.cn)
