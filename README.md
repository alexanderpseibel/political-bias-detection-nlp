# Political Bias Detection in News Articles

This repository contains an NLP project investigating whether political 
bias in news articles can be automatically detected and classified. 
The project was submitted as part of the Natural Language Processing 
and Text Analytics course (CDSCO1002E) at Copenhagen Business School.

[Read the full report](Report.pdf)

---

## TL;DR
Fine-tuned transformer models (BERT, RoBERTa) can reliably detect 
political bias in news articles with up to 87% accuracy. Simpler 
models like Logistic Regression with TF-IDF already achieve 69%, 
making them a strong choice when computational resources are limited. 
The key insight: political bias lives in contextual framing, not 
surface-level word frequency.

---

## Tech Stack
**Core:** Python · Scikit-learn · PyTorch · Hugging Face Transformers

**Embeddings & NLP:** TF-IDF · Word2Vec · SBERT · BERT · RoBERTa

**Data & Visualization:** Pandas · NumPy · Matplotlib · Seaborn · NLTK · spaCy

**Topics:** Text Classification · Political Bias Detection · 
Transformer Fine-Tuning · NLP · Feature Engineering · 
Computational Linguistics

---

## Research Goal
We examine how much classification accuracy improves as we progress 
from simple count-based features to fully fine-tuned transformer 
embeddings, and what trade-offs arise between accuracy, computational 
cost, and interpretability when detecting political bias in news articles.

---

## Methodology
We use the BIGNEWSBLN dataset, a balanced subset of 150,000 articles 
drawn from 11 major U.S. media outlets, equally split across left, 
center, and right political categories. Labels reflect the outlet's 
general ideological stance as rated by AllSides and Ad Fontes Media. 
To avoid label leakage, source metadata was excluded from all models.

We evaluated a progressive range of feature representations and model 
architectures of increasing complexity. Sparse features (BoW and TF-IDF) 
with Naive Bayes and Logistic Regression served as interpretable baselines. 
Static dense embeddings (Word2Vec) were tested with MLP and BiLSTM models 
to capture semantic similarity and sequential patterns. Sentence embeddings 
(SBERT) were evaluated with an MLP to leverage contextual sentence-level 
representations. Finally, fine-tuned transformers (BERT and RoBERTa) were 
used for end-to-end contextual classification.

Preprocessing varied by method: full preprocessing for sparse features, 
light preprocessing for dense embeddings, and no preprocessing for 
transformers, which include their own tokenization pipelines.

---

## Key Findings

The results show a clear positive relationship between model complexity 
and classification accuracy:

| Model | Features | Accuracy |
|---|---|---|
| Naive Bayes | TF-IDF | 0.56 |
| Logistic Regression | TF-IDF | 0.69 |
| MLP | Word2Vec | 0.59 |
| BiLSTM | Word2Vec | 0.63 |
| MLP | SBERT | 0.52 |
| BERT | Transformer Embeddings | 0.86 |
| **RoBERTa** | **Transformer Embeddings** | **0.87** |

RoBERTa achieved the best performance with 87% accuracy and balanced 
F1-scores across all three classes (0.86–0.89), demonstrating robust 
detection without favoring any single political category.

Notably, Logistic Regression with TF-IDF already captured a large 
proportion of bias signals at 69% accuracy, while requiring a fraction 
of the computational resources. This highlights that classical models 
remain a viable option when interpretability or efficiency is a priority.

<img width="615" height="459" alt="image" src="https://github.com/user-attachments/assets/1396cd2d-421a-4646-9d4b-f09c18b2d333" />


The EDA revealed substantial vocabulary overlap across political 
categories. Words like "president", "government", and "trump" appear 
frequently in all three classes, confirming that surface-level word 
frequency alone is insufficient for reliable bias detection. Political 
bias is expressed through contextual framing rather than specific 
vocabulary, which explains why transformer models outperform simpler 
approaches so dramatically.

---

## Conclusions
Political bias in news articles can be reliably detected using NLP 
techniques. Fine-tuned transformers achieve the strongest results by 
capturing subtle contextual and stylistic patterns. However, these 
gains come at a significant computational cost, and classical models 
with sparse features remain competitive and highly interpretable 
alternatives. Any automated bias detection system should be accompanied 
by explanations of its predictions, particularly given the ethical 
implications of misclassification in polarised media environments.
