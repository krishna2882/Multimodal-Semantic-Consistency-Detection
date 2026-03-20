# Multimodal Sentiment Analysis using CLIP and Feature Engineering

## Overview

This project explores multimodal sentiment analysis of social media posts by combining information from text, images, and hashtags. The system uses CLIP embeddings along with engineered cross modal features to capture relationships between different modalities.

The pipeline moves from simple unimodal baselines to a feature engineered multimodal model, followed by experiments on class imbalance handling, feature ablation analysis, and cross dataset generalization.

The entire workflow is implemented as a sequence of structured notebooks forming a complete experimental pipeline.

---

## Dataset

### MVSA Dataset

The primary dataset used is the MVSA Multimodal Sentiment Analysis dataset. Each sample contains

- an image
- associated tweet text
- multiple sentiment annotations

To obtain a reliable sentiment label, majority voting is applied across annotators.

Basic dataset analysis includes

- label distribution
- tweet length statistics
- hashtag frequency analysis

---

### Cross Dataset Evaluation

To test generalization, the trained model is evaluated on the Facebook Hateful Memes dataset. This dataset contains image text pairs where understanding both modalities is required to correctly interpret the meaning.

---

## Project Pipeline

### 1 Dataset Preparation

Notebook: `01-dataset-preparation.ipynb`

This stage prepares the MVSA dataset for modeling.

Main steps include

- parsing the MVSA label file
- applying majority voting across annotators
- building a structured dataset
- performing exploratory analysis
- exporting the cleaned dataset for feature extraction

---

### 2 Feature Extraction

Notebook: `02-feature-extraction.ipynb`

Multimodal representations are generated using CLIP ViT-B/32.

Extracted features include

Text features

- cleaned tweet text
- emoji demojization
- CLIP text embeddings

Image features

- CLIP image embeddings

Hashtag features

- extracted hashtags
- CLIP embeddings for hashtags

Additional metadata features

- hashtag count
- hashtag frequency

These features form the base representation used by all models.

---

### 3 Baseline Models

Notebook: `03-baseline-models.ipynb`

Initial experiments evaluate simple unimodal and multimodal baselines.

Models used

- Logistic Regression
- XGBoost

Experiments include

- text only classification
- image only classification
- hashtag only classification
- simple multimodal concatenation

These baselines establish how much information each modality contributes individually.

---

### 4 Multimodal Baseline

Notebook: `04-multimodal-baseline.ipynb`

This stage introduces interaction features between modalities.

Feature engineering includes

- embedding difference vectors
- cosine similarity between modalities

Examples

- text image similarity
- text hashtag similarity
- image hashtag similarity

These features help capture semantic agreement between modalities.

A gradient boosted classifier XGBoost is trained using the combined multimodal features.

---

### 5 Proposed Model

Notebook: `05-proposed-model.ipynb`

The proposed model combines multiple feature categories.

Core embeddings

- text embeddings
- image embeddings
- hashtag embeddings

Cross modal difference features

- text image difference
- text hashtag difference
- image hashtag difference

Similarity features

- text image similarity
- text hashtag similarity
- image hashtag similarity

Metadata features

- hashtag count
- hashtag frequency

All features are concatenated into a single multimodal feature vector and used to train an XGBoost classifier.

---

### 6 Class Imbalance Experiments

Notebook: `06-class-imbalance-methods.ipynb`

Because sentiment datasets often contain skewed label distributions, several models are evaluated with class imbalance handling strategies.

Models tested

- Logistic Regression
- Random Forest
- MLP
- XGBoost

Techniques used

- class weight balancing
- scale_pos_weight for boosting models

Performance is evaluated using classification metrics and ROC AUC.

---

### 7 Feature Ablation and Analysis

Notebook: `07-analysis-and-visualization.ipynb`

This stage evaluates the contribution of each modality using feature ablation experiments.

Feature configurations tested

- text only
- text plus image
- text plus image plus hashtag
- full engineered multimodal feature set

The analysis shows how each modality contributes to model performance.

---

### 8 Cross Dataset Evaluation

Notebook: `08-cross-dataset-valuation.ipynb`

To test generalization, the trained model is evaluated on the Hateful Memes dataset.

Steps include

- extracting CLIP embeddings for meme text and images
- applying the trained multimodal classifier
- measuring predictive performance on the new dataset

This experiment evaluates whether the learned multimodal representation transfers across domains.

---

### 9 Final Evaluation and Visualization

Notebook: `09-final-analysis-and-visualisation.ipynb`

The final stage summarizes experimental results.

Visualizations include

- confusion matrices
- classification reports
- ROC curves
- ROC AUC comparison across datasets

Results are reported for

- MVSA dataset
- Hateful Memes dataset

---

## Model Summary

The final system combines

- CLIP embeddings for multimodal representation
- cross modal similarity features
- engineered interaction features
- XGBoost classifier

This architecture captures both individual modality signals and semantic alignment between modalities.

---

## Key Insights

- CLIP embeddings provide strong representations for multimodal content
- cross modal similarity features provide complementary information
- hashtag embeddings improve context understanding
- feature engineering improves performance over simple feature concatenation
- the model demonstrates moderate generalization across datasets

---

## Technologies Used

- Python
- PyTorch
- CLIP
- Scikit learn
- XGBoost
- NumPy
- Pandas
- Matplotlib
- Seaborn

---

## Repository Structure

```
project-root/
│
├── 01-dataset-preparation.ipynb
├── 02-feature-extraction.ipynb
├── 03-baseline-models.ipynb
├── 04-multimodal-baseline.ipynb
├── 05-proposed-model.ipynb
├── 06-class-imbalance-methods.ipynb
├── 07-analysis-and-visualization.ipynb
├── 08-cross-dataset-valuation.ipynb
└── 09-final-analysis-and-visualisation.ipynb
```

Each notebook corresponds to one stage of the experimental pipeline from data preparation to final evaluation.