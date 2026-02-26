# HeteroOmics-BRCA  
A Research-Oriented Web Platform for Breast Cancer Molecular Subtype Prediction

---

## Overview

**HeteroOmics-BRCA** is a research-driven web platform designed for molecular subtype prediction in breast cancer using multi-omics data integration.

This system demonstrates the deployment of deep representation learning and graph-based modeling within an interactive web environment, bridging computational modeling and practical usability.

The platform is intended for methodological demonstration and research prototyping.

---

## Research Motivation

Breast cancer is a heterogeneous disease characterized by multiple molecular subtypes, each associated with distinct prognosis and therapeutic strategies.

Accurate subtype classification remains essential for:

- Precision medicine
- Risk stratification
- Treatment planning

This platform explores the integration of heterogeneous omics features through deep learning architectures to support automated subtype prediction.

---

## Methodological Framework

The backend computational pipeline includes:

### 1. Representation Learning
- Variational Autoencoder (VAE) for modality-specific latent embedding
- Dimensionality reduction and noise suppression

### 2. Heterogeneous Feature Integration
- Multi-scale graph-based modeling
- Feature-level interaction modeling

### 3. Supervised Classification
- Subtype prediction head
- Probability output for interpretability

The web interface encapsulates trained model inference for real-time prediction.

---

## System Architecture

### Frontend
- HTML / CSS
- Data upload interface
- Result visualization

### Backend
- Python
- Flask
- scikit-learn
- pandas
- numpy

Model inference is executed server-side upon user data submission.

---

## Local Deployment

Clone repository:

```bash
git clone https://github.com/your-username/HeteroOmics-BRCA.git
cd HeteroOmics-BRCA
pip install -r requirements.txt
python app.py

```
# 📂 Data Availability

## Public Datasets

The datasets used in this study are publicly available via **Figshare**.

They include:

- mRNA expression data  
- DNA methylation data
- copy numbers alterations data
- Corresponding clinical annotations  

All datasets have been deposited in Figshare and are assigned a DOI to ensure reproducibility and long-term accessibility.

---

## Dataset Access

The datasets can be accessed through the following Figshare repository:

🔗 https://figshare.com/account/home

Each dataset is associated with a DOI for citation and academic use.

---

## Dataset Description

| Data Type      | Description                          | Format |
|---------------|--------------------------------------|--------|
| mRNA          | Gene expression matrix               | CSV    |
| Methylation   | DNA methylation beta value matrix    | CSV    |
|  cna          | copy numbers alterations matrix      | CSV    |
| Clinical Data | Clinical metadata and subtype labels | CSV    |

---

## Reproducibility Statement

All experiments reported in this repository were conducted using the above publicly available datasets.

Researchers can reproduce the results by:

1. Downloading the datasets from Figshare  
2. Following the modeling pipeline described in this repository  
3. Using the same preprocessing and validation procedures  

---

## Citation

If you use these datasets in your research, please cite the corresponding Figshare DOI provided on the dataset page.
