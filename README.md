# CURA
> Academic Use Only | Commercial use prohibited without license  
> Must cite: Seo et al., 2025 (arXiv:2509.24601)


[CURA GitHub Repository](https://github.com/SION001122/CURA)

CURA_CORE <--- NOMAL_TASK

---

##  License

This project is licensed under the **CURA NonCommercial Academic License v1.1**.

## Intellectual Property

This project includes the CURA architecture described in the patent application:

All usage of this architecture in commercial or research applications must comply with the associated licensing terms.

 **Commercial use is strictly prohibited.**  
This includes any use by **companies, startups, government institutions, or corporate-affiliated research labs**, regardless of purpose.

 Permitted:
- Individual students and researchers
- Academic classroom or university-affiliated research
- Public model sharing for educational purposes (e.g., Hugging Face), with proper attribution

 Not permitted:
- Any use involving business registration or commercial funding
- Internal corporate testing or integration
- Product development or commercialization of derivative models

By using this software, you agree to these terms.  
For full details, see the [LICENSE](./LICENSE) file.

 For exceptions or commercial licensing, contact: **sion@curalicense.org**


##  Hugging Face Model Card Template

The file `huggingface_model_card_example.md` in this repository is a **template** for users who wish to publish CURA-based models on Hugging Face.  
It outlines the correct license tags, attribution format, and metadata requirements.

**Do not upload this template directly.**  
Please replace the placeholder fields (e.g., `CURAv1_MODEL_NAME`, `your_email@example.com`) with your own details.


##  Dataset Preparation & Execution Guide

Most tasks in this repository are configured to automatically download the required datasets during execution.

- For **`CURA[ETTm1]`** and **`CURA[UCI_HAR_Dataset]`**,  
  you can manually trigger the dataset download using the provided `download.py` script inside each directory.

- For the **FallAllD** dataset,  
  please refer to the **"Dataset: FallAllD"** section in this README.  
  
---

##  Dataset: House Prices (Kaggle)

This task uses the dataset from the Kaggle competition:  
 https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

1. **Download the dataset file** (`train.csv`) from the competition page above.

2. **Place the downloaded file** into the following directory:

3. **Run the preprocessing script:**
```bash
python preprocess_house_prices.py
python HOUSE_PRICES_REGRESSION_TASK.py
```


##  Dataset

##  House Prices – Advanced Regression Techniques
This project uses the dataset from the **Kaggle Competition:  
[House Prices – Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)**.

- The dataset is provided by **Kaggle** for research and educational purposes.
- Copyright belongs to the original data contributors and Kaggle.
- Please refer to the [competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) for more details and terms of use.

 **Note:** This repository does **not redistribute the dataset directly**.  
To obtain the data, please download it manually from Kaggle.

---


##  ETTm1
Dataset: ETTm1  
Source: Zhou et al., “Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting”  
GitHub: https://github.com/zhouhaoyi/ETDataset  
License: MIT
##  UCI HAR Dataset (Human Activity Recognition Using Smartphones)  
Dataset: UCI HAR Dataset (Human Activity Recognition Using Smartphones)  
Source: UCI Machine Learning Repository  
Original Paper:  
Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013).  
A Public Domain Dataset for Human Activity Recognition Using Smartphones.  
Proceedings of the 21st European Symposium on Artificial Neural Networks (ESANN).
URL: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones


##  MNIST

This project uses the MNIST dataset, made publicly available for research and educational purposes.  
Original source: [http://yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist)

The dataset will be automatically downloaded via `torchvision.datasets.MNIST()` if not found in the local directory.


##  S&P 500

This project uses historical stock data for the S&P 500 Index (`^GSPC`)  
downloaded via the [Yahoo Finance](https://finance.yahoo.com/) API using the [yfinance](https://github.com/ranaroussi/yfinance) library.

The data is used solely for **educational and non-commercial research purposes**.

>  No raw data files are included in this repository.  
> Users must download the data directly via the provided code.

##  Dataset: FallAllD

This project uses the FallAllD dataset, an open dataset of simulated human falls and activities of daily living.  
The dataset is published under the **Creative Commons Attribution License (CC BY)** and made available via IEEE DataPort.

Dataset Source:  
Majd Saleh, Régine Le Bouquin Jeannès, “FallAllD: An Open Dataset of Human Falls and Activities of Daily Living,”  
IEEE Sensors Journal. DOI: [10.1109/JSEN.2020.3018335](https://doi.org/10.1109/JSEN.2020.3018335)

## Dataset: AG News  
Dataset: AG News  
Source: Zhang et al., "Character-level Convolutional Networks for Text Classification" (2015)  
License: CC BY 4.0

##  Amazon Polarity
Dataset: Amazon Polarity  
Source: Zhang et al., 2015  
License: CC BY 4.0  
Collected from Amazon.com customer reviews
##   BoolQ (Boolean Questions)
Dataset: BoolQ  
Source: Clark et al., "BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions" (2019)  
License: MIT

##  CIFAR-10

Dataset: CIFAR-10  
Source: Alex Krizhevsky, Learning Multiple Layers of Features from Tiny Images  
License: MIT


##  HellaSwag
Dataset: HellaSwag  
Source: Zellers et al., "HellaSwag: Can a Machine Really Finish Your Sentence?" (2019)  
License: Apache 2.0  
https://rowanzellers.com/hellaswag/
