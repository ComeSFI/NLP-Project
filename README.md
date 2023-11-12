# NLP-Project : Movie review sentiment analysis

## Overview 
This NLP project aims to classify movie reviews as either positive or negative using machine learning techniques. The primary goal is to build a model that can automatically determine the sentiment expressed in a given movie review.

## Dataset
The model is trained on a dataset of movie reviews labeled with their corresponding sentiments. The dataset is sourced from https://www.kaggle.com/datasets/andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews/

## Features

- `Data Exploratory Analysis`
- `Baseline Model`
- `Improved baseline model (Chossing the right model)`
- `Deep Learning model`

## Structure

```
└── NLP_Project/
    ├── data/
      ├── utils.py
      ├── x-train.csv
      └── y_train.csv
    ├── 0 - preprocessing_tocsv.ipynb
    ├── 1- explortory_data_analysis.ipynb
    ├── 2 - baseline_model.ipynb
    ├── 3 - Baseline improve.ipynb
    ├── 4 - deep_learning.ipynb
    ├── requirements.txt
    └── preprocessing.py
```

## Getting Started 

1. Clone the repository and go in the cloned directory
```sh
git clone https://github.com/ComeSFI/NLP-Project
cd NLP-Project
```
2. Run the notebooks

## Result

With our baseline model, and even when improving our model we had a precision under 80%. In the result showed here the baseline model seems better than the improved model, but it is due to the fact that we did not use the same data size, because it was too long. But our deep learning model have a very good precision (>0.99)

## Futur improvement

Implement BERT Model functionning
Use same data size everywhere
Comment Code

## Contributor

Côme SAINT FORT ICHON
come.saintfortichon@epfedu.fr

## Acknowledgement

https://stackoverflow.com/
https://chat.openai.com/
https://github.com/RPegoud
