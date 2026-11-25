# Restaurant Review Sentiment Analysis

This project uses machine learning to classify restaurant reviews as positive or negative. It turns review text into bag-of-words features, compares several classifiers on historical reviews, and uses the selected model to predict sentiment for new reviews.

## Project files

| File | Purpose |
| --- | --- |
| `sentiment_model_training.ipynb` | Cleans historical reviews, compares classifiers, and saves the selected model and text vectorizer. |
| `fresh_reviews_prediction.ipynb` | Loads the saved model and vectorizer, predicts sentiment for new reviews, and exports the results. |
| `historical_reviews.tsv` | Labeled review data used for training and evaluation. |
| `fresh_reviews.tsv` | New reviews used for prediction. |
| `predicted_reviews.csv` | Exported reviews with their predicted sentiment. |

## How to run

Install the Python packages:

```powershell
python -m pip install jupyter pandas numpy nltk scikit-learn matplotlib joblib
```

Start Jupyter from the project folder:

```powershell
jupyter notebook
```

1. Run `sentiment_model_training.ipynb` to train and save the model.
2. Run `fresh_reviews_prediction.ipynb` to generate `predicted_reviews.csv`.
