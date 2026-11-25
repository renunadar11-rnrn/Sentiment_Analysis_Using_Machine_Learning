# Restaurant Review Sentiment Analysis

Reading individual reviews gives a restaurant detailed feedback, but manually sorting every new review into positive and negative feedback takes time. This project uses machine learning to automate that first pass. It trains on 900 labeled historical reviews and applies the selected model to 100 new reviews, producing a CSV with a sentiment label for each review.

## How it works

1. **Preprocess the reviews.** Clean the text, remove common English stop words while keeping `not`, and apply Porter stemming.
2. **Represent the text as features.** Use a stratified 80/20 split and fit a bag-of-words vectorizer on the training reviews. Use the same vocabulary to transform the evaluation reviews and new reviews.
3. **Compare classifiers.** Evaluate four models and select the one with the highest F1 score for positive reviews.
4. **Predict new reviews.** Load the saved model and vectorizer, predict sentiment for `fresh_reviews.tsv`, and export the review text and predicted labels to `predicted_reviews.csv`.

## Results

**Linear SVM** had the highest positive-class F1 score (**0.836**) and achieved **81.7% accuracy** on the 180-review split used to compare the models.

| Model | Accuracy | Positive-class F1 |
| --- | ---: | ---: |
| Linear SVM | 81.7% | 0.836 |
| Multinomial Naive Bayes | 78.3% | 0.815 |
| Logistic Regression | 78.9% | 0.810 |
| Random Forest | 80.6% | 0.809 |

The training notebook includes the confusion matrix and class-by-class results. The model predicted 84 of 99 positive reviews and 63 of 81 negative reviews correctly.

## Explore the project

- [Training notebook](sentiment_model_training.ipynb) — preprocessing, model comparison, and evaluation
- [Prediction notebook](fresh_reviews_prediction.ipynb) — sentiment predictions for new reviews
- [Prediction results](predicted_reviews.csv) — review text and predicted labels (`0` for negative, `1` for positive)
- [Project report](ABC_Restaurant_Sentiment_Analysis_Report.docx.docx) — Word report for the restaurant analysis

## Run locally

From the project folder, install the dependencies and start Jupyter:

```powershell
python -m pip install notebook pandas numpy nltk scikit-learn matplotlib joblib
jupyter notebook
```

Run `sentiment_model_training.ipynb` first to create the model and vectorizer. Then run `fresh_reviews_prediction.ipynb` to generate `predicted_reviews.csv`.
