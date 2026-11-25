# Restaurant Review Sentiment Analysis

An end-to-end natural language processing and machine learning project that classifies restaurant reviews, identifies recurring complaint themes, and helps ABC Restaurant prioritize customer follow-up.

The project compares four classifiers on 900 labeled historical reviews, applies the selected model to 100 new reviews, and summarizes the issues mentioned in reviews predicted as negative.

---

## Business Problem

ABC Restaurant receives customer reviews on Facebook. Manually reading each review makes it difficult to spot recurring complaints and decide which feedback needs follow-up first.

The objectives were to:

- Classify incoming reviews as positive or negative
- Identify common complaint themes in reviews flagged as negative
- Help the customer experience team prioritize follow-up and investigate recurring issues

---

## Dataset Overview

| Dataset | Reviews | Purpose |
| --- | ---: | --- |
| Historical reviews | 900 | Train and compare classifiers using 496 positive and 404 negative labeled reviews |
| Fresh reviews | 100 | Predict sentiment and summarize complaint themes in a new, unlabeled batch |

---

## Tools & Technologies

- **Python and Jupyter Notebook** for the analysis workflow
- **Pandas and NumPy** for handling review data
- **NLTK** for stop-word removal and Porter stemming
- **Scikit-learn** for feature extraction, model training, and evaluation
- **Matplotlib** for visualizing results
- **Joblib and Pickle** for saving the model and fitted vectorizer

---

## Analytical Approach

### 1. Text Preparation

I cleaned and lowercased review text, removed English stop words, and applied Porter stemming. I kept the word `not` because it can reverse the meaning of a phrase.

### 2. Feature Engineering

I converted the cleaned reviews into bag-of-words features using `CountVectorizer` with a maximum of 1,420 features. I used a stratified 80/20 split and fitted the vectorizer on the training reviews only. The same vocabulary was then used for the evaluation split and the fresh reviews.

### 3. Model Comparison

I compared four classifiers on the same 180-review evaluation split and ranked them by positive-class F1 score.

| Model | Accuracy | Precision | Recall | F1 score |
| --- | ---: | ---: | ---: | ---: |
| **Linear SVM** | **81.67%** | **82.35%** | **84.85%** | **0.836** |
| Multinomial Naive Bayes | 78.33% | 76.79% | 86.87% | 0.815 |
| Logistic Regression | 78.89% | 80.20% | 81.82% | 0.810 |
| Random Forest | 80.56% | 88.10% | 74.75% | 0.809 |

### 4. Final Model

Linear SVM had the highest F1 score. It correctly identified 63 of 81 negative reviews and 84 of 99 positive reviews on the split used for model comparison. It missed 18 negative reviews, which matters if the restaurant uses predictions to prioritize customer callbacks.

---

## Key Business Findings

### The Model Flagged 68 of 100 New Reviews as Negative

| Predicted sentiment | Reviews | Share |
| --- | ---: | ---: |
| Negative | 68 | 68% |
| Positive | 32 | 32% |

These are model predictions for the unlabeled batch. The exported CSV contains the review text and predicted label so individual cases can be checked.

### Food Quality Was the Most Frequently Tagged Complaint Theme

I used keyword matching to tag recurring issues in the 68 reviews predicted as negative.

| Complaint theme | Reviews mentioning it | Share of flagged reviews |
| --- | ---: | ---: |
| Food quality | 17 | 25.0% |
| Service | 7 | 10.3% |
| Waiting time | 6 | 8.8% |
| Price or value | 4 | 5.9% |
| Ambience | 2 | 2.9% |
| Drinks | 1 | 1.5% |
| Cleanliness | 1 | 1.5% |

A review can match more than one theme. These counts show keyword mentions and give the team a starting point for reading the underlying reviews.

---

## Business Recommendations

1. **Review food-quality feedback first.** Read the 17 tagged reviews to see whether customers are describing the same dishes or preparation issues.
2. **Use sentiment scores to prioritize callbacks.** Let the team verify flagged reviews before contacting customers, and periodically sample reviews predicted as positive to catch missed complaints.
3. **Track service and waiting-time feedback across batches.** Re-run the analysis after operational changes to see whether these themes become more or less common.

---

## Project Workflow

Historical reviews → Text cleaning → Bag-of-words features → Model comparison → Linear SVM selection

Fresh reviews → Sentiment prediction → Complaint tagging → Customer follow-up priorities

---

## Project Files

| File | Description |
| --- | --- |
| [Training notebook](sentiment_model_training.ipynb) | Text preparation, model comparison, and evaluation |
| [Prediction notebook](fresh_reviews_prediction.ipynb) | Batch prediction and complaint-theme analysis |
| [Historical reviews](data/historical_reviews.tsv) | Labeled reviews used for model development |
| [Fresh reviews](data/fresh_reviews.tsv) | Unlabeled reviews used for batch prediction |
| [Prediction results](results/predicted_reviews.csv) | Fresh reviews with predicted sentiment labels |
| `sentiment_analysis_model` and `sentiment_analysis_bow` | Saved classifier and fitted text vectorizer |

---

## Conclusion

The project turns a batch of unstructured reviews into a list the restaurant can use to prioritize customer follow-up. Linear SVM gave the strongest balance of precision and recall among the models compared, and food quality emerged as the first complaint theme to investigate in the new batch.

## Author

**Renu Nadar**  
MSc Statistics
