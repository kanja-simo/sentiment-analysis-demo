\# Sentiment Analysis Tool



A text classification tool that predicts whether a piece of text expresses positive or negative sentiment, built as a foundational NLP project ahead of applying my skills to citizen/government communications analysis.



\## How it works



\- \*\*Dataset:\*\* 2,000 labeled movie reviews (NLTK's `movie\_reviews` corpus), balanced 1,000 positive / 1,000 negative.

\- \*\*Approach:\*\* Text is converted into numerical features using TF-IDF (Term Frequency–Inverse Document Frequency), then classified using a Logistic Regression model (scikit-learn).

\- \*\*Train/test split:\*\* 80% training (1,600 reviews), 20% held-out testing (400 reviews), evaluated only on data the model never saw during training.



\## Performance



Evaluated on the 400-review held-out test set:



| Metric | Score |

|---|---|

| Accuracy | 79.5% |

| Precision (avg) | 0.80 |

| Recall (avg) | 0.80 |

| F1-score (avg) | 0.79 |



Precision and recall are balanced across both classes, indicating the model isn't biased toward either label.



\## Known limitations



\- Performs more reliably on longer, review-style text (its training domain) than on short, colloquial sentences, where confidence is noticeably lower.

\- Trained only on English movie reviews — vocabulary and sentiment patterns may not generalize well to other domains (e.g. government/citizen communications) or languages without retraining on domain-specific data.



\## Planned improvements



\- Extend to Kiswahili text classification.

\- Experiment with n-grams (two-word phrases) to better capture short-text sentiment.

\- Retrain on domain-specific text (e.g. public service feedback) rather than movie reviews, for closer alignment with real-world government/citizen use cases.



\## Running locally



\\`\\`\\`

pip install streamlit joblib scikit-learn

streamlit run app.py

\\`\\`\\`



\## Tech stack



Python, scikit-learn, NLTK, Streamlit

