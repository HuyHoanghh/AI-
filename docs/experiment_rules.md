# Experiment Rules

Shared settings:
- random_state = 42
- test_size = 0.20
- stratify = label
- max_features = 5000
- labels: ham=0, spam=1

Required comparisons:
1. Naive Bayes + CountVectorizer
2. Naive Bayes + TF-IDF
3. KNN + CountVectorizer
4. KNN + TF-IDF

Required metrics:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Training time
- Prediction time
