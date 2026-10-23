from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Reuse the data loading and split from train.py without re-training
src = open("train.py").read()
marker = "X_train, X_test, y_train, y_test = train_test_split"
cut = src.index(marker)
end = src.index("\n", cut)
exec(src[:end], globals())

params = {"objective": "binary:logistic", "max_depth": 5, "learning_rate": 0.1,
          "n_estimators": 100, "eval_metric": "logloss"}
models = {
    "Majority class": DummyClassifier(strategy="most_frequent"),
    "Logistic regression": LogisticRegression(max_iter=3000),
    "XGBoost": xgb.XGBClassifier(**params),
}

print(f"\nChurn rate in test set: {float(y_test.mean()):.3f}")
print(f"{'Model':<22}{'Acc':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}")
for name, m in models.items():
    p = m.fit(X_train, y_train).predict(X_test)
    print(f"{name:<22}{accuracy_score(y_test, p):>8.4f}"
          f"{precision_score(y_test, p, zero_division=0):>8.4f}"
          f"{recall_score(y_test, p, zero_division=0):>8.4f}"
          f"{f1_score(y_test, p, zero_division=0):>8.4f}")
