import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

class MLModels:
    def __init__(self, model_type="logreg"):
        self.model_type = model_type
        if model_type == "logreg":
            base = LogisticRegression(max_iter=200, class_weight="balanced")
            self.model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        elif model_type == "xgboost":
            base = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=1.0, n_jobs=4
            )
            self.model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        else:
            raise ValueError("unknown model_type")
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    def predict_proba(self, X):
        return self.model.predict_proba(X)[:,1]
