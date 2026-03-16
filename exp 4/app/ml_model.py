import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

class MLModel:

    def __init__(self):

        iris = load_iris()

        self.model = RandomForestClassifier()
        self.model.fit(iris.data, iris.target)

        self.classes = ["setosa","versicolor","virginica"]

    def predict(self,data):

        prediction = self.model.predict(data)[0]
        probability = self.model.predict_proba(data)[0]

        return {
            "prediction": int(prediction),
            "species": self.classes[prediction],
            "confidence": float(max(probability))
        }

ml_model = MLModel()