import os

import joblib
import numpy as np
from sklearn.exceptions import NotFittedError

MODEL_PATH = "model.joblib"


class KeystrokeModel:
    def __init__(self):
        self.mean_ = None
        self.max_dist_ = None

    def train(self, X: np.ndarray) -> None:
        """
        Treina um modelo simples baseado em distância ao centroide
        das amostras de treinamento.
        """
        X = np.asarray(X)
        self.mean_ = X.mean(axis=0)
        dists = np.linalg.norm(X - self.mean_, axis=1)
        self.max_dist_ = float(dists.max())

        joblib.dump({"mean": self.mean_, "max_dist": self.max_dist_}, MODEL_PATH)

    def load(self) -> None:
        if os.path.exists(MODEL_PATH):
            data = joblib.load(MODEL_PATH)
            self.mean_ = data.get("mean")
            self.max_dist_ = data.get("max_dist")

    def _check_fitted(self) -> None:
        if self.mean_ is None or self.max_dist_ is None:
            raise NotFittedError(
                "KeystrokeModel ainda não foi treinado. "
                "Chame train(X) antes de usar predict."
            )

    def predict(self, X):
        """
        Retorna 1 para amostras próximas do padrão treinado
        e -1 para amostras distantes.
        """
        self._check_fitted()

        X = np.asarray(X)
        dists = np.linalg.norm(X - self.mean_, axis=1)
        return np.where(dists <= self.max_dist_, 1, -1)
