"""Modelo de classificação baseado em XGBoost."""

import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


class ClassificadorXGBoost:
    """Configura e treina o XGBoost com parâmetros fáceis de estudar."""

    def __init__(
        self,
        semente: int = 42,
        balancear_classes: bool = True,
    ) -> None:
        self.semente = semente
        self.balancear_classes = balancear_classes
        self.modelo: XGBClassifier | None = None

    def ajustar(self, atributos_treino, alvo_treino: np.ndarray) -> XGBClassifier:
        quantidade_classes = len(np.unique(alvo_treino))

        parametros = {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "min_child_weight": 1,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "random_state": self.semente,
            "n_jobs": -1,
        }

        if quantidade_classes == 2:
            parametros.update(
                objective="binary:logistic",
                eval_metric="logloss",
            )
        else:
            parametros.update(
                objective="multi:softprob",
                eval_metric="mlogloss",
                num_class=quantidade_classes,
            )

        self.modelo = XGBClassifier(**parametros)
        pesos = None
        if self.balancear_classes:
            pesos = compute_sample_weight(
                class_weight="balanced",
                y=alvo_treino,
            )

        self.modelo.fit(
            atributos_treino,
            alvo_treino,
            sample_weight=pesos,
            verbose=False,
        )
        return self.modelo

    def prever(self, atributos) -> np.ndarray:
        return self._obter_modelo().predict(atributos)

    def prever_probabilidades(self, atributos) -> np.ndarray:
        return self._obter_modelo().predict_proba(atributos)

    def _obter_modelo(self) -> XGBClassifier:
        if self.modelo is None:
            raise RuntimeError("O classificador ainda não foi treinado.")
        return self.modelo
