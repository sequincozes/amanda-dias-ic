"""Pré-processamento de atributos numéricos e categóricos."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .excecoes import ErroValidacaoDataset


class PreprocessadorERENO:
    """Prepara as colunas para o XGBoost sem impor ordem às categorias.

    Números recebem imputação pela mediana. Categorias recebem imputação pelo
    valor mais frequente e One-Hot Encoding. Árvores não exigem padronização de
    escala, por isso StandardScaler não é usado.
    """

    def __init__(self, colunas_descartadas: Sequence[str] = ()) -> None:
        self.colunas_descartadas_config = tuple(colunas_descartadas)
        self.colunas_descartadas_: list[str] = []
        self.colunas_constantes_: list[str] = []
        self.colunas_modelo_: list[str] = []
        self.colunas_numericas_: list[str] = []
        self.colunas_categoricas_: list[str] = []
        self.transformador_: ColumnTransformer | None = None

    def ajustar_transformar(self, atributos: pd.DataFrame):
        preparados = self._preparar_ajuste(atributos)
        self.transformador_ = self._criar_transformador()
        return self.transformador_.fit_transform(preparados)

    def transformar(self, atributos: pd.DataFrame):
        if self.transformador_ is None:
            raise RuntimeError("O preprocessador ainda não foi ajustado.")

        ausentes = sorted(set(self.colunas_modelo_) - set(atributos.columns))
        if ausentes:
            raise ErroValidacaoDataset(
                "O novo conjunto não possui colunas usadas no treinamento: "
                f"{ausentes}"
            )
        preparados = atributos[self.colunas_modelo_].copy()
        preparados = self._normalizar_ausentes_categoricos(preparados)
        return self.transformador_.transform(preparados)

    def _preparar_ajuste(self, atributos: pd.DataFrame) -> pd.DataFrame:
        self.colunas_descartadas_ = [
            coluna
            for coluna in self.colunas_descartadas_config
            if coluna in atributos.columns
        ]
        preparados = atributos.drop(columns=self.colunas_descartadas_).copy()

        # Uma coluna constante não ajuda uma árvore a separar classes.
        self.colunas_constantes_ = [
            coluna
            for coluna in preparados.columns
            if preparados[coluna].nunique(dropna=False) <= 1
        ]
        preparados = preparados.drop(columns=self.colunas_constantes_)

        if preparados.shape[1] == 0:
            raise ErroValidacaoDataset(
                "Nenhum atributo variável restou após o pré-processamento."
            )

        self.colunas_modelo_ = preparados.columns.tolist()
        self.colunas_numericas_ = preparados.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        self.colunas_categoricas_ = [
            coluna
            for coluna in self.colunas_modelo_
            if coluna not in self.colunas_numericas_
        ]
        return self._normalizar_ausentes_categoricos(preparados)

    def _normalizar_ausentes_categoricos(
        self,
        atributos: pd.DataFrame,
    ) -> pd.DataFrame:
        """Representa None e pd.NA como np.nan antes da imputação."""

        atributos = atributos.copy()
        for coluna in self.colunas_categoricas_:
            atributos[coluna] = (
                atributos[coluna]
                .astype(object)
                .where(atributos[coluna].notna(), np.nan)
            )
        return atributos

    def _criar_transformador(self) -> ColumnTransformer:
        transformacoes: list[tuple[str, Pipeline, list[str]]] = []

        if self.colunas_numericas_:
            pipeline_numerico = Pipeline(
                steps=[("preencher_ausentes", SimpleImputer(strategy="median"))]
            )
            transformacoes.append(
                ("numerico", pipeline_numerico, self.colunas_numericas_)
            )

        if self.colunas_categoricas_:
            pipeline_categorico = Pipeline(
                steps=[
                    (
                        "preencher_ausentes",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "one_hot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=True,
                            dtype=np.float32,
                        ),
                    ),
                ]
            )
            transformacoes.append(
                ("categorico", pipeline_categorico, self.colunas_categoricas_)
            )

        return ColumnTransformer(
            transformers=transformacoes,
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def nomes_atributos_saida(self) -> list[str]:
        if self.transformador_ is None:
            raise RuntimeError("O preprocessador ainda não foi ajustado.")
        return self.transformador_.get_feature_names_out().astype(str).tolist()
