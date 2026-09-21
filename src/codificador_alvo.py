"""Codificação dos rótulos que o XGBoost deve prever."""

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .excecoes import ErroValidacaoDataset


class CodificadorAlvo:
    """Transforma os rótulos de texto em inteiros.

    - ``binario``: o rótulo normal vira 0; qualquer outro rótulo vira 1.
    - ``multiclasse``: cada tipo original recebe um inteiro pelo LabelEncoder.
    """

    def __init__(self, modo: str = "binario", rotulo_normal: str = "normal") -> None:
        if modo not in {"binario", "multiclasse"}:
            raise ValueError("O modo deve ser 'binario' ou 'multiclasse'.")
        self.modo = modo
        self.rotulo_normal = rotulo_normal
        self._label_encoder: LabelEncoder | None = None
        self.nomes_classes_: list[str] = []
        self.mapa_rotulos_originais_: dict[str, int] = {}

    def ajustar_transformar(self, alvo: Iterable[object]) -> np.ndarray:
        serie = pd.Series(alvo, dtype="string").str.strip()
        if serie.isna().any():
            raise ErroValidacaoDataset("Existem rótulos ausentes na coluna-alvo.")

        if self.modo == "binario":
            codificado = self._ajustar_binario(serie)
        else:
            codificado = self._ajustar_multiclasse(serie)

        self._validar_quantidade_classes(codificado, serie)
        return codificado

    def _ajustar_binario(self, serie: pd.Series) -> np.ndarray:
        normal_normalizado = self.rotulo_normal.strip().casefold()
        rotulos_normalizados = serie.str.casefold()

        if not (rotulos_normalizados == normal_normalizado).any():
            raise ErroValidacaoDataset(
                f"O rótulo normal '{self.rotulo_normal}' não foi encontrado. "
                "Informe o nome correto com --rotulo-normal."
            )

        codificado = np.where(
            rotulos_normalizados == normal_normalizado,
            0,
            1,
        ).astype(int)
        self.nomes_classes_ = [self.rotulo_normal, "intrusão"]
        self.mapa_rotulos_originais_ = {
            rotulo: (0 if rotulo.casefold() == normal_normalizado else 1)
            for rotulo in sorted(serie.unique().tolist())
        }
        return codificado

    def _ajustar_multiclasse(self, serie: pd.Series) -> np.ndarray:
        self._label_encoder = LabelEncoder()
        codificado = self._label_encoder.fit_transform(serie)
        self.nomes_classes_ = self._label_encoder.classes_.astype(str).tolist()
        self.mapa_rotulos_originais_ = {
            nome: indice for indice, nome in enumerate(self.nomes_classes_)
        }
        return codificado

    def _validar_quantidade_classes(
        self,
        codificado: np.ndarray,
        serie_original: pd.Series,
    ) -> None:
        classes, contagens = np.unique(codificado, return_counts=True)
        if len(classes) < 2:
            distribuicao = serie_original.value_counts().to_dict()
            raise ErroValidacaoDataset(
                "Não é possível treinar um classificador supervisionado com "
                "apenas uma classe. O arquivo fornecido contém somente: "
                f"{distribuicao}. Use uma versão do dataset ERENO que inclua "
                "registros normais e registros de ataque."
            )

        menores = {
            self.nomes_classes_[int(classe)]: int(contagem)
            for classe, contagem in zip(classes, contagens)
            if contagem < 2
        }
        if menores:
            raise ErroValidacaoDataset(
                "Cada classe precisa ter ao menos dois registros para a "
                f"separação estratificada. Classes insuficientes: {menores}"
            )

    def transformar(self, alvo: Iterable[object]) -> np.ndarray:
        serie = pd.Series(alvo, dtype="string").str.strip()
        if self.modo == "binario":
            normal = self.rotulo_normal.strip().casefold()
            return np.where(serie.str.casefold() == normal, 0, 1).astype(int)
        if self._label_encoder is None:
            raise RuntimeError("O codificador ainda não foi ajustado.")
        return self._label_encoder.transform(serie)

    def inverter(self, valores: Iterable[int]) -> np.ndarray:
        indices = np.asarray(list(valores), dtype=int)
        return np.asarray([self.nomes_classes_[indice] for indice in indices])
