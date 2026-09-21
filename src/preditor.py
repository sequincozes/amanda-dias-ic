"""Geração de uma tabela simples com classificações e probabilidades."""

from pathlib import Path

import numpy as np
import pandas as pd

from .codificador_alvo import CodificadorAlvo


class Preditor:
    """Converte as previsões numéricas em rótulos compreensíveis."""

    def gerar_tabela(
        self,
        indices_originais: pd.Index,
        alvo_real: np.ndarray,
        alvo_previsto: np.ndarray,
        probabilidades: np.ndarray,
        codificador_alvo: CodificadorAlvo,
        caminho_saida: Path,
    ) -> pd.DataFrame:
        tabela = pd.DataFrame(
            {
                "indice_original": indices_originais,
                "classe_real": codificador_alvo.inverter(alvo_real),
                "classe_prevista": codificador_alvo.inverter(alvo_previsto),
            }
        )

        for indice, nome in enumerate(codificador_alvo.nomes_classes_):
            nome_seguro = self._nome_seguro(nome)
            tabela[f"probabilidade_{nome_seguro}"] = probabilidades[:, indice]

        caminho_saida.parent.mkdir(parents=True, exist_ok=True)
        tabela.to_csv(caminho_saida, index=False, encoding="utf-8")
        return tabela

    @staticmethod
    def _nome_seguro(nome: str) -> str:
        return (
            nome.strip()
            .lower()
            .replace(" ", "_")
            .replace("ã", "a")
            .replace("á", "a")
            .replace("ç", "c")
        )
