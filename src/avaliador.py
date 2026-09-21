"""Métricas e matriz de confusão do conjunto de teste."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


class AvaliadorModelo:
    """Avalia previsões e grava resultados legíveis em arquivos."""

    def avaliar(
        self,
        alvo_real: np.ndarray,
        alvo_previsto: np.ndarray,
        probabilidades: np.ndarray,
        nomes_classes: list[str],
        diretorio_saida: Path,
    ) -> dict:
        diretorio_saida.mkdir(parents=True, exist_ok=True)
        rotulos = list(range(len(nomes_classes)))

        relatorio_texto = classification_report(
            alvo_real,
            alvo_previsto,
            labels=rotulos,
            target_names=nomes_classes,
            digits=4,
            zero_division=0,
        )
        relatorio_dict = classification_report(
            alvo_real,
            alvo_previsto,
            labels=rotulos,
            target_names=nomes_classes,
            output_dict=True,
            zero_division=0,
        )

        metricas: dict[str, object] = {"relatorio": relatorio_dict}
        if len(nomes_classes) == 2 and len(np.unique(alvo_real)) == 2:
            metricas["roc_auc"] = float(
                roc_auc_score(alvo_real, probabilidades[:, 1])
            )

        (diretorio_saida / "relatorio_classificacao.txt").write_text(
            relatorio_texto,
            encoding="utf-8",
        )
        (diretorio_saida / "metricas.json").write_text(
            json.dumps(metricas, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._salvar_matriz_confusao(
            alvo_real,
            alvo_previsto,
            nomes_classes,
            diretorio_saida / "matriz_confusao.png",
        )
        return metricas

    @staticmethod
    def _salvar_matriz_confusao(
        alvo_real: np.ndarray,
        alvo_previsto: np.ndarray,
        nomes_classes: list[str],
        caminho: Path,
    ) -> None:
        matriz = confusion_matrix(
            alvo_real,
            alvo_previsto,
            labels=list(range(len(nomes_classes))),
        )
        figura, eixo = plt.subplots(figsize=(7, 6))
        exibicao = ConfusionMatrixDisplay(
            confusion_matrix=matriz,
            display_labels=nomes_classes,
        )
        exibicao.plot(ax=eixo, cmap="Blues", colorbar=False, values_format="d")
        eixo.set_title("Matriz de confusão — conjunto de teste")
        figura.tight_layout()
        figura.savefig(caminho, dpi=180, bbox_inches="tight")
        plt.close(figura)
