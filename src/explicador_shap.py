"""Explicações globais e locais com SHAP."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import sparse


class ExplicadorSHAP:
    """Gera quatro gráficos centrais para estudar explicabilidade.

    Os gráficos são produzidos sobre os atributos já transformados pelo mesmo
    preprocessador usado no treinamento.
    """

    def __init__(
        self,
        max_amostras: int = 300,
        max_atributos: int = 15,
        semente: int = 42,
    ) -> None:
        self.max_amostras = max_amostras
        self.max_atributos = max_atributos
        self.semente = semente

    def gerar_graficos(
        self,
        modelo,
        atributos_transformados,
        nomes_atributos: list[str],
        nomes_classes: list[str],
        diretorio_saida: Path,
        indice_local: int = 0,
    ) -> dict[str, object]:
        diretorio_saida.mkdir(parents=True, exist_ok=True)
        dados = self._preparar_amostra(
            atributos_transformados,
            nomes_atributos,
        )

        explicador = shap.TreeExplainer(modelo)
        explicacoes = explicador(dados)
        explicacoes_classe, indice_classe = self._selecionar_classe(
            explicacoes,
            nomes_classes,
        )
        nome_classe = nomes_classes[indice_classe]

        self._salvar_beeswarm(
            explicacoes_classe,
            diretorio_saida / "shap_beeswarm.png",
            nome_classe,
        )
        self._salvar_barra(
            explicacoes_classe,
            diretorio_saida / "shap_importancia_global.png",
            nome_classe,
        )

        indice_local = max(0, min(indice_local, len(dados) - 1))
        self._salvar_waterfall(
            explicacoes_classe[indice_local],
            diretorio_saida / "shap_waterfall_local.png",
            nome_classe,
        )

        media_absoluta = np.abs(explicacoes_classe.values).mean(axis=0)
        indice_top = int(np.argmax(media_absoluta))
        atributo_top = nomes_atributos[indice_top]
        self._salvar_dependencia(
            explicacoes_classe[:, indice_top],
            diretorio_saida / "shap_dependencia_top_atributo.png",
            atributo_top,
        )

        metadados = {
            "classe_explicada": nome_classe,
            "indice_classe": indice_classe,
            "quantidade_amostras": len(dados),
            "atributo_mais_importante": atributo_top,
            "observacao": (
                "Para o XGBoost, os valores SHAP padrão explicam a saída "
                "interna do modelo (margem), não diretamente a porcentagem."
            ),
        }
        (diretorio_saida / "metadados_shap.json").write_text(
            json.dumps(metadados, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return metadados

    def _preparar_amostra(
        self,
        atributos,
        nomes_atributos: list[str],
    ) -> pd.DataFrame:
        quantidade = atributos.shape[0]
        if quantidade == 0:
            raise ValueError("Não há amostras para explicar.")

        if quantidade > self.max_amostras:
            gerador = np.random.default_rng(self.semente)
            indices = np.sort(
                gerador.choice(
                    quantidade,
                    size=self.max_amostras,
                    replace=False,
                )
            )
            atributos = atributos[indices]

        if sparse.issparse(atributos):
            atributos = atributos.toarray()
        return pd.DataFrame(atributos, columns=nomes_atributos)

    @staticmethod
    def _selecionar_classe(explicacoes, nomes_classes: list[str]):
        # Em problemas multiclasse, o SHAP adiciona uma dimensão de classes.
        if explicacoes.values.ndim == 3:
            indice = 1 if len(nomes_classes) == 2 else 0
            return explicacoes[:, :, indice], indice

        # No binário, o TreeExplainer geralmente retorna uma matriz 2D que
        # representa a classe positiva (intrusão).
        indice = 1 if len(nomes_classes) == 2 else 0
        return explicacoes, indice

    def _salvar_beeswarm(self, explicacoes, caminho: Path, classe: str) -> None:
        plt.close("all")
        shap.plots.beeswarm(
            explicacoes,
            max_display=self.max_atributos,
            show=False,
        )
        plt.title(f"SHAP beeswarm — classe {classe}")
        plt.tight_layout()
        plt.savefig(caminho, dpi=180, bbox_inches="tight")
        plt.close("all")

    def _salvar_barra(self, explicacoes, caminho: Path, classe: str) -> None:
        plt.close("all")
        shap.plots.bar(
            explicacoes,
            max_display=self.max_atributos,
            show=False,
        )
        plt.title(f"Importância global SHAP — classe {classe}")
        plt.tight_layout()
        plt.savefig(caminho, dpi=180, bbox_inches="tight")
        plt.close("all")

    def _salvar_waterfall(self, explicacao, caminho: Path, classe: str) -> None:
        plt.close("all")
        shap.plots.waterfall(
            explicacao,
            max_display=self.max_atributos,
            show=False,
        )
        plt.title(f"Explicação local SHAP — classe {classe}")
        plt.tight_layout()
        plt.savefig(caminho, dpi=180, bbox_inches="tight")
        plt.close("all")

    @staticmethod
    def _salvar_dependencia(explicacao_atributo, caminho: Path, atributo: str) -> None:
        plt.close("all")
        shap.plots.scatter(explicacao_atributo, show=False)
        plt.title(f"Dependência SHAP — {atributo}")
        plt.tight_layout()
        plt.savefig(caminho, dpi=180, bbox_inches="tight")
        plt.close("all")
