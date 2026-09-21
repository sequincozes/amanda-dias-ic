"""Persistência do modelo e das transformações aprendidas."""

import json
from pathlib import Path

import joblib


class RepositorioModelo:
    """Salva um pacote reutilizável e uma cópia JSON do XGBoost."""

    def salvar(
        self,
        modelo,
        preprocessador,
        codificador_alvo,
        diretorio_saida: Path,
    ) -> None:
        diretorio_saida.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "modelo": modelo,
                "preprocessador": preprocessador,
                "codificador_alvo": codificador_alvo,
            },
            diretorio_saida / "pacote_modelo.joblib",
        )
        modelo.save_model(diretorio_saida / "modelo_xgboost.json")

        metadados = {
            "classes": codificador_alvo.nomes_classes_,
            "mapa_rotulos_originais": (
                codificador_alvo.mapa_rotulos_originais_
            ),
            "colunas_modelo": preprocessador.colunas_modelo_,
            "colunas_numericas": preprocessador.colunas_numericas_,
            "colunas_categoricas": preprocessador.colunas_categoricas_,
            "colunas_descartadas": preprocessador.colunas_descartadas_,
            "colunas_constantes_no_treino": (
                preprocessador.colunas_constantes_
            ),
        }
        (diretorio_saida / "metadados_modelo.json").write_text(
            json.dumps(metadados, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def carregar(caminho_pacote: Path) -> dict:
        """Carrega apenas pacotes confiáveis criados por este projeto."""

        return joblib.load(caminho_pacote)
