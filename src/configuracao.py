"""Configurações centralizadas do projeto."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class Configuracao:
    """Reúne os parâmetros usados pelas demais classes.

    Manter as opções em um único lugar facilita os primeiros experimentos:
    altere a configuração, sem precisar modificar o código do modelo.
    """

    caminho_dados: Path
    diretorio_saida: Path = Path("resultados")
    coluna_alvo: str = "class"
    modo_alvo: Literal["binario", "multiclasse"] = "binario"
    rotulo_normal: str = "normal"
    proporcao_teste: float = 0.20
    semente: int = 42
    balancear_classes: bool = True
    max_amostras_shap: int = 300
    max_atributos_grafico: int = 15
    indice_explicacao_local: int = 0

    # Tempos absolutos podem revelar a ordem/cenário da captura e causar
    # vazamento de informação. As diferenças temporais permanecem no modelo.
    colunas_descartadas: tuple[str, ...] = (
        "Time",
        "GooseTimestamp",
        "t",
    )
