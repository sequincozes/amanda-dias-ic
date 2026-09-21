"""Interface de linha de comando mantida fora do arquivo principal."""

import argparse
import sys
from pathlib import Path

from .configuracao import Configuracao
from .excecoes import ErroValidacaoDataset


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Treina XGBoost e gera explicações SHAP para um CSV do ERENO."
        )
    )
    parser.add_argument(
        "--dados",
        type=Path,
        default=Path("data/top_50.csv"),
        help="Caminho do CSV (padrão: data/top_50.csv).",
    )
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("resultados"),
        help="Diretório dos resultados (padrão: resultados).",
    )
    parser.add_argument(
        "--alvo",
        default="class",
        help="Nome da coluna-alvo (padrão: class).",
    )
    parser.add_argument(
        "--modo-alvo",
        choices=["binario", "multiclasse"],
        default="binario",
        help="Binário agrupa todos os ataques como intrusão.",
    )
    parser.add_argument(
        "--rotulo-normal",
        default="normal",
        help="Rótulo que representa tráfego normal.",
    )
    parser.add_argument(
        "--proporcao-teste",
        type=float,
        default=0.20,
        help="Fração reservada para teste (padrão: 0.20).",
    )
    parser.add_argument(
        "--indice-explicacao",
        type=int,
        default=0,
        help="Amostra do teste usada no gráfico waterfall.",
    )
    return parser


def executar_cli() -> None:
    argumentos = criar_parser().parse_args()
    configuracao = Configuracao(
        caminho_dados=argumentos.dados,
        diretorio_saida=argumentos.saida,
        coluna_alvo=argumentos.alvo,
        modo_alvo=argumentos.modo_alvo,
        rotulo_normal=argumentos.rotulo_normal,
        proporcao_teste=argumentos.proporcao_teste,
        indice_explicacao_local=argumentos.indice_explicacao,
    )

    try:
        # Import tardio: se uma dependência faltar, a mensagem abaixo explica
        # como preparar o ambiente, em vez de mostrar um traceback longo.
        from .aplicacao import AplicacaoERENO

        AplicacaoERENO(configuracao).executar()
    except ErroValidacaoDataset as erro:
        print(f"\nDataset ainda não utilizável:\n{erro}", file=sys.stderr)
        raise SystemExit(2) from erro
    except ModuleNotFoundError as erro:
        print(
            "\nDependência não instalada: "
            f"{erro.name}. Execute: python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(3) from erro
