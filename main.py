"""Ponto de entrada do projeto.

Toda a lógica está organizada nas classes do pacote ``src``. Este arquivo
principal apenas inicia a interface de linha de comando.
"""

from src.cli import executar_cli


if __name__ == "__main__":
    executar_cli()
