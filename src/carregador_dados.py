"""Leitura, limpeza básica e descrição do CSV do ERENO Framework."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .excecoes import ErroValidacaoDataset


@dataclass(frozen=True)
class DadosDataset:
    """Dados separados entre atributos (X) e alvo (y)."""

    atributos: pd.DataFrame
    alvo: pd.Series
    codificacao_arquivo: str


class CarregadorDados:
    """Carrega o CSV e corrige detalhes comuns dos arquivos do ERENO.

    O arquivo anexado está em UTF-16 e alguns campos textuais contêm espaços.
    A classe tenta codificações conhecidas, remove esses espaços e converte para
    número apenas as colunas cujos valores são realmente numéricos.
    """

    CODIFICACOES = ("utf-16", "utf-8-sig", "utf-8", "latin-1")

    def __init__(self, coluna_alvo: str = "class") -> None:
        self.coluna_alvo = coluna_alvo

    def carregar(self, caminho: Path) -> DadosDataset:
        caminho = Path(caminho)
        if not caminho.exists():
            raise ErroValidacaoDataset(f"Arquivo não encontrado: {caminho}")

        tabela, codificacao = self._ler_csv(caminho)
        tabela = self._limpar(tabela)
        self._validar_estrutura(tabela)

        atributos = tabela.drop(columns=[self.coluna_alvo])
        alvo = tabela[self.coluna_alvo].copy()
        return DadosDataset(atributos, alvo, codificacao)

    def _ler_csv(self, caminho: Path) -> tuple[pd.DataFrame, str]:
        erros: list[str] = []
        for codificacao in self.CODIFICACOES:
            try:
                tabela = pd.read_csv(
                    caminho,
                    encoding=codificacao,
                    skipinitialspace=True,
                )
                return tabela, codificacao
            except (UnicodeError, pd.errors.ParserError) as erro:
                erros.append(f"{codificacao}: {erro}")

        detalhes = "\n".join(erros)
        raise ErroValidacaoDataset(
            "Não foi possível ler o CSV com as codificações esperadas.\n"
            f"Detalhes:\n{detalhes}"
        )

    def _limpar(self, tabela: pd.DataFrame) -> pd.DataFrame:
        tabela = tabela.copy()
        tabela.columns = [str(coluna).strip() for coluna in tabela.columns]

        if tabela.columns.duplicated().any():
            repetidas = tabela.columns[tabela.columns.duplicated()].tolist()
            raise ErroValidacaoDataset(
                f"Há nomes de colunas repetidos no CSV: {repetidas}"
            )

        for coluna in tabela.select_dtypes(include=["object", "string"]).columns:
            texto = tabela[coluna].astype("string").str.strip()
            texto = texto.replace("", pd.NA)

            # Converte somente quando todos os valores preenchidos são números.
            convertido = pd.to_numeric(texto, errors="coerce")
            preenchidos = texto.notna()
            if preenchidos.any() and convertido[preenchidos].notna().all():
                tabela[coluna] = convertido
            else:
                tabela[coluna] = texto

        # Valores infinitos não podem ser usados diretamente pelo imputador.
        tabela = tabela.replace([np.inf, -np.inf], np.nan)
        return tabela

    def _validar_estrutura(self, tabela: pd.DataFrame) -> None:
        if tabela.empty:
            raise ErroValidacaoDataset("O CSV não contém registros.")
        if self.coluna_alvo not in tabela.columns:
            raise ErroValidacaoDataset(
                f"A coluna-alvo '{self.coluna_alvo}' não existe. "
                f"Colunas disponíveis: {list(tabela.columns)}"
            )
        if tabela[self.coluna_alvo].isna().any():
            quantidade = int(tabela[self.coluna_alvo].isna().sum())
            raise ErroValidacaoDataset(
                f"A coluna-alvo possui {quantidade} valor(es) ausente(s)."
            )

    @staticmethod
    def criar_resumo(dados: DadosDataset) -> str:
        numericas = dados.atributos.select_dtypes(include=[np.number]).columns
        categoricas = dados.atributos.columns.difference(numericas)
        contagem = (
            dados.alvo.astype("string").str.strip().value_counts().to_dict()
        )
        return (
            f"Registros: {len(dados.atributos)}\n"
            f"Atributos: {dados.atributos.shape[1]}\n"
            f"Atributos numéricos: {len(numericas)}\n"
            f"Atributos categóricos: {len(categoricas)}\n"
            f"Classes originais: {contagem}\n"
            f"Codificação detectada: {dados.codificacao_arquivo}"
        )
