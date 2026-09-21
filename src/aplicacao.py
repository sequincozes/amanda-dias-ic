"""Orquestra o experimento sem concentrar lógica no arquivo principal."""

import json
import math

import numpy as np
from sklearn.model_selection import train_test_split

from .avaliador import AvaliadorModelo
from .carregador_dados import CarregadorDados
from .classificador import ClassificadorXGBoost
from .codificador_alvo import CodificadorAlvo
from .configuracao import Configuracao
from .excecoes import ErroValidacaoDataset
from .explicador_shap import ExplicadorSHAP
from .persistencia import RepositorioModelo
from .preditor import Preditor
from .preprocessamento import PreprocessadorERENO


class AplicacaoERENO:
    """Executa carregar → preparar → treinar → avaliar → explicar."""

    def __init__(self, configuracao: Configuracao) -> None:
        self.configuracao = configuracao

    def executar(self) -> None:
        cfg = self.configuracao
        print("1/6 — Carregando e verificando o dataset...")
        carregador = CarregadorDados(cfg.coluna_alvo)
        dados = carregador.carregar(cfg.caminho_dados)
        print(carregador.criar_resumo(dados))

        print("\n2/6 — Codificando os rótulos e separando treino/teste...")
        codificador = CodificadorAlvo(cfg.modo_alvo, cfg.rotulo_normal)
        alvo_codificado = codificador.ajustar_transformar(dados.alvo)
        treino, teste = self._separar_indices(alvo_codificado)

        x_treino = dados.atributos.iloc[treino].copy()
        x_teste = dados.atributos.iloc[teste].copy()
        y_treino = alvo_codificado[treino]
        y_teste = alvo_codificado[teste]

        print("3/6 — Ajustando os encoders somente com os dados de treino...")
        preprocessador = PreprocessadorERENO(cfg.colunas_descartadas)
        x_treino_transformado = preprocessador.ajustar_transformar(x_treino)
        x_teste_transformado = preprocessador.transformar(x_teste)

        print(
            f"Atributos após encoding: "
            f"{x_treino_transformado.shape[1]}"
        )
        print(f"Colunas de tempo descartadas: {preprocessador.colunas_descartadas_}")
        print(f"Colunas constantes descartadas: {preprocessador.colunas_constantes_}")

        print("\n4/6 — Treinando o XGBoost...")
        classificador = ClassificadorXGBoost(
            semente=cfg.semente,
            balancear_classes=cfg.balancear_classes,
        )
        modelo = classificador.ajustar(x_treino_transformado, y_treino)

        print("5/6 — Classificando o conjunto de teste e avaliando...")
        previstos = classificador.prever(x_teste_transformado).astype(int)
        probabilidades = classificador.prever_probabilidades(
            x_teste_transformado
        )

        diretorio_metricas = cfg.diretorio_saida / "avaliacao"
        avaliador = AvaliadorModelo()
        avaliador.avaliar(
            y_teste,
            previstos,
            probabilidades,
            codificador.nomes_classes_,
            diretorio_metricas,
        )

        preditor = Preditor()
        preditor.gerar_tabela(
            x_teste.index,
            y_teste,
            previstos,
            probabilidades,
            codificador,
            cfg.diretorio_saida / "previsoes_teste.csv",
        )

        print("6/6 — Gerando explicações e gráficos SHAP...")
        explicador = ExplicadorSHAP(
            max_amostras=cfg.max_amostras_shap,
            max_atributos=cfg.max_atributos_grafico,
            semente=cfg.semente,
        )
        metadados_shap = explicador.gerar_graficos(
            modelo,
            x_teste_transformado,
            preprocessador.nomes_atributos_saida(),
            codificador.nomes_classes_,
            cfg.diretorio_saida / "shap",
            indice_local=cfg.indice_explicacao_local,
        )

        RepositorioModelo().salvar(
            modelo,
            preprocessador,
            codificador,
            cfg.diretorio_saida / "modelo",
        )
        self._salvar_resumo(
            dados.atributos.shape,
            x_treino_transformado.shape,
            x_teste_transformado.shape,
            codificador.nomes_classes_,
            metadados_shap,
        )
        print(f"\nConcluído. Resultados salvos em: {cfg.diretorio_saida.resolve()}")

    def _separar_indices(
        self,
        alvo_codificado: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.configuracao
        quantidade = len(alvo_codificado)
        quantidade_classes = len(np.unique(alvo_codificado))

        if not 0 < cfg.proporcao_teste < 1:
            raise ErroValidacaoDataset(
                "A proporção de teste deve estar entre 0 e 1."
            )

        # Garante ao menos um exemplo de cada classe no teste.
        tamanho_teste = max(
            quantidade_classes,
            math.ceil(quantidade * cfg.proporcao_teste),
        )
        if quantidade - tamanho_teste < quantidade_classes:
            raise ErroValidacaoDataset(
                "Há poucos registros para manter todas as classes tanto no "
                "treino quanto no teste. Adicione mais exemplos ao dataset."
            )

        indices = np.arange(quantidade)
        treino, teste = train_test_split(
            indices,
            test_size=tamanho_teste,
            random_state=cfg.semente,
            stratify=alvo_codificado,
        )
        return treino, teste

    def _salvar_resumo(
        self,
        formato_original: tuple[int, int],
        formato_treino: tuple[int, int],
        formato_teste: tuple[int, int],
        classes: list[str],
        metadados_shap: dict[str, object],
    ) -> None:
        resumo = {
            "dataset": str(self.configuracao.caminho_dados),
            "formato_original": list(formato_original),
            "formato_treino_transformado": list(formato_treino),
            "formato_teste_transformado": list(formato_teste),
            "classes": classes,
            "shap": metadados_shap,
        }
        self.configuracao.diretorio_saida.mkdir(parents=True, exist_ok=True)
        caminho = self.configuracao.diretorio_saida / "resumo_execucao.json"
        caminho.write_text(
            json.dumps(resumo, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
