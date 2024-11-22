import random
import json

import numpy as np

from models.game_state import GameState
from utils.rede_neural import RedeNeural

class AlgoritmoGenetico:
    def __init__(self, tamanho_populacao, taxa_mutacao, taxa_cruzamento):
        self.tamanho_populacao = tamanho_populacao
        self.taxa_mutacao = taxa_mutacao
        self.taxa_cruzamento = taxa_cruzamento
        self.populacao = []

    def inicializar_populacao(self, rede):
        for _ in range(self.tamanho_populacao):
            individuo = {
                'pesos_entrada_oculta': np.random.uniform(-1, 1, rede.pesos_entrada_oculta.shape),
                'pesos_oculta_saida': np.random.uniform(-1, 1, rede.pesos_oculta_saida.shape),
                'bias_oculta': np.random.uniform(-1, 1, rede.bias_oculta.shape),
                'bias_saida': np.random.uniform(-1, 1, rede.bias_saida.shape),
                'aptidao': 0
            }
            self.populacao.append(individuo)

    def calcular_aptidao(self, individuo, minimax, board_checker):
        rede = RedeNeural()
        rede.pesos_entrada_oculta = individuo['pesos_entrada_oculta']
        rede.pesos_oculta_saida = individuo['pesos_oculta_saida']
        rede.bias_oculta = individuo['bias_oculta']
        rede.bias_saida = individuo['bias_saida']

        aptidao = 0
        for _ in range(10):
            board = [['b'] * 3 for _ in range(3)]
            jogador = 'x'
            while board_checker.check_status(board) == GameState.NOT_OVER:
                if jogador == 'x':
                    jogada = rede.jogada(board)
                    if jogada:
                        board[jogada[0]][jogada[1]] = 'x'
                    else:
                        aptidao -= 5
                        break
                else: 
                    jogada, _ = minimax.find_next_move(board, 'easy')
                    board[jogada[0]][jogada[1]] = 'o'

                jogador = 'o' if jogador == 'x' else 'x'

            estado_final = board_checker.check_status(board)
            if estado_final == GameState.X_WON:
                aptidao += 10
            elif estado_final == GameState.DRAW:
                aptidao += 1
            elif estado_final == GameState.O_WON:
                aptidao -= 10

        return aptidao

    def selecionar_pais(self):
        torneio = random.sample(self.populacao, 3)
        melhor = max(torneio, key=lambda x: x['aptidao'])
        return melhor

    def cruzamento(self, pai1, pai2):
        filho = {
            'pesos_entrada_oculta': (pai1['pesos_entrada_oculta'] + pai2['pesos_entrada_oculta']) / 2,
            'pesos_oculta_saida': (pai1['pesos_oculta_saida'] + pai2['pesos_oculta_saida']) / 2,
            'bias_oculta': (pai1['bias_oculta'] + pai2['bias_oculta']) / 2,
            'bias_saida': (pai1['bias_saida'] + pai2['bias_saida']) / 2,
            'aptidao': 0
        }
        return filho

    def mutacao(self, individuo):
        if np.random.rand() < self.taxa_mutacao:
            individuo['pesos_entrada_oculta'] += np.random.normal(0, 0.5, individuo['pesos_entrada_oculta'].shape)
            individuo['pesos_oculta_saida'] += np.random.normal(0, 0.5, individuo['pesos_oculta_saida'].shape)
            individuo['bias_oculta'] += np.random.normal(0, 0.5, individuo['bias_oculta'].shape)
            individuo['bias_saida'] += np.random.normal(0, 0.5, individuo['bias_saida'].shape)
        return individuo