from utils.treinamento import treinar_rede
from utils.board_checker import BoardChecker
from utils.minimax import Minimax
from utils.rede_neural import RedeNeural
from utils.algoritmo_genetico import AlgoritmoGenetico

def main():
    board_checker = BoardChecker()
    minimax = Minimax()
    rede = RedeNeural()
    ag = AlgoritmoGenetico(tamanho_populacao=180, taxa_mutacao=0.1, taxa_cruzamento=0.7)

    ag.inicializar_populacao(rede)

    print("Iniciando treinamento...")
    treinar_rede(rede, ag, minimax, board_checker, geracoes=5000)

if __name__ == "__main__":
    main()
