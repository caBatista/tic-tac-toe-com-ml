import multiprocessing
import matplotlib.pyplot as plt

def visualizar_evolucao(aptidoes_geracoes):
    plt.figure(figsize=(10, 6))
    plt.plot(aptidoes_geracoes, label='Melhor Aptidão')
    plt.title("Evolução da Aptidão")
    plt.xlabel("Gerações")
    plt.ylabel("Aptidão")
    plt.legend()
    plt.grid()
    plt.show()

def calcular_aptidao_paralelo(args):
    individuo, ag, minimax, board_checker = args
    return ag.calcular_aptidao(individuo, minimax, board_checker)

def treinar_rede(rede, ag, minimax, board_checker, geracoes):
    melhores_aptidoes = []

    pool = multiprocessing.Pool()

    for geracao in range(geracoes):
        resultados = pool.map(calcular_aptidao_paralelo, [(individuo, ag, minimax, board_checker) for individuo in ag.populacao])
        for i, aptidao in enumerate(resultados):
            ag.populacao[i]['aptidao'] = aptidao

        ag.populacao.sort(key=lambda x: x['aptidao'], reverse=True)

        nova_populacao = [ag.populacao[0]]
        while len(nova_populacao) < ag.tamanho_populacao:
            pai1 = ag.selecionar_pais()
            pai2 = ag.selecionar_pais()
            filho = ag.cruzamento(pai1, pai2)
            nova_populacao.append(ag.mutacao(filho))

        ag.populacao = nova_populacao

        melhor_aptidao = ag.populacao[0]['aptidao']
        melhores_aptidoes.append(melhor_aptidao)

        print(f"Geração {geracao}: Melhor aptidão = {melhor_aptidao}")
        ag.salvar_pesos(ag.populacao[0], geracao)

    pool.close()
    pool.join()

    visualizar_evolucao(melhores_aptidoes)