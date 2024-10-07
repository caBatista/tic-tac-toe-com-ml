# Trabalho Prático de Inteligência Artificial

Este projeto tem como objetivo analisar o estado de um tabuleiro de jogo da velha.

## Endpoints

### EVALUATE
Este endpoint foi feito para demonstração dos testes realizados a cada requisição da API. Ele utiliza todos os algoritmos disponíveis e realiza testes de predição, retornando as métricas de cada um. Durante o jogo, este processo é realizado a cada operação, visando utilizar sempre o algoritmo mais adequado para a predição.

### STATUS
Este endpoint é utilizado pelo front-end para prever o status do jogo, utilizando o melhor algoritmo possível.

## Algoritmos Utilizados

- **KNN**
- **MLP** (Perceptron Multicamadas), definido com a seguinte topologia:
  - **Camada de Entrada (n)**: 9 neurônios (um para cada posição do tabuleiro 3x3).
  - **Camada Oculta (p)**: 100 neurônios por padrão.
  - **Camada de Saída (k)**: 4 neurônios, correspondentes aos possíveis estados:
    - X_WON
    - O_WON
    - DRAW
    - NOT_OVER
- **Decision Tree**
- **Random Forest**
  - O algoritmo de Random Forest utiliza múltiplas árvores de decisão para melhorar a precisão da predição e reduzir o risco de overfitting. Cada árvore é construída a partir de um subconjunto aleatório dos dados, e a predição final é baseada na média das predições de todas as árvores.

## Tecnologias Utilizadas

- **Python Flask**

## Autores
- Arthur Bonnazi
- Caio Batista
- Gustavo Rocha
- Rodrigo Renck
- Leonardo Ramos