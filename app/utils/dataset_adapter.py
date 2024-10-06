import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo

from models.game_state import GameState

class DatasetAdapter:
    def __init__(self):
        self.tic_tac_toe_endgame = fetch_ucirepo(id=101)

        self.features = self.tic_tac_toe_endgame.data['features']
        self.target = self.tic_tac_toe_endgame.data['targets']

        # Combinar e balancear amostras
        data = pd.concat([self.features, self.target], axis=1)
        positive_samples = data[data['class'] == 'positive']
        negative_samples = data[data['class'] == 'negative']

        n_samples = min(len(positive_samples), len(negative_samples), 400)
        positive_samples = positive_samples.sample(n=n_samples, random_state=42)
        negative_samples = negative_samples.sample(n=n_samples, random_state=42)

        combined_data = pd.concat([positive_samples, negative_samples])

        self.features = combined_data.drop(columns=['class'])
        self.target = combined_data['class']

        self.features = self.features.apply(LabelEncoder().fit_transform)
        self.target = self.target.apply(lambda x: self.map_class(x, self.features.loc[self.target.index[self.target == x]].iloc[0]))

        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42
        )

    def map_class(self, state, row):
        if state == 'positive':
            return GameState.X_WON
        elif state == 'negative':
            return self.determine_negative_state(row)
        else:
            return GameState.NOT_OVER

    def determine_negative_state(self, row):
        board = row.values.reshape(3, 3)
        if self.check_winner(board, 'O'):
            return GameState.O_WON
        elif self.check_winner(board, 'X'):
            return GameState.X_WON
        elif self.check_draw(board):
            return GameState.DRAW
        else:
            return GameState.NOT_OVER

    def check_winner(self, board, player):
        # Verificar linhas e colunas
        for i in range(3):
            if all(board[i][j] == player for j in range(3)):
                return True
            if all(board[j][i] == player for j in range(3)):
                return True

        # Verificar diagonais
        if all(board[i][i] == player for i in range(3)):
            return True
        if all(board[i][2-i] == player for i in range(3)):
            return True

        return False

    def check_draw(self, board):
        # Verifica se todas as células estão preenchidas sem um vencedor
        return all(cell != ' ' for row in board for cell in row) and not self.check_winner(board, 'X') and not self.check_winner(board, 'O')

    def check_status(self, board, model):
        board_flat = np.array(board).flatten()
        board_encoded = LabelEncoder().fit_transform(board_flat).reshape(1, -1)

        prediction = model.predict(board_encoded)[0]

        return GameState(prediction)

    def evaluate_model(self, model):
        model.fit(self.X_train, self.y_train)
        predicao = model.predict(self.X_val)
        
        accuracy = accuracy_score(self.y_val, predicao)
        precision = precision_score(self.y_val, predicao, average='weighted')
        recall = recall_score(self.y_val, predicao, average='weighted')
        f1 = f1_score(self.y_val, predicao, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def evaluate_all_models(self):
        knn  = KNeighborsClassifier(n_neighbors=3)
        mlp  = MLPClassifier(random_state=42)
        tree = DecisionTreeClassifier(random_state=42)

        knn_metrics  = self.evaluate_model(knn)
        mlp_metrics  = self.evaluate_model(mlp)
        tree_metrics = self.evaluate_model(tree)

        return {
            'knn': knn_metrics,
            'mlp': mlp_metrics,
            'decision_tree': tree_metrics
        }
    
    def get_best_model(self):
        knn = KNeighborsClassifier(n_neighbors=3)
        mlp = MLPClassifier(random_state=42)
        tree = DecisionTreeClassifier(random_state=42)

        models = [knn, mlp, tree]

        best_model = None
        best_accuracy = 0

        for model in models:
            metrics = self.evaluate_model(model)
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_model = model

        return best_model