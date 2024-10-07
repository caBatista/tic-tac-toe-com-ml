import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
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

        data = pd.concat([self.features, self.target], axis=1)

        positive_samples = data[data['class'] == 'positive']
        negative_samples = data[data['class'] == 'negative']

        n_samples = min(len(positive_samples), len(negative_samples), 400)
        positive_samples = positive_samples.sample(n=n_samples, random_state=42)
        negative_samples = negative_samples.sample(n=n_samples, random_state=42)

        combined_data = pd.concat([positive_samples, negative_samples])
        combined_data = combined_data.apply(lambda x: self.map_class(x), axis=1)

        self.features = combined_data.drop(columns=['class'])
        self.target = combined_data['class']

        le = LabelEncoder()
        le.fit(['x', 'o', 'b'])

        self.features = self.features.apply(le.fit_transform)
        self.target = self.target.apply(lambda x: x.to_string())  

        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)

        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42
        )

    def map_class(self, row):
        state = row['class']
        
        if state == 'positive':
            new_state = GameState.X_WON
            row['class'] = new_state
            return row
        else:
            row = row.drop('class')
            board = row.values.reshape(3, 3)

            new_state = self.check_tic_tac_toe_status(board)
            row['class'] = new_state

            return row   

    def check_tic_tac_toe_status(self, board):
        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2] != 'b':
                return GameState.X_WON if board[i][0] == 'x' else GameState.O_WON
            if board[0][i] == board[1][i] == board[2][i] != 'b':
                return GameState.X_WON if board[0][i] == 'x' else GameState.O_WON

        if board[0][0] == board[1][1] == board[2][2] != 'b':
            return GameState.X_WON if board[0][0] == 'x' else GameState.O_WON
        if board[0][2] == board[1][1] == board[2][0] != 'b':
            return GameState.X_WON if board[0][2] == 'x' else GameState.O_WON

        for row in board:
            if 'b' in row:
                return GameState.NOT_OVER
            
        return GameState.DRAW

    def check_status(self, board, model):
        board_flat = np.array(board).flatten()

        le = LabelEncoder()
        le.fit(['x', 'o', 'b'])
        board_encoded = le.transform(board_flat).reshape(1, -1)

        prediction = model.predict(board_encoded)[0]

        return GameState(prediction)

    def evaluate_model(self, model):
        model.fit(self.X_train, self.y_train)
        predicao = model.predict(self.X_val)
        
        accuracy = accuracy_score(self.y_val, predicao)
        precision = precision_score(self.y_val, predicao, average='weighted', zero_division=0)
        recall = recall_score(self.y_val, predicao, average='weighted', zero_division=0)
        f1 = f1_score(self.y_val, predicao, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def evaluate_all_models(self):
        knn  = KNeighborsClassifier(n_neighbors=3)
        mlp  = MLPClassifier(random_state=42, max_iter=1000)
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
        knn = KNeighborsClassifier(metric='manhattan')
        mlp = MLPClassifier(activation='tanh', max_iter=1000, random_state=42)
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