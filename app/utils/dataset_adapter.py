from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

class DatasetAdapter:
    def __init__(self):
        self.tic_tac_toe_endgame = fetch_ucirepo(id=101)

        self.features = self.tic_tac_toe_endgame.data['features']
        self.target = self.tic_tac_toe_endgame.data['targets']

        # Reduzir o dataset para 400 linhas
        combined_data = pd.concat([self.features, self.target], axis=1)
        combined_data = combined_data.sample(n=400, random_state=42)
        
        self.features = combined_data.iloc[:, :-1]
        self.target = combined_data.iloc[:, -1]

        # Aplicar LabelEncoder antes do tratamento de outliers
        self.features = self.features.apply(LabelEncoder().fit_transform)
        self.target = LabelEncoder().fit_transform(self.target.values.ravel())

        # Tratar dados ruidosos usando Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels = iso_forest.fit_predict(self.features)
        
        # Remover outliers
        clean_indices = np.where(outlier_labels != -1)[0]
        self.features = self.features.iloc[clean_indices]
        self.target = self.target[clean_indices]

        # Divisão em treino, validação e teste
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42
        )

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
        knn = KNeighborsClassifier(n_neighbors=3)
        mlp = MLPClassifier(random_state=42)
        tree = DecisionTreeClassifier(random_state=42)

        knn_metrics = self.evaluate_model(knn)
        mlp_metrics = self.evaluate_model(mlp)
        tree_metrics = self.evaluate_model(tree)

        return {
            'knn': knn_metrics,
            'mlp': mlp_metrics,
            'decision_tree': tree_metrics
        }