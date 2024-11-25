from flask_restx import Namespace, Resource
from flask import jsonify
from models.neural_network import NeuralNetwork
from utils.training import initialize_training, step_training

api = Namespace('training', description='Treinamento da Rede Neural')

trained_network = NeuralNetwork().load_from_csv('best_individual.csv')

@api.route('/start')
class StartTraining(Resource):
    def post(self):
        initialize_training()
        return jsonify({'message': 'Training initialized'})

@api.route('/step')
class StepTraining(Resource):
    def get(self):
        state = step_training()
        return jsonify(state)