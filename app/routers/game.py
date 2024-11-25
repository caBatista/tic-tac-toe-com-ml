from flask_restx import Namespace, Resource, fields
from flask import jsonify
from pydantic import ValidationError
from models.neural_network import NeuralNetwork
from utils.training import initialize_training, step_training
from models.game_state import GameState
from models.board import Board
from utils.board_checker import BoardChecker
from algorithm.minimax import Minimax

api = Namespace('game', description='Operações com o jogo da velha')

checker = BoardChecker()
minimax = Minimax()
trained_network = NeuralNetwork().load_from_csv('best_individual.csv')
untrained_network = NeuralNetwork()

minimax_model = api.model('Board', {
    'board': fields.List(fields.List(fields.String, required=True), required=True, description='Board data in JSON format', example=[
        ['x', 'b', 'b'],
        ['b', 'o', 'b'],
        ['b', 'b', 'x']
    ]),
    'difficulty': fields.String(required=True, description='Game`s difficulty', example='hard')
})

@api.route('/minimax')
class PlayMinimax(Resource):
    @api.expect(minimax_model)
    def post(self):
        args = api.payload
        board_data = args['board']
        difficulty = args['difficulty']

        try:
            board = Board(board=board_data)
        except ValidationError as e:
            return jsonify(e.errors()), 400

        status = checker.check_status(board.board)
        next_move = None
        used_minimax = False

        if(status == GameState.NOT_OVER):
            next_move, used_minimax = minimax.find_next_move(board.board, difficulty)

        return jsonify({'status': status.to_string(), 
                        'next_move': next_move,
                        'used_minimax': used_minimax})

network_model = api.model('Board', {
    'board': fields.List(fields.List(fields.String, required=True), required=True, description='Board data in JSON format', example=[
        ['x', 'b', 'b'],
        ['b', 'o', 'b'],
        ['b', 'b', 'x']
    ])
})

@api.route('/network')
class PlayNetwork(Resource):
    @api.expect(network_model)
    def post(self):
        args = api.payload
        board_data = args['board']

        try:
            board = Board(board=board_data)
        except ValidationError as e:
            return jsonify(e.errors()), 400

        status = checker.check_status(board.board)
        next_move = None

        if(status == GameState.NOT_OVER):
            next_move  = trained_network.find_next_move(board.board)

            row, col = next_move
            if(board.board[row][col] == 'b'):
                status = GameState.CORRUPTED

        return jsonify({
            'status': status.to_string(),
            'next_move': (int(next_move[0]), int(next_move[1]))
        })
    