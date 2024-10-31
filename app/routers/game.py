from flask import jsonify
from pydantic import ValidationError
from models.game_state import GameState
from utils.minimax import Minimax
from models.board import Board
from utils.board_checker import BoardChecker
from flask_restx import Namespace, Resource, fields

api = Namespace('game', description='Operações com o jogo da velha')

checker = BoardChecker()
minimax = Minimax()

status_model = api.model('Board', {
    'board': fields.List(fields.List(fields.String, required=True), required=True, description='Board data in JSON format', example=[
        ['x', 'b', 'b'],
        ['b', 'o', 'b'],
        ['b', 'b', 'x']
    ]),
    'difficulty': fields.String(required=True, description='Game`s difficulty', example='hard')
})

@api.route('/status')
class Status(Resource):
    @api.expect(status_model)
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

        return jsonify({"status": status.to_string(), 
                        "next_move": next_move,
                        "used_minimax": used_minimax})