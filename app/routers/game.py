from flask import Blueprint, request, jsonify
from pydantic import ValidationError
from models.board import Board
from utils.dataset_adapter import DatasetAdapter, GameState
from flask_restx import Namespace, Resource, fields

dataset_adapter = DatasetAdapter()
api = Namespace('game', description='Operações com o jogo da velha')

@api.route('/evaluate')
class Evaluate(Resource):
    def get(self):
        adapter = DatasetAdapter()
        metrics = adapter.evaluate_all_models()
        return jsonify(metrics)

board_model = api.model('Board', {
    'board': fields.List(fields.List(fields.String, required=True), required=True, description='Board data in JSON format', example=[
        ['x', 'o', 'x'],
        ['o', 'x', 'o'],
        ['x', 'o', 'x']
    ])
})

@api.route('/status')
class Status(Resource):
    @api.expect(board_model)
    def post(self):
        args = api.payload
        board_data = args['board']

        try:
            board = Board(board=board_data)
        except ValidationError as e:
            return jsonify(e.errors()), 400

        best_model = dataset_adapter.get_best_model()
        
        status = dataset_adapter.check_tic_tac_toe_status(board.board)

        return jsonify({"status": status.to_string(), "model_used": best_model.__class__.__name__})