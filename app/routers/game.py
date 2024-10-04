from flask import Blueprint, request, jsonify
from pydantic import ValidationError
from models.board import Board
from utils.dataset_adapter import DatasetAdapter, GameState

game_bp = Blueprint('game', __name__)
dataset_adapter = DatasetAdapter()

@game_bp.route('/evaluate', methods=['GET'])
def evaluate():
    adapter = DatasetAdapter()
    metrics = adapter.evaluate_all_models()
    return jsonify(metrics)

@game_bp.route('/status', methods=['POST'])
def predict_game_status():
    try:
        board_data = request.json
        board = Board(**board_data)
    except ValidationError as e:
        return jsonify(e.errors()), 400

    best_model = dataset_adapter.get_best_model()
    status = dataset_adapter.check_status(board.board, best_model)

    return jsonify({"status": status.to_string(), "model_used": "none"})