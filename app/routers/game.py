from flask import Flask, request, jsonify
from utils.dataset_adapter import DatasetAdapter

app = Flask(__name__)

@app.route('/', methods=['GET'])
def root():
    board_json = request.json
    return board_json

@app.route('/evaluate', methods=['GET'])
def evaluate():
    adapter = DatasetAdapter()
    metrics = adapter.evaluate_all_models()
    return jsonify(metrics)

if __name__ == '__main__':
    app.run(debug=True)