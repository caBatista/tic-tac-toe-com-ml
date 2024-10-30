from flask import Flask
from flask_restx import Api, Resource
from routers.game import api as game_ns
from flask_cors import CORS

app = Flask(__name__)
api = Api(app, 
          title='Tic Tac Toe com ML',
          base_url='localhost:5000',
          description='Trabalho prático da disciplina de Inteligência Artificial.',
          doc='/docs') 
api.add_namespace(game_ns, path='/game')

CORS(app, resources={r"/": {"origins": "*"}})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)