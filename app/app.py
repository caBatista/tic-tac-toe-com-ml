from flask import Flask
from flask_restx import Api
from routers.game import api as game_ns
from routers.training import api as training_ns
from flask_cors import CORS

app = Flask(__name__)
api = Api(app, 
          title='Tic Tac Toe com ML',
          base_url='localhost:5000',
          description='Trabalho prático da disciplina de Inteligência Artificial.',
          doc='/docs') 
api.add_namespace(game_ns, path='/game')
api.add_namespace(training_ns, path='/training')

CORS(app, resources={r'/*': {'origins': '*'}})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)