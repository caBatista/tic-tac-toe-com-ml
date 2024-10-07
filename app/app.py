from flask import Flask
from routers.game import game_bp

app = Flask(__name__)
app.register_blueprint(game_bp, url_prefix='/game')

if __name__ == '__main__':
    app.run(debug=True)