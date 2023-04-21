from flask import Flask
from flask_cors import CORS
from app.services import createAndTrainModel

app = Flask(__name__)
CORS(app)
from app import routes
createAndTrainModel()