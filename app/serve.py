import os, sys, inspect

currentdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
parentdir = (currentdir)
sys.path.insert(0, parentdir) 

from config import Config

from flask import Flask, jsonify, request

from PIL import Image
from vision.model_factory import create_model


app = Flask(__name__)

model = create_model(
    type=Config.get("model_type"),
    dataset_name=Config.get("dataset_name"), 
    saved_model_filename=Config.get("saved_model_filename")
)


@app.route('/', methods = ['PUT'])
def detect_litter():
    print(request.files)
    file = request.files['image']
    img = Image.open(file.stream)

    result = model.evaluate(img, score_threshold=Config.get("score_threshold"))
    return jsonify({'result': result})


if __name__=='__main__':
    app.run(host=Config.get("host"), port=Config.get("port"))
