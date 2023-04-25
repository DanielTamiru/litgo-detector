from config import Config

from argparse import ArgumentParser

from PIL import Image
from vision.model_factory import create_model
from vision.draw import draw_boxes

parser = ArgumentParser(
    prog='Litgo Detector Runner',
    description="""
        Given a image filepath,, it will create a copy with boxes
        (and optionally labels and confidence scores)
        around all instances of litter in the original image
    """,
)

parser.add_argument('filepath')  
parser.add_argument('-c', '--color', default="red", help='color value of boxes: https://drafts.csswg.org/css-color-4/#named-colors')
parser.add_argument('-l', action='store_true', help='display litter label on boxes')
parser.add_argument('-s', action='store_true', help='display confidence score on boxes')

args = parser.parse_args()

model = create_model(
    type=Config.get("model_type"),
    dataset_name=Config.get("dataset_name"), 
    saved_model_filename=Config.get("saved_model_filename")
)

with Image.open(args.filepath) as img:
    result = model.evaluate(img, score_threshold=Config.get("score_threshold"))
    result_img = draw_boxes(img, result, color=args.color, draw_categories=args.l, draw_score=args.s)
    result_img.show()
