from mask_rcnn.model import MaskRCNNLitgoModel
from mask_rcnn.draw import draw_boxes
from PIL import Image

if __name__ == "__main__":
    model = MaskRCNNLitgoModel(dataset_name="TACO", saved_model_filename="TACO-state-1681930038.pt")
    
    with Image.open("litter.jpeg") as img:
        result = model.evaluate(img, score_threshold=0.5)
        result_img = draw_boxes(img, result, color="blue", draw_score=False)
        # result_img.show()
        result_img.save('labeled_litter.jpeg')
