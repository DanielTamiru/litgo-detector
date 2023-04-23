
from mask_rcnn.model import EvalutationResult
from PIL import Image, ImageDraw

def draw_boxes(img: Image, result: EvalutationResult, color="red", draw_categories=True, draw_score=True) -> Image:
    box_num = len(result["boxes"])
    draw = ImageDraw.Draw(img)

    for i in range(box_num):
        x_min, y_min = result["boxes"][i][0], result["boxes"][i][1]
        x_max, y_max = result["boxes"][i][2], result["boxes"][i][3]

        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=color, width=3)

        if draw_categories:
            text = result["supercategories"][i] if result["supercategories"][i] else result["categories"][i]
            pos = (x_min + 5, y_min + 5)

            textbox = draw.textbbox((x_min + 5, y_min + 5), text=text)
            draw.rectangle(textbox, fill=color)  # fill textbox with color
            draw.text(pos, text=text) # draw text over textbox

        if draw_score:
            score = round(result["scores"][i], 2)
            pos = (x_max - 30, y_max - 15)

            textbox = draw.textbbox(pos, text=str(score))
            draw.rectangle(textbox, fill=color) # fill textbox with color
            draw.text(pos, text=str(score)) # draw text over textbox
    
    return img