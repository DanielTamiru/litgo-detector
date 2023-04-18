import os
from typing import Callable


from mask_rcnn.coco_dataset import CocoDatasetImpl
from mask_rcnn.helpers.train import get_transform

from definitions import ROOT_DIR


class UAVVasteDataset(CocoDatasetImpl):
    IMG_HEIGHT = 2160; IMG_WIDTH = 3840; 
  
    def __init__(self, transforms: Callable):
        super().__init__(
            img_dir=os.path.join(ROOT_DIR, 'data/UAVVaste/images'),
            annot_path=os.path.join(ROOT_DIR, 'data/UAVVaste/annotations.json'),
            transforms=transforms,
            auto_download=True
        )

    def name(self) -> str:
        return "UAVVaste"
    

class TacoDataset(CocoDatasetImpl):
  
    def __init__(self, transforms: Callable):
        super().__init__(
            img_dir=os.path.join(ROOT_DIR, 'data/TACO/images'),
            annot_path=os.path.join(ROOT_DIR, 'data/TACO/annotations.json'),
            transforms=transforms,
            auto_download=True
        )

    def name(self) -> str:
        return "TACO"
    
        
########### Factory ############

def coco_dataset_factory(name: str, train: bool) -> CocoDatasetImpl:
    match name:
        case "UAVVaste": return UAVVasteDataset(get_transform(train=train))
        case "TACO": return TacoDataset(get_transform(train=train))
    