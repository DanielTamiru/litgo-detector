import os
from typing import Callable


from vision.dataset import CocoDataset
from vision.helpers.train import get_transform

from constants import ROOT_DIR


class UAVVasteDataset(CocoDataset):
  
    def __init__(self, name: str, transforms: Callable, train: bool = False):
        super().__init__(
            img_dir=os.path.join(ROOT_DIR, 'data/UAVVaste/images'),
            annot_path=os.path.join(ROOT_DIR, 'data/UAVVaste/annotations.json'),
            transforms=transforms,
            auto_download=train
        )

    def get_name(self) -> str:
        return self.name
    

class TacoDataset(CocoDataset):
  
    def __init__(self, name: str, transforms: Callable, train: bool = False):
        self.name = name

        super().__init__(
            img_dir=os.path.join(ROOT_DIR, 'data/TACO/images'),
            annot_path=os.path.join(ROOT_DIR, 'data/TACO/annotations.json'),
            transforms=transforms,
            auto_download=train
        )

    def get_name(self) -> str:
        return self.name
    
        
########### Factory ############

def create_dataset(name: str, train: bool) -> CocoDataset:
    if name == "UAVVaste": return UAVVasteDataset("UAVVaste", get_transform(train=train), train)
    elif name == "TACO": return TacoDataset("TACO", get_transform(train=train), train)
    raise Exception(f"'{name}' is not a valid dataset name")
    