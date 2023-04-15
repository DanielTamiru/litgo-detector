from io import BytesIO
import os
from typing import Callable

import torch
from vision.transforms import Compose, ToTensor, RandomHorizontalFlip

from PIL import Image
from tqdm import tqdm
import requests

from pycocotools.coco import COCO # common object in context

#### --------- O: Gather info ----------- ####

# Size of images used in dataset - image tensors have shape (C, IMG_HEIGHT, IMG_WIDTH)
IMG_HEIGHT = 2160; IMG_WIDTH = 3840; 

# Paths to json annotation file and to the directory where images are downloaded to
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) 
ANNOTATIONS_FILE = os.path.join(ROOT_DIR, 'data/UAVVaste/annotations.json') 
IMAGES_DIR = os.path.join(ROOT_DIR, 'data/UAVVaste/images')


#### --------- 1: Defining the dataset ----------- ####


class UAVVasteDataset(torch.utils.data.Dataset):
    def __init__(self, transforms: Callable):
        self.root = IMAGES_DIR
        self.transforms = transforms

        self.coco = COCO(ANNOTATIONS_FILE)
        self.download_images()

    def download_images(self) -> None:
        for img_obj in tqdm(self.coco.imgs.values()):
            file_path = f'{self.root}/{img_obj["file_name"]}'

            # Create subdirectory in images if necessary
            subdir = os.path.dirname(file_path)
            if not os.path.isdir(subdir): os.mkdir(subdir)

            # download and save image if it has a flickr url and if it doesn't already exist
            if img_obj['flickr_url'] is not None and not os.path.isfile(file_path):
                response = requests.get(img_obj['flickr_url'], allow_redirects=True)

                if response.ok:
                    img = Image.open(BytesIO(response.content))
                    if img._getexif(): img.save(file_path, exif=img.info["exif"])
                    else: img.save(file_path)
                else:
                    print(f'Failed to download: {img_obj["file_name"]}')


    def __getitem__(self, idx):
        img_path =  os.path.join(self.root, self.coco.loadImgs([idx])[0]["file_name"])
        img = Image.open(img_path).convert("RGB")

        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[idx]))
        
        boxes = self._get_converted_annotation_bboxes(annotations)
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)

        target = {
            "boxes": boxes_tensor,
            "labels": torch.ones((len(annotations),), dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0]),
            "iscrowd": torch.tensor(list(map(lambda a: a["iscrowd"], annotations)), dtype=torch.uint8)
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
    

    def _get_converted_annotation_bboxes(self, annotations):
        """
        COCO represent bounding boxes using (xmin, ymin, width, height)
        but we need them in (xmin, ymin, xmax, ymax) format
        """
        boxes = []

        for annotation in annotations:
            xmin, ymin = annotation["bbox"][0], annotation["bbox"][1]
            
            xmax = xmin + annotation["bbox"][2] # + width
            ymax = ymin + annotation["bbox"][3] # + height
        
            boxes.append([xmin, ymin, xmax, ymax])

        return boxes


def get_transform(train=False) -> Callable:
    transforms = [ToTensor()]
    if train: transforms.append(RandomHorizontalFlip(0.5))
        
    return Compose(transforms)



#### --------- 2: Defining the model ----------- ####
d = UAVVasteDataset(get_transform(train=True))
print(d[0][1])