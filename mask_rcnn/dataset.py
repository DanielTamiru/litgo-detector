import os
from io import BytesIO
from typing import Callable, Tuple, List

from definitions import ROOT_DIR
import mask_rcnn.helpers.transforms as T

import torch
from pycocotools.coco import COCO

from PIL import Image
from tqdm import tqdm
import requests



class CocoDataset(torch.utils.data.Dataset):

    def __init__(self, img_dir: str, annot_path: str, transforms: Callable, auto_download=False):
        self.root = img_dir # training image directory
        self.transforms = transforms # transformations 

        self.coco = COCO(annot_path) # Common Objects in Context manager
        if auto_download: self.download_images()


    def __getitem__(self, idx) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        img_path =  os.path.join(self.root, self.coco.loadImgs([idx])[0]["file_name"])
        img = Image.open(img_path).convert("RGB")

        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[idx]))
        
        # convert annotation boxes to tensor
        boxes = self._get_converted_annotation_bboxes(annotations)
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        # convert annotation segmentation masks to tensor
        masks = [self.coco.annToMask(annot) for annot in annotations]
        masks_tensor = torch.as_tensor(masks, dtype=torch.uint8)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.ones((len(annotations),), dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0]),
            "iscrowd": torch.tensor(list(map(lambda a: a["iscrowd"], annotations)), dtype=torch.uint8),
            "masks": masks_tensor
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
    

    def __len__(self) -> int:
        return len(self.coco.imgs)
    

    def _get_converted_annotation_bboxes(self, annotations) -> List[Tuple[int, int, int, int]]:
        """
        COCO represents bounding boxes using (xmin, ymin, width, height)
        but we need them in (xmin, ymin, xmax, ymax) format
        """
        boxes = []

        for annotation in annotations:
            xmin, ymin = annotation["bbox"][0], annotation["bbox"][1]
            
            xmax = xmin + annotation["bbox"][2] # + width
            ymax = ymin + annotation["bbox"][3] # + height
        
            boxes.append((xmin, ymin, xmax, ymax))

        return boxes
    

    def download_images(self) -> None:
        for img_obj in tqdm(self.coco.imgs.values()):
            file_path = f'{self.root}/{img_obj["file_name"]}'

            # Create subdirectory in images if necessary
            subdir = os.path.dirname(file_path)
            if not os.path.isdir(subdir): os.mkdir(subdir)

            # download and save image if it has a flickr url and if it doesn't already exist
            if img_obj['flickr_url'] is not None and not os.path.isfile(file_path):
                img_url = img_obj['flickr_url']
                response = requests.get(img_url, allow_redirects=True)

                if response.ok:
                    img = Image.open(BytesIO(response.content))
                    if img._getexif(): img.save(file_path, exif=img.info["exif"])
                    else: img.save(file_path)
                else:
                    print(f'Failed to download: {img_obj["file_name"]}')




############ UAVVaste ############

class UAVVasteDataset(CocoDataset):
    IMG_HEIGHT = 2160; IMG_WIDTH = 3840; 
  
    def __init__(self, transforms: Callable):
        super().__init__(
            img_dir=os.path.join(ROOT_DIR, 'data/UAVVaste/images'),
            annot_path=os.path.join(ROOT_DIR, 'data/UAVVaste/annotations.json'),
            transforms=transforms,
            auto_download=True
        )
