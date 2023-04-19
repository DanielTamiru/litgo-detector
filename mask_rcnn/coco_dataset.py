import os
from io import BytesIO

import torch, numpy
from pycocotools.coco import COCO

from PIL import Image
from tqdm import tqdm
import requests

from typing import Callable, Tuple

from mask_rcnn.helpers.coco_utils import ConvertCocoPolysToMask


class CocoDataset(torch.utils.data.Dataset):
    coco: COCO

    def download_images(self) -> None:
        raise NotImplementedError

    def get_num_classes(self) -> int:
        raise NotImplementedError
    
    def get_category_from_label(self, label: int) -> Tuple[str, str]:
        raise NotImplementedError
    
    @staticmethod
    def to_label(category_id: int) -> int: 
        # bboox label is offset by 1 because 0 should be considered 'background'
        return category_id + 1
    
    @staticmethod
    def to_category_id(label: int) -> int: 
        # bboox label is offset by 1 because 0 should be considered 'background'
        return label - 1
    


class CocoDatasetImpl(CocoDataset):

    def __init__(self, img_dir: str, annot_path: str, transforms: Callable, auto_download=False):
        self.root = img_dir # training image directory
        self.transforms = transforms # transformations 

        self.coco = COCO(annot_path) # Common Objects in Context manager
        self.ids = list(sorted(self.coco.imgs.keys())) # indexes correspond to ids in the coco annotations

        self.prepare = ConvertCocoPolysToMask()

        if auto_download: self.download_images()


    def __getitem__(self, idx) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_id = self.ids[idx]

        img_path =  os.path.join(self.root, self.coco.loadImgs([image_id])[0]["file_name"])
        img = Image.open(img_path).convert("RGB")

        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[image_id]))
        
        img, target = self.prepare(image=img, target={'image_id': image_id, 'annotations': annotations})
  
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
    

    def __len__(self) -> int:
        return len(self.coco.imgs)
    

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


    def get_num_classes(self) -> int:
        return len(self.coco.getCatIds()) + 1


    def get_category_from_label(self, label: int) -> Tuple[str, str]:
        cat_id = self.to_category_id(label)
        category = self.coco.loadCats([cat_id])[0]
        return category["name"], category["supercategory"]
