from typing import List, Optional, Tuple, TypedDict

import torch, torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from vision.dataset import CocoDataset

from vision.dataset_factory import create_dataset

from vision.helpers.engine import train_one_epoch, evaluate
from vision.helpers import utils

from os import path
from constants import ROOT_DIR

from PIL import Image
from datetime import datetime



SAVED_MODELS_DIR = path.join(ROOT_DIR, "vision/saved_models/")



class EvalutationResult(TypedDict):
    categories: List[str]
    supercategories: List[str]
    scores: List[int]
    boxes: List[Tuple[float, float, float, float]]



class LitgoModel:
    model: GeneralizedRCNN
    dataset: CocoDataset

    def __init__(self, name: str) -> None:
        self.name

    def train(self, num_epochs: int, batch_size: int, test_batch_size: int):
        raise NotImplementedError
    
    def evaluate(self, image: Image, score_threshold: float) -> EvalutationResult:
        raise NotImplementedError
    
    def name(self):
        return self.name

    def save(self):
        timestamp = int(datetime.today().timestamp())
        state_filename = f"{self.dataset.get_name()}-{self.name()}-state-{timestamp}.pt"
        torch.save(self.model.state_dict(), path.join(SAVED_MODELS_DIR, state_filename))
    
    


class MaskRCNNLitgoModel(LitgoModel):
    model: MaskRCNN

    def __init__(self, dataset_name: str, saved_model_filename: Optional[str] = None) -> None:
        super().__init__("MaskRCNN")

        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        self.dataset = create_dataset(dataset_name, train=False)

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.dataset.get_num_classes())

        # now get the number of input features for the mask classifier
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, self.dataset.get_num_classes()
        )

        if saved_model_filename: # use pre-trained model:
            if not path.isfile(path.join(SAVED_MODELS_DIR, saved_model_filename)):
                raise Exception(f"Could not find {saved_model_filename} in saved_models directory")
            
            if torch.cuda.is_available():
                self.model.load_state_dict(torch.load(path.join(SAVED_MODELS_DIR, saved_model_filename)))
            else:
                self.model.load_state_dict(torch.load(
                    path.join(SAVED_MODELS_DIR, saved_model_filename),
                    map_location=torch.device('cpu')
                ))


    def train(self, num_epochs: int, batch_size: int, test_batch_size: int):
        # get data loaders
        data_loader, data_loader_test = self._get_data_loaders(batch_size, test_batch_size)

        # train on the GPU or on the CPU, if a GPU is not available
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # move model to the right device
        self.model.to(device)

        # construct an optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=3,
                                                        gamma=0.1)

        print("Starting Training...")
        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(self.model, optimizer, data_loader, device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(self.model, data_loader_test, device=device)
        print("Done Training!")


    def evaluate(self, image: Image, score_threshold: float):
        self.model.eval()
        image_tensor = ToTensor()(image)

        with torch.no_grad():
            prediction = self.model([image_tensor])[0]

            # scores are guanteed to be in descending order so find left sublist that meets threshold
            cutoff = 0
            while prediction["scores"][cutoff] >= score_threshold: cutoff += 1 # TODO: Do binary search instead of linear
            labels = prediction["labels"][:cutoff].tolist()
    
            return EvalutationResult(
                categories=list(map(lambda l: self.dataset.get_category_from_label(l)[0],labels)),
                supercategories=list(map(lambda l: self.dataset.get_category_from_label(l)[1],labels)),
                scores= prediction["scores"][:cutoff].tolist(),
                boxes=prediction["boxes"][:cutoff].tolist()
            )

    
    def _get_data_loaders(self,  batch_size: int, test_batch_size: int) -> Tuple[DataLoader, DataLoader]:
        
        # split the dataset into train and validation sets
        indices = torch.randperm(len(self.dataset)).tolist()
        training_dataset = torch.utils.data.Subset(self.dataset, indices[:-50])
        validation_dataset = torch.utils.data.Subset(
            create_dataset(self.dataset.get_name(), train=True), indices[-50:]
        )

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            training_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
            collate_fn=utils.collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            validation_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2,
            collate_fn=utils.collate_fn)
        
        return data_loader, data_loader_test
    
