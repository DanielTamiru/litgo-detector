from typing import Optional, Tuple

import torch, torchvision
from torch.utils.data import DataLoader

from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from mask_rcnn.coco_dataset import CocoDataset

from mask_rcnn.datasets import coco_dataset_factory

from mask_rcnn.helpers.engine import train_one_epoch, evaluate
from mask_rcnn.helpers import utils

from os import path
from definitions import ROOT_DIR

from datetime import datetime



SAVED_MODELS_DIR = path.join(ROOT_DIR, "mask_rcnn/saved_models/")


class LitgoModel:
    model: MaskRCNN
    dataset: CocoDataset

    def train(self, num_epochs: int, batch_size: int, test_batch_size: int, save: bool = True):
        raise NotImplementedError
    

class LitgoModelImpl(LitgoModel):

    def __init__(self, dataset_name: str, saved_model_filename: Optional[str] = None) -> None:
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        self.dataset = coco_dataset_factory(dataset_name, train=True)

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
            self.model.load_state_dict(torch.load(path.join(SAVED_MODELS_DIR, saved_model_filename)))


    def train(self, num_epochs: int, batch_size: int, test_batch_size: int, save: bool = True):
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

        if save: 
            filename = f"{self.dataset.name()}-{int(datetime.today().timestamp())}.pt"
            torch.save(self.model.state_dict(), path.join(SAVED_MODELS_DIR, filename))

    
    def _get_data_loaders(self,  batch_size: int, test_batch_size: int) -> Tuple[DataLoader, DataLoader]:
        validation_dataset = coco_dataset_factory(self.dataset.name(), train=False)

        # split the dataset into train and validation sets
        indices = torch.randperm(len(self.dataset)).tolist()
        training_dataset = torch.utils.data.Subset(self.dataset, indices[:-50])
        validation_dataset = torch.utils.data.Subset(validation_dataset, indices[-50:])

        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            training_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            validation_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4,
            collate_fn=utils.collate_fn)
        
        return data_loader, data_loader_test
        
