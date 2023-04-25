from typing import Optional

from vision.dataset_factory import create_dataset
from vision.model import LitgoModel

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor



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
            self.load_saved_state(saved_model_filename)



########### Factory ############

def create_model(type: str, dataset_name: str, saved_model_filename: Optional[str] = None) -> LitgoModel:
    if type == "MaskRCNN": return MaskRCNNLitgoModel(dataset_name, saved_model_filename)
    raise Exception(f"'{type}' is not a valid model type")
    