import os, sys, inspect

currentdir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
parentdir = (currentdir)
sys.path.insert(0, parentdir) 

from config import Config
from vision.model_factory import create_model

if __name__ == '__main__':
    model = create_model(
        type=Config.get("model_type"),
        dataset_name=Config.get("dataset_name"), 
        saved_model_filename=Config.get("saved_model_filename")
    )


    model.train(
        num_epochs=Config.get("num_epochs"),
        batch_size=Config.get("batch_size"),
        test_batch_size=Config.get("test_batch_size")
    )

    model.save()
