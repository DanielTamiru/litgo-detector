from os import environ as env


class Config:

  __conf = {
    "host": env.get("HOSTNAME", 'localhost'),
    "port": int(env.get("LITGO_SERVER_PORT", '8000')),

    "dataset_name": "UAVVaste", # alternatively: "TACO"
    "model_type": "MaskRCNN", # alternatively: ...
    "saved_model_filename": "UAVVaste-MaskRCNN-state-1681836817.pt",

    # Evaluate
    "score_threshold": 0.6,

    # Training
    "num_epochs": 10,
    "batch_size": 2,
    "test_batch_size": 1
  }
  
  __settable = []

  @staticmethod
  def get(name):
    return Config.__conf[name]

  @staticmethod
  def set(name, value):
    if name in Config.__settable:
      Config.__conf[name] = value
    else:
      raise NameError(f"Either could not find config var {name} or it could not be set")
