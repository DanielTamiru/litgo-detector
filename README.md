# litgo-detector

### **LITGO.AI**

Litgo is a microservice-based application I am developing for reporting/cleaning city litter, where pictures of urban waste are evaluated by a trained convolutional neural network (CNN). Users take pictures of litter in their environment and either report it to region representatives or clean it themselves. They earn points based this choice and on the amount (and potentially type) of litter identified.

### **Detector Service**

This is the decomposed Litgo detector service, an HTTP server that receives images in PUT requests and returns information about the amount and type of litter in those images, including an annotated image with labels and boxes.

![tmp0t507npo](https://user-images.githubusercontent.com/46784904/236659272-3964160d-4f84-4db1-8a44-06c8e64a837a.png)

While this service was built for Litgo, none of its code or functionality is Litgo-specific. Feel free to incorporate this service into your own innovations. If you are new to computer vision models, object detection, or machine learning in general, don't worry! So was I before working on this project. I included a [wiki doc](https://github.com/DanielTamiru/litgo-detector/wiki/CNNs-and-Object-Detection) that helps explain what's going on at a high level.

The service uses COCO datasets (two for now: [TACO](http://tacodataset.org/) and [UAVVaste](https://github.com/UAVVaste/UAVVaste)), to train an object detection model (just [Mask R-CNN](https://pytorch.org/vision/main/models/mask_rcnn.html) so far). At start up, you can choose to either retrain a model instance on the training images in [data/](https://github.com/DanielTamiru/litgo-detector/tree/main/data) or use a saved model in [mask_rcnn/saved_models/](https://github.com/DanielTamiru/litgo-detector/tree/main/saved_models). Other configs, including training and evaluation parameters, can be set in `config.py`.

I made this service loosely following pytorch's [object detection finetuning tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) - please check that out for more context.

### **Getting Started**

You very likely want to use saved models and to do so, you need to download the files in this [drive folder](https://drive.google.com/drive/folders/1ep7SQn6rUYkviYS-qWdAQSzOT7gH_BYe?usp=share_link) into `vision/saved_models`.

There are three ways to run this service:
- `app/serve.py`: starts a flask HTTP server that evaluates images from form-data PUTs. 
<img width="606" alt="image" src="https://user-images.githubusercontent.com/46784904/235442191-2c328f0f-cbdd-4165-8a55-350fbfd69f91.png">

- `app/run.py`: a local script you can run on the file path to an image - `python app/run.py path/to/image -ls --color red`. The `l` option adds litter labels to the bounding boxes and the `s` option adds the confidence score.

- `app/train.py`: a local script you can use to train a given model on a specified dataset. Once training is complete, the model will be saved in `vision/saved_models` with the name `[dataset]-[model]-state-[timestamp].pt`. **Note:** training will only work if your machine has a CUDA-capable GPU, or `if torch.cuda.is_available() == True`.

### Baremetal
Simply create a python virtual environment using the venv tool of your choice and run `pip install -r requirements.txt`

### Docker

The Dockerfile builds an image that runs `server.py`. Build the image with the command `docker build -t [image name]:[tag]` and run it with
`docker run -p [host port]:[port in config.py] [image]`.


