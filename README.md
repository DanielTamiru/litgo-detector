# litgo-detectoor


### **LITGO.AI**

Litgo is a microservice-based application I am developing for reporting/cleaning city litter, where pictures of urban waste are evaluated by a trained convolutional neural network (CNN). Users take pictures of litter in their environment and either report it to region representatives or clean it themselves. They earn points based this choice and on the amount (and potentially type) of litter identified.

### **Detector Service**

This is the repo for the decomposed Litgo detector service, an HTTP server that receives images in PUT requests and returns information about the amount and type of litter in those images, including an annotated image with labels and boxes.

[Image]

While this service was built for Litgo, none of its code or functionality is Litgo-specific. Feel free to incorporate this service into your own innovations. If you are new to computer vision models, object detection, or machine learning in general, don't worry! So was I before working on this project. I included a [wiki doc](https://github.com/DanielTamiru/litgo-detector/wiki/CNNs-and-Object-Detection) that helps explain what's going on at a high level.

The service uses a choice of COCO datasets (two for now), [TACO](http://tacodataset.org/) and [UAVVaste](https://github.com/UAVVaste/UAVVaste), to train a Mask R-CNN (object detection) model. At start up, you can choose to either retrain a model instance on the training images in [data/](https://github.com/DanielTamiru/litgo-detector/tree/main/data) or use a saved model in [mask_rcnn/saved_models/](https://github.com/DanielTamiru/litgo-detector/tree/main/saved_models).

I made this service loosely following pytorch's [object detection finetuning tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html), please check that out for more context.

### **Getting Started**

### Docker

### Baremetal
