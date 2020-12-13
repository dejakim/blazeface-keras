# blazeface-keras
Keras-Tensorflow implementation of [BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs](https://arxiv.org/pdf/1907.05047.pdf)

![Application](./screenshot.gif)

## Prerequisites
* tensorflow (2.x)
* opencv-python
* matplotlib
* tqdm

## Dataset Preparation
I used [Caltech 10,000 Web Faces](http://www.vision.caltech.edu/Image_Datasets/Caltech_10K_WebFaces/) for training. Download dataset from the website and decompress it in the [data](./data/) folder.

After decompression is done, run
```
$ python prepare.py
```
before training.

## Data Augmentation
For data augmentation, DataGenerator class is implemented in [datagenerator.py]('./datagenerator.py')

To use DataGenerator, change __use_generator = False__ to True which is defined in [train.py](./train.py).
DataGenerator randomly change rotation angle (0-45 deg) and scale (1.0-0.7).
Translation was not available so far since I wannted to keep as much valid ground truth box as possible.

## Training
Model is defined in [blazeface.py](./blazeface.py). I added dropout layer with __rate = 0.3__ at the end of Double BlazeBlock to prevent overfitting. Also, unlike original [Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf) adopts 3x3 kernel for classifier layer, I used 5x5 kernel for classification layer since 5x5 kernel shows better classification accuracy.

Initially, I implemented Online Hard Example Mining (OHEM) in loss function which is defined in [blazeface.py](./blazeface.py), but it didn't do much to improve accuracy.

Pretrained model is already available in _model/blazeface.h5_. You can skip this phase if you want to check live webcam application first.

To train this model, run
```
$ python train.py
```
Following parameters are defined by default
* batch_size: 16
* Adam optimizer with amsgrad
* Learning rate: 1e-3 for 50000 iter, 2e-4 for 40000 iter and 1e-4 for 30000 iter

Unfortunately, after model.fit is finished, memory was not cleared properly so that when model.fit called 2nd times, system raises 'memory not enough error' and terminated forcibly in CoLab free tier environment.

Thus, to train with dynamic learning rate by iterations, you need to run train.py multiple times. _If you know how to cleanup the memory properly after model.fit, please let me know!_

After training is done, model will be saved under _model/blazeface.h5_.

## Application
To test with live webcam application, run
```
$ python app.py
```

## Future Works
Increase accuracy with following methods:
* Test with FocalLoss
* Online Hard Example Mining
* Training with Data Augmentation
* More dataset
