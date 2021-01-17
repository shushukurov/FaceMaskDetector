# FaceMaskDetector

### Face Mask Detection system built with PIL, Pytorch using Deep Learning and Computer Vision algorithms to detect face masks in images and real-time video streams.

![My current project](https://github.com/shushukurov/FaceMaskDetector/blob/main/new.gif)


![fig2](https://github.com/shushukurov/FaceMaskDetector/blob/main/test_et7oykBp.gif)


## Motivation

During these tough	times, we are still in search of efficient tools and solutions agains spread of COVID-19.

According to WHO 'Masks are a key measure to suppress transmission and save lives'

As a machine learning engineer and the member of society I feel that I have a responsibility to make things better in society

I can utilise the power of deep learning for the effective monitoring of following mask wearing guidelines that enables proactive, real-time responses.

For this purpose, I have developed ‘Face mask detection system’ which is extremely beneficial for transportation means, densely populated areas, residential districts, large-scale manufacturers and other enterprises to ensure safety.

## Project demo

### [Deployed model on Azure web service](https://facemask.azurewebsites.net)

Note: Please use appropriate picture 

Model returns image with bounding box on detected face filled with red or green depending whether mask is detected or not

## TechStack/framework used

[Pytorch](https://pytorch.org)

[FaceNet](https://github.com/timesler/facenet-pytorch)

[MobileNetV2](https://arxiv.org/abs/1801.04381)

[PIL](https://pillow.readthedocs.io/en/stable/#)

[OpenCV](http://opencv.org)

[FastAI](http://fast.ai)

[Flask](https://flask.palletsprojects.com/en/1.1.x/)

[Docker](http://docker.com)

[Microsoft Azure](http://azure.microsoft.com)

## Description


This new feature relies on AI-enabled video analytics in the Enterprise Suite which determines when an individual is not wearing a face mask within your establishment. Once the feature detects an individual is not wearing a mask, this prompts an automated alert.

## Dataset

This dataset consists of 3800 images belonging to two classes:

with_mask: ~2000 images
without_mask: ~2000 images
The images used were real images of faces wearing masks. The images were collected from the following sources:

Bing Search API (See Python script)
Kaggle datasets
RMFD dataset (See here)

## Results

## License
MIT © [Shakhzod Shukurov](https://github.com/shushukurov/FaceMaskDetector/blob/main/LICENSE)
