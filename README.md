# Face Mask Detector

### Face Mask Detection system built with PIL, Pytorch using Deep Learning and Computer Vision algorithms to detect face masks in images and real-time video streams.

![My current project](https://github.com/shushukurov/FaceMaskDetector/blob/main/new.gif)


![fig2](https://github.com/shushukurov/FaceMaskDetector/blob/main/test_et7oykBp.gif)



## Purpose

In the ongoing global challenge of mitigating COVID-19, efficient tools and strategies for minimizing the spread of the virus remain crucial.

The World Health Organization identifies masks as a significant preventive measure to reduce transmission and save lives.

As a machine learning engineer, I believe in the capacity of technology to make a positive impact in our societies. Leveraging deep learning, we can enhance the observance of mask-wearing guidelines through real-time, proactive monitoring.

With this in mind, I've developed the 'Face Mask Detection System.' This tool is especially useful in settings such as public transportation, crowded urban areas, residential districts, large-scale manufacturing facilities, and various enterprises aiming to ensure public safety and health.

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

### FastAI (for BING search image data scraping) => Pytorch (for MobileNetV2 model training) => FaceNet (For face detecetion) => PIL and OpenCV (for image and video processing) => Flask (for serving model via REST API) => Docker (For containerization and CI/CD) => Microsoft Azure (For deployment) 

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
### TBA (After publication) and training screepts will be uploaded
## License
This project is under MIT license © [Shakhzod Shukurov](https://github.com/shushukurov/FaceMaskDetector/blob/main/LICENSE)

## Further info
Feel free to reach me by email: shushukurov@gmail.com to get more information
