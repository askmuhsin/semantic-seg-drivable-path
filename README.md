# Semantic Segmentation
### Introduction
A pixel wise segmentation of drivable path using Fully Convolutional Neural-Network (FCN).
![task at hand](https://github.com/askmuhsin/semantic-seg-drivable-path/blob/master/images/task_explain.png)

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

##### Run
Run the following command to run the project:
```
python main.py
```

### Discussion
The network implemented in this project is based on this [paper](https://arxiv.org/pdf/1605.06211.pdf).    
![fcn structure](https://github.com/askmuhsin/semantic-seg-drivable-path/blob/master/images/fcn_schema.png)

#### Training Details
##### System :    AWS EC2 || Instance type : g3.4xlarge     
Ubuntu 16.04    
Python 3.6+    
Tensorflow 1.2+     
GPU --> Nvidia Tesla M60 (8GB)    
Training time : 40 mins (with following hyperparameters)     

---
##### Model params :    
Learning rate : 0.001    
Dropout (training keep prob) : 0.5    
Epochs : 50    
Batch size : 12    
Kernel Regularizer : l2_regularizer(1e-3)     

### Result
Some of the images obtained after training:    
![sample_1](https://github.com/askmuhsin/semantic-seg-drivable-path/blob/master/images/um_000032.png)
![sample_2](https://github.com/askmuhsin/semantic-seg-drivable-path/blob/master/images/um_000034.png)
![sample_3](https://github.com/askmuhsin/semantic-seg-drivable-path/blob/master/images/um_000072.png)
![sample_4](https://github.com/askmuhsin/semantic-seg-drivable-path/blob/master/images/um_000085.png)    
False positive sample:    
![sample_5](https://github.com/askmuhsin/semantic-seg-drivable-path/blob/master/images/um_000078.png)     
False negative sample:     
![sample_6](https://github.com/askmuhsin/semantic-seg-drivable-path/blob/master/runs/1526462155.1956172/um_000073.png)   

#### Future improvements
[ ] Augment training images for better performance     
[ ] Use more datapoints (from other similar datasets; cityscape like)    
[ ] Increase classes (add pedestrians, vehicles, ...)      

### Rubric achieved:
1. All the unit test have passed.
2. main.py runs without errors
3. Newest inference images from `runs` folder included  (**all images from the most recent run**)
