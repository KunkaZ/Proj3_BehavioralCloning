# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/centerlane.jpg "Grayscaling"
[image3]: ./examples/recover1.jpg "Recovery Image"
[image4]: ./examples/recover2.jpg "Recovery Image"
[image5]: ./examples/recover3.jpg "Recovery Image"
[image6]: ./examples/normal.jpg "Normal Image"
[image7]: ./examples/normal_flipped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I am using the architecture from NVIDIA self driving group. Image cropping and normalizaiton are the first two layers. Then it's followed by 5 convolution layers with RELU activation function. It's fully connected neuron netowrk has 4 layers without activation funcitons. Summary of this architecture is shown as following.

Total params: 850,081

Trainable params: 850,081

Non-trainable params: 0

|Layer (type)                | Output Shape                  |  Param #|
|:--------------------------:|:-----------------------------:|:----------------:| 
| cropping2d_1 (Cropping2D)   | (None, 85, 320, 3)       | 0    
| lambda_1 (Lambda)           | (None, 85, 320, 3)       | 0         
| conv2d_1 (Conv2D)           | (None, 41, 158, 24)      | 1824      
| conv2d_2 (Conv2D)           | (None, 19, 77, 36)       | 21636     
| conv2d_3 (Conv2D)           | (None, 8, 37, 48)        | 43248     
| conv2d_4 (Conv2D)           | (None, 3, 18, 64)        | 27712     
| conv2d_5 (Conv2D)           | (None, 1, 8, 64)         | 36928     
| flatten_1 (Flatten)         | (None, 512)              | 0         
| dense_1 (Dense)             | (None, 1164)             | 597132    
| dense_2 (Dense)             | (None, 100)              | 116500    
| dense_3 (Dense)             | (None, 50)               | 5050      
| dense_4 (Dense)             | (None, 1)                | 51        
|




#### 2. Attempts to reduce overfitting in the model

Tried using LeNet architecture before and it has some overfitting issue. But I switched to NVIDIA architecture and collected more data. Didn't see overfitting issue in 10 epoch training. 

The model was trained and validated on different data sets to ensure that the model was not overfitting.



 The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 147).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
##### Extra data was collected to train the model:
1) Three laps of center lane driving 
2) one lap of recovery driving from the sides
3) two laps of driving counter-Clockwise


For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

####1. Solution Design Approach

My first try was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. When using this model, I always have overfitting issue and simulated driving is not good. I tried dropout but it didn't fix the issue.

Then I switched to NVIDIA CNN architecture for self driving. This model has more layers. But during training, overfitting issue becomes much better.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track or stuck somewhere, to improve the driving behavior in these cases, I recolloect some data at those spots showing how to avoid these situation.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture is in (model.py lines 122-145). Image cropping and normalizaiton are the first two layers. Then it's followed by 5 convolution layers with RELU activation function. It's fully connected neuron netowrk has 4 layers without activation funcitons. Summary of this architecture is shown as following.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover. These images show what a recovery looks like:

![alt text][image3]

![alt text][image4]

![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would  For example, here is an image that has then been flipped:

![alt text][image6]

![alt text][image7]

I had also collect 2 laps of data of driving counter-clockwise.

After the collection process, I had 21163*3*2=126978 number of data points. 


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 7 as evidenced by loss and val_loss data.  I used an adam optimizer so that manually training the learning rate wasn't necessary.

Demo video can be found at https://youtu.be/8VXqWbKseok
