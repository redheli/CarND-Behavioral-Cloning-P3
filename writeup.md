**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/keyboard-control-data.png "keyboard Control Data"
[image2]: ./images/mouse-control-data.png "keyboard Control Data"
[image3]: ./images/speed.png "Speed Value"
[image4]: ./images/flip.png "Flip Center Image"
[image5]: ./images/corp.png "Corp Image"

---
### Data Collection And Preprocessing

At beginning I run the training simulator with keyboard buttons to control the steering angle.
When analysis the data, I found the significant proportion of the data is for driving straight, see following histogram.    
![alt text][image1]

But in the actual simulation track, the straight road is not so much long distance. Then why I got so many zero steering angle? This is because the design of the steering value with keyboard control. It is impossible mentain a certain steering value with keyboard. press the keybaord will increase/decrease the angle , once release the steering anlge turn to zero. So when drive the car, I cannot make smooth curlve path , and the car turn with a lot line segments.
The next question comes: Is it worth to deal with such data? I saw some people try to filter out the zero steering vale. But intuitive tell me, I need better data.
So I recollect the data with mouse control. It turns out the steering angle distribution is more reasonable. As the track is anti-clockwise , the steering angle have more negative value.
![alt text][image2]

I drive the car with speed around 10 mph, as my driving skill only can stable the car in such speed :)
![alt text][image3]

I collect two laps data. And I also collect two times data just at one sharp turn.

### Data Augmentation

I flip the center camera images and steering measurements.
In track 1, most of the turns are left turns, flipping can make more right turn data. So the model wont bials to left turn.
As a result, the network would learn both left and right turns properly.
![alt text][image4]

I also corp the images, for all center,left,right images.
Because in most of the images,the up part is sky, and bottom part are car , which are not very useful for training, and on the other hand, it might lead to overfitting. So that I decided to crop out only the most useful part.
![alt text][image5]

### Resize Image?

At fist try, I resize the image to 64x64. It make the training faster, as input data size is small. But when test with autonouse mode, It always fail at one sharp turn, the car leave the track surface.
TODO image
I notice the road edge of that place is defferent with others, and in resize the image , the edge is blur. This is may confuse the model. I remove the step of resize, then the model works well.  Of course , the training time become longer.
TODO image


### Use Multiple Cameras

I use all three cameras as training inputs. This is because we need to handle the issue of recovering from being off-center driving.
** Turns out this is the most important step to make the model work. ** If we train the model to associate a given image from the center camera with a left turn, then we could also train the model to associate the corresponding image from the left camera with a somewhat softer left turn.And we could train the model to associate the corresponding image from the right camera with an even harder left turn.
To estimate the steering angle of the left and right images, I use a correction value of 0.2
TODO image

### Model Architecture and Training Strategy

I borrow the model from [End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) by Nvidia. Here is my model:
* First I normalize input image to -0.5 to 0.5.
* 3 convolution layers are applied with 5x5 filter size but the depth increases at each layer such as 24, 36, 48.
* 2 convolution layers are applied with 3x3 filter size and 64 depth. To avoid overfitting at convolution layers, Relu activation is applied after every convolution layers.
* flatten the data to input to FC layer
* The FC layers to 80, 40, 16, 10 and 1. At each layer, 50% Dropout is also applied for the first 3 dense layer to avoid overfitting.
* Adam optimizer is used. I started with 0.0001 learning rate ,and It produces a smoother ride. Therefore, I kept it.
TODO image

Here is tarining result. The final validation loss is 0.1283.
Epoch number is 10. I think the loss  is small enough.

TODO image

## Training Data

I split the data to training and validation(30%). For training data, I used a combination of center lane driving, recovering from the left and right sides of the road. After the collection process, I had 52853 number of data points.
But the left ,right images are not included in validation data. As the correction value(0,2) of measurements is an estimation. It may not represent the correct value.

## Generators
I didnot use the Generators, as my laptop has enough momery to load all the data :)
