# **Behavioral Cloning** 

## Intro.

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
## Files Submitted & Code Quality

### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

### 1. An appropriate model architecture has been employed

<center><img src="./examples/LeNet.png"></center>

처음 적용한 모델은 기존 수업에 사용했었던 LeNet을 적용하여 수행했습니다.

이후 수업에서 좀 더 powerful 한 모델로 nVidia 모델을 소개해서 해당 모델을 최종적으로 적용했습니다.

### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 112,117). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 129). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 141).

### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

그리고 추가적으로 역주행으로 1 lap을 데이터에 추가하였습니다.

For details about how I created the training data, see the next section. 

## Model Architecture and Training Strategy

### 1. Solution Design Approach

LeNet 모델을 적용한 이후에 실제로 시뮬레이터를 통해 AD 주행을 하였을 때, 자동차의 운행이 매끄럽게 진행되지 않음을 확인했습니다.

그래서 수정 중에 소개된 새로운 모델을 적용해보았습니다.

다만, 한번 지정된 도로를 벗어나면 되돌아 오지 못하고 그대로 retire 하는 문제가 발생하였습니다.

이를 방지하기 위해 일부러 training data set에 line을 나갔다가 돌아오는 모습을 일부 추가하였습니다.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### 2. Final Model Architecture

구현한 모델은 수업 중에 배운 nVidia 모델을 이용하였습니다.

The final model architecture (model.py lines 102-119) consisted of a convolution neural network with the following layers and layer sizes .

일부분은 data에 맞게 그리고 dropout을 추가하기 위해 수정했습니다.

| Layer     |     Description   |
|:---------------------:|:---------------------------------------------:|
| Input         | 160x320x3 RGB image   |
| Normalization | outputs 160x320x3     |
| Corpping2D    |  outputs 90x320x3     |
| Convolution 5x5   | 2x2 stride, Valid padding, outputs 43x158x24 |
| RELU          |                       |
| Convolution 5x5   | 2x2 stride, Valid padding, outputs 20x77x36   |
| RELU          |                       |
| Convolution 5x5   | 2x2 stride, Valid padding, outputs 8x37x48    |
| RELU          |                       |
| Convolution 5x5   | 2x2 stride, Valid padding, outputs 6x35x64    |
| RELU          |                       |
| Convolution 5x5   | 2x2 stride, Valid padding, outputs 4x33x36    |
| RELU          |                       |
| Dropout       | 0.25                  |
| Flatten       | input 4x33x64, outputs 8448                       |
| Fully connected   | input 8448, output 100                        |
| Fully connected   | input 100, output 50                          |
| Fully connected   | input 50, output 10                           |
| Dropout           | 0.25                                          |
| Fully connected   | input 10, output 1                            |

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

<center><img src="./examples/nVidia_model.png"></center>

### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

총 2번의 lap으로 데이터를 구성했고, 최대한 도로의 중앙으로 운행할 수 있도록 하였습니다.

<center><img src="./examples/normal.jpg"></center>

I then recorded the vehicle recovering from the left side and right sides of the road back to center. 앞서 언급하였듯이 한번 자동차가 도로 밖으로 나가게 되면 그대로 retire 하게 되는 경우를 발견하였습니다. 이를 해결하기 위해 데이터에 recovery 운행에 대해 추가하였습니다.  

아래 예시 이미지는 도로 오른쪽 밖으로 나가는 경우에 대한 예시 입니다.

<center><img src="./examples/recovery1.jpg"></center>
<center><img src="./examples/recovery2.jpg"></center>
<center><img src="./examples/recovery3.jpg"></center>

To augment the data sat, I also flipped images. 기본적으로 lap의 데이터로만 구성할 경우 generlize 되지 않을 거라 판단되어 flip 된 image을 추가하였습니다. 

After the collection process, I had 10108 number of data points. I then preprocessed this data. 중요한 부분만 보기 위해 Cropping2D 로 일부 데이터를 잘랐습니다.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2. 여러번 수행을 통해 얻어냈습니다.

 I used an adam optimizer so that manually training the learning rate wasn't necessary.

 accuracy 결과는 아래와 같이 나왔습니다.

 <center><img src="./examples/results.png"></center>

해당 모델을 이용하여 최종적으로 수행한 1 lap 자율주행 결과는 아래 동영상과 같습니다.

[![Video Label](http://img.youtube.com/vi/z4rIIFeOBak/0.jpg)](https://youtu.be/z4rIIFeOBak?t=0s)
