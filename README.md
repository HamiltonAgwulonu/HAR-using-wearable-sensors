2019 August

Human-Activity-Recognition Using Sensor Data 
=======================================


---------------------------------------------------------------------------

Overview of Repository:
----------------------------------

The purpose of this project is to build a machine learning model that is able to predict human activities such as 

- Race("run) -- Walking -- Sitting -- Standing -- Get-up-and-sit-down ('sit-to-stand', 'stand-to-sit') -- Up and downstairs (stair-up, stair-down) -- Jump on one or both legs ('jump-one-leg', 'jump-two-leg') -- Run left or right ('curve-left-step', 'curve-right-step)       -- Turn left or right on the spot, left or right foot first ('curve-left-spin-Lfirst#, 'curve-left-spinRfirst', 'curve-right-spin-Lfirst', 'curve-right- spin-Rfirst') 
-- Lateral steps to the left or right ('lateral-shuffle-left', 'lateral-shuffle-right'), 
-- Change of direction when running to the right or left, left or right foot first ('v-cut-left-left',
'v-cut-left-right', 'v-cut-right-left', 'v-cut' right-Rfirst') from the Sensor data attached to the left thigh of an individual. 
<br><br>
The repository contains 5 ipython notebooks listed in the order below
<br>
1 [HAR_Exploratory_Data_Analysis.ipynb](https://github.com/HamiltonAgwulonu/HAR-using-wearable-sensors/blob/master/HAR_Exploratory_Data_Analysis.ipynb) : Data pre-processing and Exploratory Data Analysis
<br>
2 [HAR_Predictions.ipynb] : Machine Learning models predictions on test data data
<br>
3 [HAR_CNN.ipynb]  : CNN model on raw timeseries data
<br>
4 [HAR_HMM.ipynb] : CNN model on raw timeseries data
<br>
5 [HAR_CNN+HMM.ipynb] : CNN model on raw timeseries data
<br><br>

All the code is written in python 3.
<br><br>
**LIST OF DEPENDENCIES**
* tensorflow
* keras
* numpy
* pandas
* matplotlib
* seaborn
* sklearn
* itertools
* datetime


Introduction:
------------------

Sensing devices are quiped with a number of [sensors](https://www.gsmarena.com/glossary.php3?term=sensors). our focus is on two of the sensors: Accelerometer and Gyroscope.
<br>
The data recordings where taken by the sensors
<br>
This is a 22 class classification problem as we have 22 activities to detect.<br>

This project has two parts, the first part trains, tunes and compares Convolutional Neural Networks (CNN) and Hidden Marcov Models (HMM) and uses the data featured by domain expert.<br>
 
The second part uses the raw time series windowed data to train Hybrid Model CNN + HMM. Both models are tuned using the Rectified Linear units to fast forward the training task.

-----------------------------------------------------

Dataset:
--------

The dataset can be downloaded from
https://bbdc.csl.uni-bremen.de/images/2019/bbdc_2019_Bewegungsdaten_mit_referenz.zip
<br><br>
dataset is also included in the Repository with in the folder [UCI_HAR_Dataset](https://github.com/HamiltonAgwulonu/HAR-using-wearable-sensors)
<br><br>
Human Activity Recognition database is built from the recordings of 30 persons performing activities of daily living (ADL) while carrying a waist-mounted smartphone with embedded inertial sensors(accelerometer and Gyroscope).
<br>
**Activities**
* Race('run')
* Get up and sit down ('sit-to-stand', 'stand-to-sit')
* Up and down stairs ('stair-up', 'stair-down')
* Sitting ('sit')
* Standing ('stand')
* Jump on one or both legs ('jump-one-leg', 'jump-two-leg')
* Run left or right ('curve-left-step', 'curve-right-step')
* Lateral steps to the left or right ('lateral-shuffle-left', 'lateral-shuffle-right')
* Change of direction when running to the right or left, left or right foot first  ('v-cut-left-left',
'v-cut-left-right', 'v-cut-right-left', 'v-cut-right-right')

[**Accelerometers**](https://en.wikipedia.org/wiki/Accelerometer) detect magnitude and direction of the proper acceleration, as a vector quantity, and can be used to sense orientation (because direction of weight changes)
<br><br>
[**GyroScope**](https://en.wikipedia.org/wiki/Gyroscope) maintains orientation along a axis so that the orientation is unaffected by tilting or rotation of the mounting, according to the conservation of angular momentum.
<br><br>
Accelerometer measures the directional movement of a device but will not be able to resolve its lateral orientation or tilt during that movement accurately unless a gyro is there to fill in that info.
<br>
With an accelerometer you can either get a really "noisy" info output that is responsive, or you can get a "clean" output that's sluggish. But when you combine the 3-axis accelerometer with a 3-axis gyro, you get an output that is both clean and responsive in the same time.
<br><br>
#### Understanding the dataset
* Both sensors generate data in 3 Dimensional space over time. Hence the data captured are '3-axial linear acceleration'(_tAcc-XYZ_) from accelerometer and '3-axial angular velocity' (_tGyro-XYZ_) from Gyroscope with several variations.
* prefix 't' in those metrics denotes time.
* suffix 'XYZ' represents 3-axial signals in X , Y, and Z directions.
* The available data is pre-processed by applying noise filters and then sampled in fixed-width windows(sliding windows) of 2.56 seconds each with 50% overlap. ie., each window has 128 readings.
#### Featurization
For each window a feature vector was obtained by calculating variables from the time and frequency domain. each datapoint represents a window with different readings.<br>
Readings are divided into a window of 2.56 seconds with 50% overlapping. 
* Accelerometer readings are divided into gravity acceleration and body acceleration readings,
  which has x,y and z components each.

* Gyroscope readings are the measure of angular velocities which has x,y and z components.

* Jerk signals are calculated for BodyAcceleration readings.

* Fourier Transforms are made on the above time readings to obtain frequency readings.

* Now, on all the base signal readings., mean, max, mad, sma, arcoefficient, engerybands,entropy etc., are calculated for each window.

* We get a feature vector of 561 features and these features are given in the dataset.

* Each window of readings is a datapoint of 561 features,and we have 10299 readings.

* These are the signals that we got so far.(prefix t means time domain data, prefix f means frequency domain data)

#### Train and test data were saperated

 - The readings from ___70%___ of the volunteers(21 people) were taken as ___trianing data___ and remaining ___30%___ volunteers recordings(9 people) were taken for ___challenge data___
* All the data is present in '/' folder in present working directory.
     - Feature names are present in 'UCI_HAR_dataset/features.txt'
     - ___Train Data___ (7352 readings)
         - 'UCI_HAR_dataset/train/X_train.txt'
         - 'UCI_HAR_dataset/train/subject_train.txt'
         - 'UCI_HAR_dataset/train/y_train.txt'
     - ___Test Data___ (2947 readinds)
         - 'UCI_HAR_dataset/test/X_test.txt'
         - 'UCI_HAR_dataset/test/subject_test.txt'
         - 'UCI_HAR_dataset/test/y_test.txt'
 
-------------------------------------------------------------------------------
