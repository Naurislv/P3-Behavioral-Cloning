# P3-Behavioral-Cloning
Third project submission at Udacity Self Driving Nanodegree program - training neural network using Keras to close driver behavioral 

Simulator is not included, download it at : https://github.com/udacity/self-driving-car-sim
Result video link : https://www.youtube.com/watch?v=dtxsSOoHhQQ

### Instructions ###

Plug and pray (play) evaluation instructions :

	1. start Udactiy Self Driving Car simulator
	2. choose any track - works for both. Actually little better even for second track.
	3. execute in terminal :                                       python drive.py model.json
	   optional set throttle, recomended 1 for faster experience : python drive.py model.json 1
	4. watch car to drive itself - could you drive better ?

Plug and pray (play) training instructions :
	
	1. prepare training dataset which consists of :
		1.1 dirving_log.csv - logs of driving
		1.2 images in IMG directory and IMG must be in same directory where driving_log.csv is
	2. optionally - prepare validation set in same manner
	3. execute in terminal : python model.py --train_log path/to/logs/driving_log.csv
	
	Optional arguments:
		--output_model path/to/directory/  # to save model path must be provided
		--valid_log path/to/validation/logs/driving_log.csv
		--keras_weights path/to/keras/model/model.h5  # reused pretrained weights
		--epochs 27  # total epoch counts
		--dropout 1  # 1 enable dropout | 0 disable dropout
		--batch_size  # batch size for generator
	
	It is possible to make use of early stopping by just pressing ctrl+c whenever you feel that model has
	reached expectations.

### Information about technics, network and more ###

# Neural network structure  #

  Based on : https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

  Only different part of their network might be normalization which they did not fully explain.
  BatchNormalization from Keras were used.
  
  Loss function : mean squared error (mse). Also tried mae but mse worked better.

  The convolution layers are designed to perform feature extraction. The fully connected layers are designed
  to function as a controller for steering.

  Identical input image size were used because of kernel/stride sizes which by experiments they (NVidia) concluded
  worked best.
  
  I made various tests with Dropout placing after different layers (also Convolutions) but in my experience
  it's not good idea to put DO after convolutions - experiments approved that.

  Networks consists of 9 layers from which 5 are Convolutions and 4 are Dense layers. Total of 253 million parameters.
  Each layers size ,again, was based on NVidia provided information. Architecture is provided in zip - model.png.

# Training #

Training dataset consisted only of images from first track (left one). I used several datasets. 
At first dataset where I drove with keyboard. Second dataset I used my steering wheel and got smoother results.
Then recorded images when recovering after weaving over track but that is not necessary now because of Image
augmentation which is discussed further.

Generator was used to pass images to keras model.

# Overfitting | Image Augmentation #

Instead of splitting training dataset which might be affected by some un-noticed biases
I prepared other dataset with other logs which can be passed to this script therefore splitting
input dataset into valid/test datasets is not necessary.

When dataset is loaded script chooses not only center images but also left/right randomly

All images resized to smaller size to be specific 66x200 as NVidia model suggested.

Then to reduce oeverfitting image augmentation were used sequentially on all images:
    
    1. Random shifts added and steering angle adjusted accordingly;
    2. Reduce/Increase brightness randomly;
    3. Add shadows of different figures and powers randomly;
    4. Dropout were not needed but if dataset is really small then it might be helpful.
       Model learns faster without Dropout therefore less epochs needed.
       By default DO is disabled but you can enable it by passing --dropout 1 argument.
       Tests before showed that Dropouts best worked if placed after 6. and 8. layers, both 0.4

Examples of those images are available in input_images.png

# Optimizer #

Used Adam optimizer because in most scenarios it works best instead of spending week to find
best learning rate sequence. Initialized with 0.0001 LR because previously tests/datasets showed that with
hihger starting learning rate it learns (at start) much slower.

Training time for model to drive second (right) track successfully was 54 s or 27 epochs. It might aswell be less.

### Further challenges ###

1. Only predicting steering angle in real life is not enough. So steering angle combines with throttle/brake is one
   of future challenges. This can be accomplished predicting two values instead of one.

2. If it would be possible to use higher speeds in this environment then we would notice that car wouldn't be able to
   make some track parts without crashing - brakes would be needed but also very usefull would be RNN networks
   with which you could plan further steps ahead and keep information about past.
   
3. How to drive track as fast as possible ? This is challenge where we would need Reinforcement Learning because
   it would be very fustrating to provide enough learning materials for car to drive just perfect any track.
   Reinforcement learning also would provide as capability to learn in any track.
   
4. Parking, avoiding obstacle, drifting, detecting ojbects, localization and more to come.
