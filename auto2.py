import tensorflow as tf
import numpy as np
import cv2
from imutils.video import VideoStream
import time
import pandas as pd
import RPi.GPIO as GPIO
import os
import threading

print("Setting up GPIOs\n")
        
#output pins connected to motors
servo_control = 13
motorone_1 = 17
motorone_2 = 27
motortwo_1 = 22
motortwo_2 = 23
en1 = 19
en2 = 26
#servo 5V pin 2 GND pin 9
#motor driver 5v pin4 GND pin 6
angle=11

#intializing the pins
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_control, GPIO.OUT)
GPIO.setup(motorone_1, GPIO.OUT)
GPIO.setup(motorone_2, GPIO.OUT)
GPIO.setup(motortwo_1, GPIO.OUT)
GPIO.setup(motortwo_2, GPIO.OUT)
GPIO.setup(en1, GPIO.OUT)
GPIO.setup(en2, GPIO.OUT)

#initializing DC motors thorugh motor driver IC
p1 = GPIO.PWM(en1, 100)
p2 = GPIO.PWM(en2, 100)
p1.start(15)
p2.start(15)
GPIO.output((motorone_1,motortwo_1,motorone_2,motortwo_2),0)

#initializing the servo motor
p = GPIO.PWM(servo_control, 100)
p.start(0)
p.ChangeDutyCycle(angle)

print("GPIOs ready")


#function for moving the car forward
#running it as a separate thread
def forward() :
    print('up')
    GPIO.output((motorone_1,motortwo_1),0)
    while True :
        GPIO.output((motorone_2,motortwo_2),1)
        global stop_thread
        if stop_thread :
            break
    GPIO.output((motorone_2,motortwo_2),0)

#function for turning left
def left(angle) :
    
    angle = angle - 0.1
    if(angle<=9.6) :
        angle = 9.6
    print('left')
    p.ChangeDutyCycle(angle)
    return angle

#function for turning right
def right(angle) :
    
    angle = angle + 0.1
    if(angle>=12.5) :
        angle = 12.5
    print('right')
    p.ChangeDutyCycle(angle)
    return angle


print("Loading model")
#define the architecture
model1 = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32,(3,3),input_shape=(160,320,1),activation=tf.nn.relu),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),  
  tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, kernel_initializer = 'uniform', activation=tf.nn.relu),
  tf.keras.layers.Dense(256, kernel_initializer = 'uniform', activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Dense(3, kernel_initializer = 'uniform', activation=tf.nn.softmax)
])

#load the saved model
model1.load_weights('mod98.h5')

print("model ready!")

#initializing pi camera
vs = VideoStream(usePiCamera = True).start()
time.sleep(2.0)

#starting the thread
stop_thread = False
t1 = threading.Thread(target = forward)
t1.start()

try :
    angle = 11
    print('starting video')
    
    while True :
        img = vs.read() #read video frames
        cv2.imshow('op',img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grayscale
        image = img[80:,:] #cut top 80 pixels

        image = image[np.newaxis,:,:,np.newaxis] #add extra dimension 
        image = image/255.0 #normalize the pixel values
        
        direction = np.argmax(model1.predict(image)) #predict the direction
        
        #updating the angle of servo motor
        if direction==1 :
            angle = right(angle)
        else :
            angle = left(angle)
        
        key = cv2.waitKey(1) & 0xFF
        if key==ord('q') :
            stop_thread = True
            break
    t1.join()
    print('thread killed')
    vs.stop()
    cv2.destroyAllWindows()
    GPIO.cleanup()
    
except KeyboardInterrupt :
    stop_thread = True
    t1.join
    print('thread killed')
    vs.stop()
    cv2.destroyAllWindows()
    GPIO.cleanup()
