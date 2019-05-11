from imutils.video import VideoStream
import cv2
import numpy as np
import time
import pandas as pd
import RPi.GPIO as GPIO
import os

save =0
angle = 11
        
#output pins
servo_control = 13
motorone_1 = 17
motorone_2 = 27
motortwo_1 = 22
motortwo_2 = 23
en1 = 19
en2 = 26
#servo 5V pin 2 GND pin 9
#motor driver 5v pin4 GND pin 6

#intializing GPIO pins
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_control, GPIO.OUT)
GPIO.setup(motorone_1, GPIO.OUT)
GPIO.setup(motorone_2, GPIO.OUT)
GPIO.setup(motortwo_1, GPIO.OUT)
GPIO.setup(motortwo_2, GPIO.OUT)
GPIO.setup(en1, GPIO.OUT)
GPIO.setup(en2, GPIO.OUT)

#initializing dc motors
p1 = GPIO.PWM(en1, 100)
p2 = GPIO.PWM(en2, 100)
p1.start(30)
p2.start(30)

#input remote control pins
left = 10
up = 9
right = 11

GPIO.setup(up, GPIO.IN)
GPIO.setup(left, GPIO.IN)
GPIO.setup(right, GPIO.IN)

#initializing the servo motor
p = GPIO.PWM(servo_control, 100)
p.start(0)
p.ChangeDutyCycle(angle)

#intializing pi camera
vs = VideoStream(usePiCamera = True, resolution = (480,360)).start()
time.sleep(2.0)

i=0
img = []
ang = []
output = []
#create folder to save data
folder = input("Enter folder name : ")
os.mkdir("/home/pi/tensorflow-env/Project/" + folder)
op = ''

try :
    
    while True :
        #read and display video frame
        image = vs.read()
        cv2.imshow('live',image)

        save = 0
        
        #move forward if button is pressed
        if(GPIO.input(up)) :
            op = 'up'                         
            GPIO.output((motorone_1,motortwo_1),0)
            GPIO.output((motorone_2,motortwo_2),1)
            time.sleep(0.2)
            GPIO.output((motorone_2,motortwo_2),0)
            save = 1
            
        #turn left if button is pressed
        if(GPIO.input(left)) :
            save = 1
            while(GPIO.input(left)) :
                op = 'left'
                angle = angle - 0.000005
                if(angle<=9) :
                    p.ChangeDutyCycle(0)
                    angle=9
                p.ChangeDutyCycle(angle)
            p.ChangeDutyCycle(0)
            
        #turn right if button is pressed
        if(GPIO.input(right)) :
            save = 1
            while(GPIO.input(right)) :
                op = 'right'
                angle = angle + 0.000005
                if(angle>=12.5) :
                    p.ChangeDutyCycle(0)
                    angle=12.5
                p.ChangeDutyCycle(angle)
            p.ChangeDutyCycle(0)
        print(op)
        
        #add data to list when button is pressed
        if(save==1) :
            cv2.imwrite(str(folder) + '/' + str(i) + '.jpg', image)
            i = i+1
            img.append(str(folder) + '/' + str(i) + '.jpg')
            ang.append(angle)
            output.append(op)

        key = cv2.waitKey(1) & 0xFF
        if key==ord('q') :
            break
        
        
except KeyboardInterrupt :
    GPIO.cleanup()

#turn off GPIO and camera
GPIO.cleanup()
cv2.destroyAllWindows()
vs.stop()
print("camera off")

image = np.array(img)
image = image.reshape(image.shape[0],1)
angle = np.array(ang)
angle = angle.reshape(angle.shape[0],1)
output = np.array(output)
output = output.reshape(output.shape[0],1)
print("Processing\n")

temp = np.concatenate((image,angle,output), axis = 1)

#save the data as a csv file
print("writing to csv")
df = pd.DataFrame(temp,columns = ['Image', 'Angle', 'ouptut'])
df.to_csv('dataset_' + str(folder) + '.csv')
print("done")

