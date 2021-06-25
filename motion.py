from picamera.array import PiRGBArray 
from picamera import PiCamera 
import time 
import cv2 
import numpy as np 
from gtts import gTTS
import os 




def speak(a):
    tts = gTTS(text=a, lang='en')
    tts.save("audio.mp3")
    os.system("mpg321 audio.mp3")









# Initialize the camera
camera = PiCamera()
 
# Set the camera resolution
camera.resolution = (640, 480)

camera.rotation = 180
 
# Set the number of frames per second
camera.framerate = 30
 
# Generates a 3D RGB array and stores it in rawCapture
raw_capture = PiRGBArray(camera, size=(640, 480))
 
# Create the background subtractor object
# Feel free to modify the history as you see fit.
back_sub = cv2.createBackgroundSubtractorMOG2(history=150,
  varThreshold=25, detectShadows=True)
 
# Wait a certain number of seconds to allow the camera time to warmup
time.sleep(0.1)
 
# Create kernel for morphological operation. You can tweak
# the dimensions of the kernel.
# e.g. instead of 20, 20, you can try 30, 30
kernel = np.ones((20,20),np.uint8)
 
# Capture frames continuously from the camera
for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
     
    # Grab the raw NumPy array representing the image
    image = frame.array
 
    # Convert to foreground mask
    fg_mask = back_sub.apply(image)
     
    # Close gaps using closing
    fg_mask = cv2.morphologyEx(fg_mask,cv2.MORPH_CLOSE,kernel)
       
    # Remove salt and pepper noise with a median filter
    fg_mask = cv2.medianBlur(fg_mask,5)
       
    # If a pixel is less than ##, it is considered black (background). 
    # Otherwise, it is white (foreground). 255 is upper limit.
    # Modify the number after fg_mask as you see fit.
    _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)
 
    # Find the contours of the object inside the binary image
    contours, hierarchy = cv2.findContours(fg_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    areas = [cv2.contourArea(c) for c in contours]
  
    # If there are no countours
    if len(areas) < 1:
  
      # Display the resulting frame
      cv2.imshow('Frame',image)
  
      # Wait for keyPress for 1 millisecond
      key = cv2.waitKey(1) & 0xFF
  
      # Clear the stream in preparation for the next frame
      raw_capture.truncate(0)
     
      # If "q" is pressed on the keyboard, 
      # exit this loop
      if key == ord("q"):
        break
     
      # Go to the top of the for loop
      continue
  
    else:
         
      # Find the largest moving object in the image
         max_index = np.argmax(areas)
    # Draw the bounding box
    cnt = contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
  
    # Draw circle in the center of the bounding box
    x2 = x + int(w/2)
    y2 = y + int(h/2)
    cv2.circle(image,(x2,y2),4,(0,255,0),-1)
  
    # Print the centroid coordinates (we'll use the center of the
    # bounding box) on the image
    text = "x: " + str(x2) + ", y: " + str(y2)
    cv2.putText(image, text, (x2 - 10, y2 - 10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
          
    # Display the resulting frame
    cv2.imshow("Frame",image)
     
    # Wait for keyPress for 1 millisecond
    key = cv2.waitKey(1) & 0xFF
    if max_index == 1:
       print(max_index)
       speak("Sir There is Motion Please Check")
    # Clear the stream in preparation for the next frame
    raw_capture.truncate(0)
  
    # If "q" is pressed on the keyboard, 
    # exit this loop
    if key == ord("q"):
      break
    
# Close down windows
cv2.destroyAllWindows()
