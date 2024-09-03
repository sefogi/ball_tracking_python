from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import sys

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="video file path")
ap.add_argument("-b", "--buffer", type=int, default=64, help="Max buffer size")
args = vars(ap.parse_args()) if len(sys.argv) > 1 else {"video": None, "buffer": 64}

# Color boundaries in HSV
colors = {
    # "green": ((29, 86, 6), (64, 255, 255)),
    # "blue": ((90, 50, 50), (130, 255, 255)),
    # "red": ((0, 50, 50), (10, 255, 255)),
    # "yellow": ((20, 100, 100), (30, 255, 255)),
    "black": ((0, 0, 0), (179, 255, 30)),
}

# Initialize tracked points
pts = {color: deque(maxlen=args["buffer"]) for color in colors}

# Initialize video capture
if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    if not cv2.VideoCapture(args["video"]).isOpened():
        print(f"Error: could not open video file {args['video']}")
        sys.exit(1)
    vs = cv2.VideoCapture(args["video"])

time.sleep(2.0)

# Main loop
while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    
    if frame is None:
        break
    
    # Resize frame
    frame = imutils.resize(frame, width=1000)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostros
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Convert to HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for color, (lower, upper) in colors.items():
        # Create mask
        mask = cv2.inRange(hsv, lower, upper)
        
        # Apply erosion and dilation
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # Find contours
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
        
        if len(cnts) > 0:
            # Find largest contour
            c = max(cnts, key=cv2.contourArea)
            
            # Find minimum enclosing circle
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            
            # Calculate moments
            M = cv2.moments(c)
            if M["m00"] > 0:
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
                if radius > 10:
                    # Draw circle
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                    
                    # Draw center
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                    
                    # Append center to points
                    pts[color].appendleft(center)
                    
                    # Draw lines
                    for i in range(1, len(pts[color])):
                        if pts[color][i - 1] is None or pts[color][i] is None:
                            continue
                        cv2.line(frame, pts[color][i - 1], pts[color][i], (0, 255, 0), 2)
        
    # Display frame
    cv2.imshow("Frame", frame)
    
    # Check for quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Clean up
if not args.get("video", False):
    vs.stop()
else:
    vs.release()

cv2.destroyAllWindows()
