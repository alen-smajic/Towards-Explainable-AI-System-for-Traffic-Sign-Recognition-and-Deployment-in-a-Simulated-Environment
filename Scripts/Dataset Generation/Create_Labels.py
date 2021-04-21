import numpy as np
import cv2
import argparse
import json
import os


ap = argparse.ArgumentParser()
ap.add_argument("-tp", "--target_path", required=True, 
                help="Target path where the dataset folder is located")
ap.add_argument("-bf", "--box_format", default="corner", 
                help="Bounding box format. Possible values: corner, middle")
args = ap.parse_args()

    
# Threshold for the pink color in HSV color space
lower_threshold = np.array([150, 100, 100])
upper_threshold = np.array([165, 255, 255])

# Will be used to store the data for the .json format
data = {}


# Loops over all target images from the target folder
for sample_name in os.listdir(args.target_path + "/Target Data"): 
    
    # Reads the target image
    img = cv2.imread(args.target_path + "/Target Data/" + sample_name) 
    
    # Converts the color space from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Thresholds the image color space
    mask = cv2.inRange(hsv, lower_threshold, upper_threshold)
    
    # Finds contours around the thresholded color objects
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Creates the json element for the target image
    data[sample_name] = []


    # Loops over all contours objects (traffic signs), found in the target image
    for contour in contours:
        
        # Extracts the contour coordinates
        # x, y are the pixel coordinates of the upper left corner for the 
        # bounding box
        # w, h are the width and height values in pixels for the bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Transforms the coordinates into the corner format  
        # x_1, y_1 coordinates of the upper left corner of the bounding box
        # x_2, y_2 coordinates of the lower right corner of the bounding box
        if(args.box_format == "corner"): 
            x_1 = x
            y_1 = y
            x_2 = x + w 
            y_2 = y + h        
            # Stores the coordinates inside the json element
            data[sample_name].append({
                'x1' : x_1,
                'y1' : y_1,
                'x2' : x_2,
                'y2' : y_2
                })
            print(sample_name + ' ' + 'x1:' + str(x_1) + ' ' + 'y1:' + str(y_1) + ' ' +
                  'x2:' + str(x_2) + ' ' + 'y2:' + str(y_2))
             
        # Transforms the coordinates into the middle format
        # x, y coordinates of the center point of the bounding box
        # w, h are the width and height values in pixels for the bounding box
        elif(args.box_format == "middle"):
            x = x + (w/2)
            y = y + (h/2)   
            # Stores the coordinates inside the json element
            data[sample_name].append({
                'x' : x,
                'y' : y,
                'w' : w,
                'h' : h
                }) 
            print(sample_name + ' ' + 'x:' + str(x) + ' ' + 'y:' + str(y) + ' ' +
                  'w:' + str(w) + ' ' + 'h:' + str(h))
 
    
# Stores the json data inside a json file
with open(args.target_path + "/labels.json", 'w') as outfile:
    json.dump(data, outfile)
