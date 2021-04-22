import numpy as np
import cv2
import random
import os

path = "./Concept_Dataset_script_generated/"

def make_geometric_objects(path, number, geom):

    path = path + geom + "/" 
    if not os.path.exists(path):
        os.makedirs(path)
        
    for i in range(number):
        image = np.ones((240,240,3), np.uint8) #*255

        #[red, green, blue]
        bg_colors = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]

        for index in range(3):
            image[:,:,index] = image[:,:,index] * bg_colors[index]

        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)

        #color of object
        color = (r, g, b) 

        #possible thickness of object lines in pixel, -1 is full filled
        thickness_list = [[-1],[1,2,3,4,5]] 

        #choose with probability of 50% full filled ord not, when not uniform distributed over [1,5]
        thickness = random.choice(thickness_list[random.randint(0,1)])
        
        if(geom == "circle"):
            
            x = random.randint(20,220)
            y = random.randint(20,220)

            center_coordinates = (x, y)

            max_radius = min(240-x, 240-y, x, y)

            radius = random.randint(20, max_radius)

            image = cv2.circle(image, center_coordinates, radius, color, thickness)

        elif(geom == "triangle"):

            point1 = (random.randint(0,240), random.randint(0,240))
            point2 = (random.randint(0,240), random.randint(0,240))
            point3 = (random.randint(0,240), random.randint(0,240))

            triangle_cnt = np.array([point1, point2, point3])  

            image = cv2.drawContours(image, [triangle_cnt], 0, color , thickness)
        

        #cv2.imshow("img", image)
        cv2.imwrite(path + geom + "_" + str(i) + ".png", image)

def make_color_images(path, number, color):

    path = path + color + "/"

    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(number):

        image = np.zeros((240,240,3), np.uint8)

        color_map = np.random.randint(0, 255, size=(240,240))
        #print(color_map)

        channel = 0 if color == "blue" else 2

        image[:,:,channel] = color_map

        #cv2.imshow("img", image)
        cv2.imwrite(path + color + "_" + str(i) + ".png", image)
    
    

    
#create circles
print("create circles")
make_geometric_objects(path, 200, "circle")
print("create triangles")
make_geometric_objects(path, 200, "triangle")
print("create reds")
make_color_images(path, 200, "red")
print("create blues")
make_color_images(path, 200, "blue")



