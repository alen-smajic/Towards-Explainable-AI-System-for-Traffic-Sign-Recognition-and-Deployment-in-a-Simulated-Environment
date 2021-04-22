#file:///C:/Users/Pasca/Downloads/HoubenEtAl_GTSDB.pdf
#https://dikshitkathuria1803.medium.com/training-your-own-cascade-classifier-detector-opencv-9ea6055242c2

import os
import shutil
import random

categorie_dict = dict()

categorie_dict["red_circle"] = ["00","01","02","03","04","05","07","08","09","10","15"]
categorie_dict["red_triangle"] = ["18","19","20","21","22","23","24","25","26","27","28","29","30","31"]
categorie_dict["slashes"] = ["06","32","41","42"]
categorie_dict["blue_circle"] = ["33","34","35","36","37","38","39","40"]

data_folder = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Cascade_Classifier/TrainIJCNN2013/TrainIJCNN2013/"
gt_file = "gt.txt"


transformed_folder = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/transformed_rebalanced_data/"
output_folder = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Cascade_Classifier/"
pos_count = 1000


def calc_negatives():
    for cat in categorie_dict:
        #negativs:
        print(cat)
        f = open(data_folder + gt_file, "r")
        if not os.path.exists(output_folder + "neg_" + cat):
                os.makedirs(output_folder + "neg_" + cat)
            
        files = list(set([line.split(";")[0] for line in f]))
        print(len(files))
        files.sort()
    
        for fi in files:
            print(fi)
    
        f = open(data_folder + gt_file, "r")
        for line in f:
            img, x1, y1, x2, y2, categorie = line[:-1].split(";")
        
            if(categorie == ""):
                continue

            if(int(categorie) in categorie_dict[cat]):
                if(img in files):
                    files.remove(img)

        print(len(files))

        for img in files:
            shutil.copy(data_folder + img, output_folder + "neg_" + cat + "/" + img)

def calc_positives():
    for cat in categorie_dict:
        #positivs:
        if not os.path.exists(output_folder + "pos_" + cat + "_new"):
                os.makedirs(output_folder + "pos_" + cat + "_new")

        number = pos_count // len(categorie_dict[cat])
        print(number)
    
        for num in categorie_dict[cat]:
            #imgs = os.listdir(transformed_folder + str(num))
            imgs = os.listdir(data_folder + num)
            #for i in range(number):
            for i in range(len(os.listdir(data_folder + num))):
                #img = random.choice(imgs)
                img = imgs[i]
                #imgs.remove(img)
                #print(transformed_folder + str(num) + "/" + img)
                print(data_folder + num + "/" + img)
                #print(output_folder + "pos_" + cat + "/" + img)
                
                #shutil.copy(transformed_folder + str(num) + "/" + img, output_folder + "pos_" + cat + "_new/" + img)
                shutil.copy(data_folder + num + "/" + img, output_folder + "pos_" + cat + "_new/" + img)

#calc_negatives()
calc_positives()            
