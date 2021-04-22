import os
import random
import cv2

dataset_file = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Concept_whitening/selfmade_dataset_test/"

for concept in os.listdir(dataset_file):
    for img_path in os.listdir(dataset_file + concept):
        #print(number)
        img = cv2.imread(dataset_file + concept + "/" + img_path)
        img = cv2.flip(img, 0)

        out_path = dataset_file + concept + "/flip_" + img_path
        print(out_path)
        cv2.imwrite(out_path, img)

