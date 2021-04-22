import os
import random
import cv2

dataset_file = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Concept_whitening/Dataset/train/"

output_dir = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Concept_whitening/"

# required number of images for concept
class_size = 1000

#rotation angle for double images
angle = 20

if not os.path.exists(output_dir + "rebalanced_data"):
            os.makedirs(output_dir + "rebalanced_data")

for categorie in os.listdir(dataset_file):
    if not os.path.exists(output_dir + "rebalanced_data/" + categorie):
        os.makedirs(output_dir + "rebalanced_data/" + categorie)
            
    first_round = True

    number = 0
    while(number < 1000):
        
        for img_path in os.listdir(dataset_file + categorie):
            #print(number)
            img = cv2.imread(dataset_file + categorie + "/" + img_path)
            if not first_round :
                cur_angle = random.randint(-angle,angle)
                (h, w) = img.shape[:2]
                #print(h,w)
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, cur_angle, 1.0)
                #rotate image
                img = cv2.warpAffine(img, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

            out_path = output_dir + "rebalanced_data/" + categorie + "/" + img_path if first_round else output_dir + "rebalanced_data/" + categorie + "/" + str(cur_angle) + "_" + img_path
            print(out_path)
            cv2.imwrite(out_path, img)
            number += 1
            if(number == 1000):
                break

        first_round = False
            
        
