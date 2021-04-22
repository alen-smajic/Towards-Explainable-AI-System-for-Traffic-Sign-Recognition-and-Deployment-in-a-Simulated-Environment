import os
import cv2
import random

neg_img_path = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Cascade_Classifier/Negativ_images/"

for num, path in enumerate(os.listdir(neg_img_path)):

    for i in range(1):
        img = cv2.imread(neg_img_path + path)
        #print(img.shape)
        x1 = random.randint(0, img.shape[1] - 33)
        y1 = random.randint(0, img.shape[0] - 33)
        x2 = random.randint(x1+32, img.shape[1])
        y2 = random.randint(y1+32, img.shape[0])

        img = img[y1:y2, x1:x2]

        x = min(img.shape[0], 224)
        y = min(img.shape[1], 224)

        img = cv2.resize(img, (x,y), interpolation=cv2.INTER_NEAREST)

        #print(x1, y1, x2, y2)
        cv2.imwrite("C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Cascade_Classifier/Concept_Whitening_new_class/" + "_" + path, img)
    print(num)
