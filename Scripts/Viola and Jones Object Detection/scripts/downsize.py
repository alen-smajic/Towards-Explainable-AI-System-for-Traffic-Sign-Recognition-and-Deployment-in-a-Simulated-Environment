import os
import cv2
import random

neg_img_path = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Cascade_Classifier/neg_fail_class/"

for num, path in enumerate(os.listdir(neg_img_path)):

    img = cv2.imread(neg_img_path + path)

    x = min(img.shape[0], 224)
    y = min(img.shape[1], 224)

    img = cv2.resize(img, (x,y), interpolation=cv2.INTER_NEAREST)

    #print(x1, y1, x2, y2)
    cv2.imwrite("C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Cascade_Classifier/neg_fail_class_small/" + path, img)
    print(num)
