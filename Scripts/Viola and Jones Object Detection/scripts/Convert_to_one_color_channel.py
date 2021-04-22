import cv2
import os

dataset_dir = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Cascade_Classifier/pos_red_triangle/"
output_dir = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Cascade_Classifier/pos_red_triangle_channel/"
channel = 2 # 0:blue, 1:green, 2:red
img_nums = len(os.listdir(dataset_dir))
for i, path in enumerate(os.listdir(dataset_dir)):
    print(i, "/", img_nums)
    color_channel = cv2.imread(dataset_dir + path)[:,:,channel]
    #print(color_channel.shape)
    cv2.imwrite(output_dir + path, color_channel)
