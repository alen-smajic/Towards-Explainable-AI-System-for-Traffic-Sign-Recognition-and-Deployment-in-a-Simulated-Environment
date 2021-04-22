import os

folder = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Cascade_Classifier/"

neg_data = "neg_red_triangle_channel++"

f = open(folder + neg_data + "_bg.txt", "w")

for img in os.listdir(folder + neg_data):

    f.write(neg_data + "/" + img + "\n")

f.close()
