import os

folder = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Cascade_Classifier/"
pos_data = "pos_red_triangle_channel"

f = open(folder + pos_data + ".info", "w")

for img in os.listdir(folder + pos_data):
    #print(img + " 1 0 0 32 32 \n")

    f.write(pos_data + "/" + img + " 1 0 0 32 32 \n")

f.close()
