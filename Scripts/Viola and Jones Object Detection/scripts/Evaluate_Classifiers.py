import cv2
import matplotlib.pyplot as plt

gt_path = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Cascade_Classifier/TrainIJCNN2013/TrainIJCNN2013/gt.txt"

def find_categorie(categorie):
    if (categorie in [33,34,35,36,37,38,39,40]):
        return 0
    elif (categorie in [6,32,41,42]):
        return 1
    elif (categorie in [0,1,2,3,4,5,7,8,9,10,15]):
        return 2
    elif (categorie in [18,19,20,21,22,23,24,25,26,27,28,29,30,31]):
        return 3
    else:
        return -1

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

classifier_blue_circle = cv2.CascadeClassifier("C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Cascade_Classifier/viola_jones_blue_circle_channel/cascade.xml")
classifier_slashes = cv2.CascadeClassifier("C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Cascade_Classifier/viola_jones_slashes_speed999/cascade.xml")
classifier_red_circle = cv2.CascadeClassifier("C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Cascade_Classifier/viola_jones_red_circle_channel/cascade.xml")
classifier_red_triangle = cv2.CascadeClassifier("C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Cascade_Classifier/viola_jones_red_triangle_channel/cascade.xml")

classifiers = [classifier_blue_circle, classifier_slashes, classifier_red_circle, classifier_red_triangle]

trues = [0,0,0,0]
failes = [0,0,0,0]
ious = [0,0,0,0]
real_numbers = [0,0,0,0]


f = open(gt_path, "r")
for i, line in enumerate(f):
    path, x1_gt, y1_gt, x2_gt, y2_gt, categorie = line[:-1].split(";")

    if(categorie == ''):
        continue
    
    real_cat = find_categorie(int(categorie))

    if(real_cat == -1):
        continue

    #print("categorie", categorie)

    print(i)


    if (real_cat == 1):
        img = cv2.imread("C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Cascade_Classifier/TrainIJCNN2013/TrainIJCNN2013/" + path)
    elif(real_cat == 0):
        img = cv2.imread("C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Cascade_Classifier/TrainIJCNN2013/TrainIJCNN2013/" + path)[:,:,0]
    else:
        img = cv2.imread("C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Cascade_Classifier/TrainIJCNN2013/TrainIJCNN2013/" + path)[:,:,2]
        
    rectangles = classifiers[real_cat].detectMultiScale(img, 1.1, 1)

    #print(rectangles)

    real_numbers[real_cat] +=1

    if(len(rectangles) == 0):
        continue

    #print("rectangles: ", len(rectangles))
    iou_max = -1
    for x1, y1, width, height in rectangles:
        x2 = x1 + width
        y2 = y1 + height
        
        #img = cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 3)
        
        iou = bb_intersection_over_union([x1,y1,x2,y2], [int(x1_gt), int(y1_gt), int(x2_gt), int(y2_gt)])
        #print("iou", iou)
        if(iou > iou_max):
            iou_max = iou

        #if(iou < 0.2):
        #    cv2.imwrite("fail_classifications/" + str(i) + ".png", img[y1:y2, x1:x2])

        print(x1,y1,x2,y2)
        print(x1_gt, y1_gt, x2_gt, y2_gt)
    
    if(iou_max >= 0.5):
        trues[real_cat] += 1
    else:
        failes[real_cat] +=1

    ious[real_cat] += iou_max

    #plt.imshow(img)
    #plt.show()

for i in range(4):
    ious[i] /= (trues[i] + failes[i])
    
print(trues)
print(failes)
print(ious)

prec = [0,0,0,0]

for i in range(4):
    prec[i] += trues[i] / (trues[i] + failes[i])

print(prec)

cats = ["blue_circle", "slashes", "red_circle", "red_triangle"]
for i in range(10):
    print()

for i in range(4):
    print("class: {:<12}, avg_prec: {:.2f}, mean_iou: {:.2f}, trues: {:>3}, failes: {:>3}, numbers: {:>3}".format(cats[i], prec[i], ious[i], trues[i], failes[i], real_numbers[i]))

print()
print()
print()
