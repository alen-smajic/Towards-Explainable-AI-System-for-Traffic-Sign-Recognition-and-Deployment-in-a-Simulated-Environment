import os
import shutil
import pandas as pd

concept_test_dataset = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Concept_whitening/Dataset/concept_test/"
concept_train_dataset = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Concept_whitening/Dataset/concept_train/"

rebalanced_train_dataset = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Concept_whitening/rebalanced_data/"

original_train_dataset = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Concept_whitening/GTSRB/Train/"
original_train_labels = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Concept_whitening/GTSRB/Train.csv"

original_test_dataset = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Concept_whitening/GTSRB/Test/"
original_test_labels = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Concept_whitening/GTSRB/Test.csv"

concepts_dict = {}
concepts_dict["red"] = [0,1,2,3,4,5,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
concepts_dict["blue"] = [33,34,35,36,37,38,39,40]
concepts_dict["circle"] = [0,1,2,3,4,5,6,7,8,9,10,15,16,17,32,33,34,35,36,37,38,39,40,41,42]
concepts_dict["triangle"] = [11,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
concepts_dict["slashes"] = [6,32,41,42]
concepts_dict["numbers"] = [0,1,2,3,4,5,6,7,8]
concepts_dict["vehicle"] = [9,10,16,23,41,42,29]
concepts_dict["humans"] = [25,27,28]
concepts_dict["curve"] = [19,20,21]

concept_size = 2000

def create_concept_train():   
    for concept in concepts_dict:
        if not os.path.exists(concept_train_dataset + concept):
            os.makedirs(concept_train_dataset + concept)
            
        if not os.path.exists(concept_train_dataset + concept + "/" + concept):
            os.makedirs(concept_train_dataset + concept + "/" + concept)

        class_number = concept_size // len(concepts_dict[concept])
        
        for number in concepts_dict[concept]:
            num = 0
            img_folder = os.listdir(rebalanced_train_dataset + str(number))

            for image in img_folder:
                if(num == class_number):
                    break
                
                shutil.copy(rebalanced_train_dataset + str(number) + "/" + str(image), concept_train_dataset + concept + "/" + concept)
                num += 1
                
             
"""                
def create_concept_test():
    for concept in concepts_dict:
        if not os.path.exists(concept_test_dataset + concept):
            os.makedirs(concept_test_dataset + concept)
            
    df = pd.read_csv(original_test_labels)
    
    for i in range(len(df['ClassId'])):
        class_id = df["ClassId"][i]
        path = df["Path"][i]
        concepts = ""
        
        for concept in concepts_dict:
            if class_id in concepts_dict[concept]:
                concepts += concept
                
        if concepts == "":
            continue
        
        file_name = ""
        for c in concepts:
            file_name += str(c)           
            
        shutil.copy(original_test_dataset + path[5:], concept_test_dataset + concepts + "/" + (",".join(concepts)) + "_" + str(i) + ".png")
"""

def create_concept_test():
    for concept in concepts_dict:
        if not os.path.exists(concept_test_dataset + concept):
            os.makedirs(concept_test_dataset + concept)
            
    df = pd.read_csv(original_test_labels)
    
    for i in range(len(df['ClassId'])):
        class_id = df["ClassId"][i]
        path = df["Path"][i]
        concepts = ""
        
        for concept in concepts_dict:
            if class_id in concepts_dict[concept]:
                #concepts += concept
                #print(concept_test_dataset + concept + "/" + str(i) + ".png")
                #print(original_test_dataset + path[5:])
                shutil.copy(original_test_dataset + path[5:], concept_test_dataset + concept + "/" + str(i) + ".png")
                
        #if concepts == "":
        #    continue
        
        #file_name = ""
        #for c in concepts:
        #    file_name += str(c)           
            
        #shutil.copy(original_test_dataset + path[5:], concept_test_dataset + concepts + "/" + (",".join(concepts)) + "_" + str(i) + ".png")
        
#create_concept_test()
create_concept_train()
