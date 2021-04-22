import os
import shutil
import pandas as pd


original_train_dataset = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Concept_whitening/GTSRB/Train/"
original_train_labels = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Concept_whitening/GTSRB/Train.csv"
original_test_dataset = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Concept_whitening/GTSRB/Test/"
original_test_labels = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Concept_whitening/GTSRB/Test.csv"


final_test_dataset = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Concept_whitening/Dataset/test/"
final_train_dataset = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Concept_whitening/Dataset/train/"
final_val_dataset = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Concept_whitening/Dataset/val/"
concept_test_dataset = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Concept_whitening/Dataset/concept_test/"
concept_train_dataset = "C:/Users/Pasca/OneDrive/Dokumente/Master_3.Semester/Master_3.Semester/SYSL/Concept_whitening/Dataset/concept_train/"


concepts_dict = {}
concepts_dict["red"] = [0,1,2,3,4,5,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
concepts_dict["blue"] = [33,34,35,36,37,38,39,40]
concepts_dict["circle"] = [0,1,2,3,4,5,6,7,8,9,10,15,16,17,32,33,34,35,36,37,38,39,40,41,42]
concepts_dict["triangle"] = [11,18,19,20,21,22,23,24,25,26,27,28,29,30,31]


def create_concept_train():   
    for concept in concepts_dict:
        if not os.path.exists(concept_train_dataset + concept):
            os.makedirs(concept_train_dataset + concept)
            
        if not os.path.exists(concept_train_dataset + concept + "/" + concept):
            os.makedirs(concept_train_dataset + concept + "/" + concept)
        for number in concepts_dict[concept]:
            img_folder = os.listdir(original_train_dataset + str(number))
            
            for image in img_folder:
                shutil.copy(original_train_dataset + str(number) + "/" + str(image), concept_train_dataset + concept + "/" + concept)
                
                
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


def create_train_dataset():
    img_folder = os.listdir(original_train_dataset) 
    counter = 0

    for subfolder in img_folder:
        img_subfolder = os.listdir(original_train_dataset + str(subfolder))
        
        for img in img_subfolder:
            shutil.copy(original_train_dataset + str(subfolder) + "/" + str(img), final_train_dataset + str(subfolder) + "_" + str(counter) + ".png")
            counter += 1
    
    
def create_test_dataset():
    df = pd.read_csv(original_test_labels)
    
    for i in range(len(df['ClassId'])):
        class_id = df["ClassId"][i]
        path = df["Path"][i]
        
        if not os.path.exists(final_test_dataset + str(class_id)):
            os.makedirs(final_test_dataset + str(class_id))
        
        shutil.copy(original_test_dataset + path[5:], final_test_dataset + str(class_id) + "/" + str(class_id) + "_" + str(i) + ".png")
        
        
def create_val_dataset():
    df = pd.read_csv(original_test_labels)
    
    for i in range(len(df['ClassId'])):
        class_id = df["ClassId"][i]
        path = df["Path"][i]
        
        shutil.copy(original_test_dataset + path[5:], final_val_dataset + str(class_id) + "_" + str(i) + ".png")
        
#create_test_dataset()
#create_concept_train()
create_concept_test()
#create_train_dataset()
#create_test_dataset()
#create_val_dataset()
