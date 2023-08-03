import cv2
import numpy as np

import os
# Load the dataset from the specified directory and create a dictionary 'dataset'
dataset = []


content = [item for item in os.listdir('training_dataset')]
for i in content:
    contents = [os.path.join(i, item) for item in os.listdir(r'training_dataset\\' + i)]
    
    mini = {}
    for k in range(int(len(contents))):
        if k % 2 == 0:
            if contents[k][len(contents[k]) - 4:len(contents[k])] == '.jpg':
                mini[contents[k]] = contents[k + 1]
    dataset.append(mini)
# print(1)
# Merge the dictionary elements into a single dictionary 'dataset1'
dataset1 = {}
for i in dataset:
    for j, k in i.items():
        dataset1[j] = k

# Load the test data from 'imgList.txt' and save it as 'test_list.json'
with open(r'imgList.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
test_data = []
for i in lines:
    test_data.append(i[:len(i) - 1])

# print(2)
# Load training data and labels
def load_data(dataset):
    images = []
    labels = []
   
    for img, txt in dataset.items():
        img = r'training_dataset\\' + img
        txt = r'training_dataset\\' + txt
        with open(txt, 'r') as f:
            data = f.read()
        datalist = [float(k) for k in data.split()]
        if len(datalist) == 0:
            continue
        else:
            datalist = datalist[:10]
            labels.append(np.array(datalist, dtype=np.int32))
            image = cv2.imread(img)
            image = cv2.resize(image, (160, 160))
            image = image / 255.0
            images.append(image)
    return np.array(images,dtype=np.float32), np.array(labels, dtype=np.int32)


# Load test data
def load_test_data(dataset):
    images = []
    
    for img in dataset:
        img = r'test_dataset\\' + img
        
        try:
            image = cv2.imread(img)
            image = cv2.resize(image, (160, 160))
            image = image / 255.0
            images.append(image)
            
        except cv2.error:
            continue
    return np.array(images)



