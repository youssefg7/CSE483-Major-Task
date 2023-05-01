import numpy as np
import cv2
from scipy.io import loadmat
import json
import torch
from torchvision import ops
from operator import itemgetter
import tensorflow as tf

# for visualization
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


# CONSTANTS
useCONTRAST = 1 
useCLAHE = 0 
useBLUR = 0 
useCANNY = 0 
cannyTH1 = 150 
cannyTH2 = 200 

# Fn import pics, labels, true boxes
def getPics(chosen_set):
    images = []
    labels = []
    boxes = []
    
    picsFolder_path = "SVHN/" + chosen_set + "/"
    with open(picsFolder_path + 'digitStruct.json') as f:
        data = json.load(f)

# import colored pictures
    for i in range(len(data)):
        image = cv2.imread(picsFolder_path + data[i]['filename'])
        images.append(image)
        temp=[]
        for j in range(len(data[i]['boxes'])):
            temp.append(data[i]['boxes'][j]['label'])
        temp = np.array(temp)
        labels.append(temp)
        boxes.append(data[i]['boxes'])

    print("There are ", len(data), " images in " + chosen_set + " set.")
    images = np.array(images)
    labels = np.array(labels)
    boxes = np.array(boxes)
    return images, labels, boxes

# Fn performs CV techniques on a single picture
def rectanglesModel(img):
    image = img.copy()
    boxes = []
    
    image = cv2.bilateralFilter(image,11,9,9) 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    if(useCONTRAST):
        cv2.convertScaleAbs(image, image)
        
    if(useCLAHE):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        
    bnr = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,1)

    if(useCANNY):
        bnr = cv2.Canny(image, cannyTH1, cannyTH2, 255)
        
    contours, hierarchy = cv2.findContours(bnr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    for i in range(len(contours)):
        boxes.append({})
        x,y,w,h = cv2.boundingRect(contours[i])
        boxes[i]['left'], boxes[i]['top'], boxes[i]['width'], boxes[i]['height'] = x,y,w,h
        twopointsRec = [x,y,x+w,y+h]

    return boxes

# IOU average accuracy test per picture
def iouPicTest(truth, predicted, threshold1=0.5, threshold2=0.5): 
    filtered = []
    boxesTensors = []
    # Check IOU of predicted against all true boxes
    for i in range(len(truth)):
        for j in range(len(predicted)):
            truth_box = torch.tensor(
                [[truth[i]['left'], truth[i]['top'], truth[i]['left'] + truth[i]['width'],
                  truth[i]['top']+truth[i]['height']]], dtype=torch.float)
            predicted_box = torch.tensor(
                [[predicted[j]['left'], predicted[j]['top'], predicted[j]['left']+predicted[j]['width'],
                  predicted[j]['top']+predicted[j]['height']]], dtype=torch.float)
            x = ops.box_iou(truth_box, predicted_box)
    # Append possible true boxes to "filtered" array
            if (x >= threshold1):
                filtered.append([float(x), predicted[j]])
                boxesTensors.append([predicted[j]['left'], predicted[j]['top'],predicted[j]['left']+predicted[j]['width'],predicted[j]['top']+predicted[j]['height']])
    
    scoresTensors = torch.tensor(filtered[:,0])
    boxesTensors = torch.tensor(boxesTensors)
    # Apply Non-maximum suppression to get 0/1 corresponding predicted box for every true box
    selected_indices = tf.image.non_max_suppression(boxesTensors,scoresTensors,15,threshold2)
    selected_boxes = tf.gather(filtered,selected_indices)
    selected_scores = tf.gather(scoresTensors,selected_indices)
    
    # filtered = sorted(filtered, key=itemgetter(0), reverse=True)
    i=0
    acc = 0
    iou = []
    # if len(filtered) > 0:
    #     while i < len(filtered) and filtered[i][0] != 0 :
    #         for j in range(i+1, len(filtered)):
    #             predicted_box1 = torch.tensor(
    #             [[filtered[i][1]['left'], filtered[i][1]['top'], filtered[i][1]['left']+filtered[i][1]['width'],
    #               filtered[i][1]['top']+filtered[i][1]['height']]], dtype=torch.float)
    #             predicted_box2 = torch.tensor(
    #             [[filtered[j][1]['left'], filtered[j][1]['top'], filtered[j][1]['left']+filtered[j][1]['width'],
    #               filtered[j][1]['top']+filtered[j][1]['height']]], dtype=torch.float)
    #             if (ops.box_iou(predicted_box1, predicted_box2)) >= threshold2:
    #                 filtered[j][0]= 0
    #         i+=1
    #         filtered = sorted(filtered, key=itemgetter(0), reverse=True)
    #     cnt=0
        
    # Calculate weighted average IOU accuracy of filtered boxes
    # for i in range(len(filtered)):
    #     if (filtered[i][0]==0):
    #         break
    #     cnt+=1
    #     iou.append(filtered[i][0])
    
    # filtered = filtered[:cnt]
    acc = np.sum(np.array(selected_scores))/len(truth)
    return acc, selected_boxes
    
def showRectangles(image, rectangles, title=""):
    image2 = image.copy()
    for i in rectangles:
        cv2.rectangle(image2, (i['left'], i['top']), (i['left'] +
                      i['width'], i['top']+i['height']), (0, 255, 0), 1)
    plt.figure()
    plt.title(title)
    plt.imshow(image2)

def getPictureIOUAccuracy(image, box):
    image = image.copy()
    predicted_boxes = rectanglesModel(image)
    true_boxes = box.copy()
    accuracy, ret = (iouPicTest(true_boxes,predicted_boxes))
    return accuracy

def getAllIOUAccuracy(images, boxes):
    all_accuracy = []
    for i in range(len(images)):
        all_accuracy.append(getPictureIOUAccuracy(images[i],boxes[i]))
    acc = np.average(np.array(all_accuracy))
    return acc