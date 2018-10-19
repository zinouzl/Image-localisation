# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 00:07:42 2017

@author: Latioui Zine Eddine
"""
import itertools
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')












flist = []
klist = []
#imglist = []
#label_image = ["13","13","13","13","13","13","13","13","13","13","14","14","14","14","14","14","14","14","14","14","15","15","15","15","15","16","16","16","23","23","23","23","23","23","23","23","23","23","23","23","23","24","24","24","24","24","24","24","24","24","24","24","24","24","24","25","25","25","25","25","25","25","25","25","25","25","25","25","25","26","26","26","26","26","26","26","26","26","33","33","33","33","33","33","33","33","33","33","33","33","34","34","34","34","34","34","34","34","34","34","34","34","43","43","43","43","43","43","43","43","43","43","43","43","44","44","44","44","44","44","44","44","44","44","44","44","45","45","45","45","45","45","45","45","45","45","45","45","46","46","46","46","46","46","46","46","46","46","46","46","46-26","46-26","46-26","46-26","46-26","46-26","46-26","53","53","53","53","53","53","53","53","53","54","54","54","54","54","54","54","54","54","54","54","54","55","55","55","55","55","55","55","55","55","55","55","55","56","56","56","56","56","56","56","56","56","56","56","56"]
#app_label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18]
app_label1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7]
testlabel = ['23','24','26','43','46','46-26','54','56']
#apprlabel = ["13","14","15","16","23","24","25","26","33","34","43","44","45","46","46-24","53","54","55","56"]
mattest = [0] * 19
pre_label = ['23','23','23','24','24','24','26','26','26','26','43','43','43','46','46','46','46','46-26','46-26','46-26','46-26','46-26','54','54','54','54','54','56','56','56','56']
test_label = []
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
for p in pathlib.Path('./DATA/SIFT_descriptors').iterdir():
    if p.is_dir():
        "print(p)"
        flist.append(p)
for t in flist: 
    print(t)      
    for p in pathlib.Path(t).iterdir():
        if p.is_file():
            imagedes = np.load(p)
            cluster = np.array([imagedes])
            flann.add(cluster)
            
print('training...')            
flann.train()
print('end of training.')
print('testing...')
for p in pathlib.Path('./Campus(Test images)').iterdir():
    if p.is_dir():
        "print(p)"
        klist.append(p)
for t in klist: 
    print(t)      
    for p in pathlib.Path(t).iterdir():
        if p.is_file():
            print(p)
            img = cv2.imread(str(p))
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SIFT_create()
            kp,des= sift.detectAndCompute(gray,None)
            matches = flann.knnMatch(des,k=2)
            good = []
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.75*n.distance:
                    good.append(m.imgIdx)
            for i in range(len(good)):
                mattest[app_label1[good[i]]] +=1 
                
            index = np.argmax(mattest)
            mattest = [0] * 19
            test_label.append(testlabel[index])
            print('predected Rock: '+test_label[-1])
print(pre_label)
print(test_label)   
count = 0         
cnf_matrix = confusion_matrix(pre_label, test_label)

np.set_printoptions(precision=2)
for ma in range(len(pre_label)):
    if (pre_label[ma]==test_label[ma]):
        count+=1
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=testlabel,
                      title='Confusion matrix, without normalization')            
print('recognition rate : '+str(count/len(pre_label)*100) )




