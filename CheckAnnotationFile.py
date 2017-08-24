from __future__ import print_function
import xml.etree.ElementTree as ET
import pickle
import os
import json
from operator import itemgetter
from os import walk
import numpy as np
import cv2
import matplotlib.pyplot as plt
import AnnotationHelper as AnnHelp



def checkWithJson():
    jsonPath = "../HandDataset/Yolo_Test/out/json/"
    ImagePath = "../HandDataset/Yolo_Test/out/img/"

    annHelp = AnnHelp.AnnotationHelper()
    f = []
    # getlist file name in path
    for (dirpath, dirnames, filenames) in walk(jsonPath):
        f.extend(filenames)

    for fileInpath in f:
        # extract filename and exetypy file
        filename, extfile = fileInpath.split(os.extsep)
        print(filename)
        predic = annHelp.convert_json(jsonPath + filename + ".json")
        image = cv2.imread(ImagePath + filename + ".jpg")

        if not len(predic) == 0:
            bbplot = np.asarray(predic[0][1], dtype=np.int)
            cv2.rectangle(image, (bbplot[0], bbplot[1]), (bbplot[2], bbplot[3]), (0, 0, 255), 5)
        cv2.imshow("CheckImage", image)
        cv2.waitKey(100)

def checkWithAnnotation():
    AnnPath = "../HandDataset/Annotation_Hand_25_Class/xml/"
    ImagePath = "../HandDataset/Annotation_Hand_25_Class/imageresize/"

    annHelp = AnnHelp.AnnotationHelper()
    f = []
    # getlist file name in path
    for (dirpath, dirnames, filenames) in walk(AnnPath):
        f.extend(filenames)

    for fileInpath in f:
        # extract filename and exetypy file
        filename, extfile = fileInpath.split(os.extsep)
        print(filename)
        truth = annHelp.convert_annotation(AnnPath + filename + ".xml")
        image = cv2.imread(ImagePath + filename + ".jpg")
        bbplot = np.asarray(truth[0][1], dtype=np.int)
        cv2.rectangle(image, (bbplot[0], bbplot[1]), (bbplot[2], bbplot[3]), (255, 255, 0), 2)
        cv2.imshow("CheckImage", image)
        cv2.waitKey(100)



if __name__ == "__main__":
    checkWithJson()