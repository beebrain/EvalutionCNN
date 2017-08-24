from __future__ import print_function

import xml.etree.ElementTree as ET
import pickle
import os
import json
import cv2
from operator import itemgetter
from os import walk
import numpy as np
np.set_printoptions(linewidth =1000)
import matplotlib.pyplot as plt
import AnnotationHelper as AnnHelp


class Evaluate:
    def __init__(self):
        self.AnnPath = "../HandDataset/Annotation_Hand_validateAndUnsign/xml/"
        self.jsonPath = "../HandDataset/Result_test/YoLo_Test/Yolo_26Test/out/"
        # jsonPath = "../CNNFinger/All_datasetModel/26_3Layer/CNN_model_3_8_8_8/out/"
        # jsonPath = "../CNNFinger/All_datasetModel/HOG_26_2048/ANNHog/out/"
        self.ImagePath = "../HandDataset/Annotation_Hand_validateAndUnsign/imageresize/"
        self.font = cv2.FONT_HERSHEY_PLAIN

    def divide_zero_is_zero(self,a, b):
        return float(a)/float(b) if b != 0 else 0

    def EvaluateMAP(self):
        totalClass = 26
        AccAP = 0
        annHelp = AnnHelp.AnnotationHelper()
        f = []
        for (dirpath, dirnames, filenames) in walk(self.AnnPath):
            f.extend(filenames)

        # fig = plt.figure(1)
        # ax1 = fig.add_subplot(1, 1, 1)

        Missrate = []
        FPI = []
        Aptable = np.zeros((3,totalClass))
        evaluateFile = 'Evaluate_Missrate.npy'
        ForceCal = True
        if not ForceCal and os.path.exists(evaluateFile):
            # Just change the True to false to force re-training
            print('Loading Evaluate')
            Missrate = np.load("Evaluate_Missrate.npy")
            FPI = np.load("Evaluate_FPI.npy")
        else:
            # start Evaluate
            thresholdnum = 0.1
            indexImage = 0
            for file in f:
                filename, extfile = file.split(os.extsep)
                print(filename)
                indexImage += 1
                # print(extfile)
                truth = annHelp.convert_annotation(self.AnnPath + filename + ".xml")
                predic = annHelp.convert_json(self.jsonPath + filename + ".json", threshold=thresholdnum)
                imagefile = self.ImagePath + filename + ".jpg"
                image = cv2.imread(imagefile)
                # image,TP, FN, FP = annHelp.eveluate(truth, predic, image, thresholdIOU=0.6)
                matrixAP, AP, mAP = annHelp.eveluateOrderIOU(truth, predic, image)
                truthClass = truth[0][0]
                truthClass = truthClass.replace("Hand","")
                print(truthClass)
                # save ApTable each class
                Aptable[0,int(truthClass)-1] +=1
                Aptable[1, int(truthClass) - 1] += AP
                Aptable[2, int(truthClass) - 1] =round(float(Aptable[1, int(truthClass) - 1]/Aptable[0,int(truthClass)-1])*100,2)
                print(Aptable[1, int(truthClass) - 1]/Aptable[0,int(truthClass)-1])
                print(Aptable)
                AccAP += AP

                ###################################  Print reSult on Image ###############################
                textResult = "       AP\n"
                textResult += "AccAP=" + str(AccAP) + "\n"
                textResult += "Cur= " + str(AP) + "\n"
                textResult += "TotalImg= " + str(indexImage) + "\n"
                for iy, line in enumerate(textResult.split("\n")):
                    x = 10
                    y = 20 + iy * 15
                    cv2.putText(image, line, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.putText(image, np.array_str(matrixAP), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1,
                            cv2.LINE_AA)
                mAp = AccAP / indexImage * 100
                cv2.putText(image, str("mAp=") + str(round(mAp, 2)), (10, 100), self.font, 1, (255, 255, 255), 1,
                            cv2.LINE_AA)


                cv2.imshow("Result Check", image)
                cv2.waitKey()
                print(str("mAp=") + str(round(mAp, 2)))

    def EvaluateMaximumIOU(self):
        AccTP = 0
        AccFN = 0
        AccFP = 0
        font = cv2.FONT_HERSHEY_PLAIN

        annHelp = AnnHelp.AnnotationHelper()
        f = []
        for (dirpath, dirnames, filenames) in walk(self.AnnPath):
            f.extend(filenames)

        # fig = plt.figure(1)
        # ax1 = fig.add_subplot(1, 1, 1)

        Missrate = []
        FPI = []
        evaluateFile = 'Evaluate_Missrate.npy'
        ForceCal = True
        if not ForceCal and os.path.exists(evaluateFile):
            # Just change the True to false to force re-training
            print('Loading Evaluate')
            Missrate = np.load("Evaluate_Missrate.npy")
            FPI = np.load("Evaluate_FPI.npy")
        else:
            #start Evaluate
            thresholdnum = 0.1
            for file in f:
                filename, extfile = file.split(os.extsep)
                print(filename)
                # print(extfile)
                truth = annHelp.convert_annotation(self.AnnPath + filename + ".xml")
                predic = annHelp.convert_json(self.jsonPath + filename + ".json", threshold=thresholdnum)
                imagefile = self.ImagePath + filename + ".jpg"
                image = cv2.imread(imagefile)
                image,TP, FN, FP = annHelp.eveluate(truth, predic, image, thresholdIOU=0.6)
                # matrixAP, AP, mAP = annHelp.eveluateOrderIOU(truth, predic, image)
                AccFN += FN
                AccFP += FP
                AccTP += TP

                ###################################  Print reSult on Image ###############################
                textResult = "       TP \t\t        FN \t          FP \n"
                textResult += "Acc= \t\t" + str(AccTP) + "\t" + str(AccFN) + '\t' + str(AccFP) + "\n"
                textResult += "Cur= \t\t" + str(TP) + "\t" + str(FN) + '\t' + str(FP) + "\n"
                for iy, line in enumerate(textResult.split("\n")):
                    x = 10
                    y = 20 + iy * 15
                    for ix, lines in enumerate(line.split("\t")):
                        x = x + ix * 20
                        cv2.putText(image, lines, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)

                precision = self.divide_zero_is_zero(AccTP,AccTP+AccFP) * 100.00
                recall = self.divide_zero_is_zero(AccTP, AccTP+AccFN) * 100.00
                mAp = precision * recall / 100.0
                cv2.putText(image, str("Precision=")+str(round(precision,2)), (10, 70),font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, str("Recall=")+str(round(recall,2)), (10, 85), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, str("mAp=")+str(round(mAp,2)), (10, 100), font, 1, (255, 255, 255), 1, cv2.LINE_AA)


                # for iy, line in enumerate(textResult.split("\n")):
                #     y = 50 + iy * 20
                #     cv2.putText(image, lines, (0, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
                #

                cv2.imshow("Result Check",image)
                cv2.waitKey()
        print(str("Precision=")+str(round(precision,2)))
        print( str("Recall=")+str(round(recall,2)))
        print(str("mAp=")+str(round(mAp,2)))


if __name__ == "__main__":
    ev = Evaluate()
    ev.EvaluateMAP()
