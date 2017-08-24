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

# import matplotlib.pyplot as plt

class AnnotationHelper:

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_PLAIN

    def isValidJson(self,data_file):
        fh = open(data_file, "r")
        stringline = fh.readline()
        fh.close()
        if len(stringline) > 2:
            return True
        else:
            return False

    def convert_json(self,jsonfile,threshold=0):
        arraydata = []
        with open(jsonfile) as data_file:
            if not self.isValidJson(jsonfile):
                return arraydata
            data = json.load(data_file)
            for items in data:

                topleft = items["topleft"]
                bottomright = items["bottomright"]
                item = items["label"]
                bbox = [topleft['x'],topleft['y'],bottomright['x'],bottomright['y']]
                IOU = -1
                firstConf = 0
                secondConf = 0
                if "FirstConf" in item:
                    firstConf = items["FirstConf"]
                    secondConf = items["SecondConf"]
                arraydata.append([item,bbox,firstConf,secondConf])
                #print(arraydata)
        return arraydata

    def convert_annotation(self,annotationfile):
        in_file = open(annotationfile)
        tree=ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = float(size.find('width').text)
        h = float(size.find('height').text)
        arraymeta = []
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text

            xmlbox = obj.find('bndbox')
            xmin = float(xmlbox.find('xmin').text)
            xmax = float(xmlbox.find('xmax').text)
            ymin = float(xmlbox.find('ymin').text)
            ymax = float(xmlbox.find('ymax').text)

            #[{"label": "Hand","confidence": 1.11,"topleft": {"x": 118, "y": 58},"bottomright": {"x": 427,"y": 537}},]
            metadata = {"label":cls,"topleft":(xmin,ymin),"bottomright":(xmax,ymax)}
            bbox = [xmin,ymin,xmax,ymax]
            IOU = -1
            lable = cls
            arraymeta.append([lable,bbox,IOU,w,h])
        return arraymeta

    def bb_intersection_over_union(self,boxA, boxB,threshold=0.6):
        # determine the (x, y)-coordinates of the intersection rectangle

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = (xB - xA + 1) * (yB - yA + 1)

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        if iou<0:
            return 0
        return iou

    def eveluateOrderIOU(self,truth,predic,image,thresholdIOU=0.6):
        numberTruth = len(truth)
        numberPredic = len(predic)
        listTruthClass = []             # for truth class order
        listThreshold = []
        listPredicClass = []

        # check gound truth ok!
        # for indextrouth in truth:
        #     bbox = np.asarray(indextrouth[1],dtype=np.int)
        #     cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),1)
        # cv2.imshow("x",image)
        # cv2.waitKey(0)

        for itruth in range(len(truth)):
            indextruth = truth[itruth]
            bbptruth = indextruth[1]
            classTruth = indextruth[0]
            listTruthClass.append(classTruth)
            bbplot = np.asarray(bbptruth, dtype=np.int)
            # print(bbplot)
            cv2.rectangle(image, (bbplot[0], bbplot[1]), (bbplot[2], bbplot[3]), (0, 255, 0), 2)
            cv2.putText(image, classTruth, (bbplot[0], bbplot[1]-2), self.font, 1, (0, 255, 0), 1, cv2.LINE_AA)


            for ipredic in range(len(predic)):
                indexpredic = predic[ipredic]
                bbpredic = indexpredic[1]

                # print(truth)
                # check IOU
                IOU = self.bb_intersection_over_union(bbpredic, bbptruth)
                listPredicClass.append([indexpredic[0],IOU])

                print(indexpredic[2])
                bbplot = np.asarray(bbpredic, dtype=np.int)
                cv2.rectangle(image, (bbplot[0], bbplot[1]), (bbplot[2], bbplot[3]), (0, 0, 255), 2)
                info_IOU = str(indexpredic[0])+"("+str(round(IOU*100,2))+"%)"
                cv2.putText(image,info_IOU , (bbplot[0],bbplot[1]+15), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, "First="+str(indexpredic[2]),(10,520), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, "Second=" + str(indexpredic[3]), (10, 540), self.font, 1, (255, 255, 255), 1,cv2.LINE_AA)
                # cv2.imshow("checkBB",image)
                # cv2.waitKey(100)

        ################################# Start confusion Matrix to list Threshold #################################
        #               truth
        #           -------------
        # predic   |    0   0   0
        #          |    0   0   0
        #          |    0   0   0
        matrixAP = np.zeros((1,2))
        # print(listPredicClass)
        # Sort List order by IOU from maximum to minimum
        sortList = sorted(listPredicClass, key=itemgetter(1), reverse=True)
        # select IOU only pass threshold
        listPassTh = filter(lambda (_, ThIOU): ThIOU > thresholdIOU, sortList)
        # print(listPassTh)

        numPredic = 0.   # Amount of predic
        numCorrectPedic = 0.  # Amount of coorect predic

        for index in listPassTh:
            print(index)
            pClass = index[0]
            iouClass  = index[1]
            numPredic += 1
            if(classTruth == pClass):
                numCorrectPedic += 1
                pecision = 1 / numPredic
                recall = numCorrectPedic / len(truth) - matrixAP[-1][1]  # current recall - last recall
                matrixAP = np.vstack((matrixAP, (pecision, recall)))
                break
            else:
                pecision = 0 / numPredic
                recall = numCorrectPedic / len(truth) - matrixAP[-1][1]  # current recall - last recall
                matrixAP = np.vstack((matrixAP, (pecision, recall)))

        matrixAP = matrixAP[1:,:]  # remove temp array 0,0
        AP = np.sum(matrixAP[:, 0]* matrixAP[:, 1]) # sum of Percision * delta(Recall)
        mAP = AP/len(truth)
        print(matrixAP)
        return matrixAP,AP,mAP

    def eveluate(self,truth,predic,image,thresholdIOU=0.6):
        FN = 0  # + if amount truth morethan predic
        FP = 0  # + if amount predic morethan truth
        TP = 0  #
        numberTruth = len(truth)
        numberPredic = len(predic)
        listTruthClass = [None]*numberTruth             # for truth class order
        listPredicClass = [None]*numberPredic            # for predic class order
        listThreshold = []

        # check gound truth ok!
        # for indextrouth in truth:
        #     bbox = np.asarray(indextrouth[1],dtype=np.int)
        #     cv2.rectangle(image,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),1)
        # cv2.imshow("x",image)
        # cv2.waitKey(0)
        matrix_IOU = np.zeros((len(predic), len(truth)), dtype=np.float)
        matrix_mask = np.ones((len(predic), len(truth)), dtype=np.float)

        for itruth in range(len(truth)):
            indextruth = truth[itruth]
            bbptruth = indextruth[1]
            listTruthClass[itruth] = indextruth[0]
            bbplot = np.asarray(bbptruth, dtype=np.int)
            # print(bbplot)
            cv2.rectangle(image, (bbplot[0], bbplot[1]), (bbplot[2], bbplot[3]), (0, 255, 0), 2)
            cv2.putText(image, indextruth[0], (bbplot[0], bbplot[1]-2), self.font, 1, (255, 255, 0), 1, cv2.LINE_AA)


            for ipredic in range(len(predic)):
                indexpredic = predic[ipredic]
                bbpredic = indexpredic[1]
                listPredicClass[ipredic] = indexpredic[0]
                # print(truth)
                # check IOU
                IOU = self.bb_intersection_over_union(bbpredic, bbptruth)
                matrix_IOU[ipredic][itruth] = IOU

                bbplot = np.asarray(bbpredic, dtype=np.int)
                cv2.rectangle(image, (bbplot[0], bbplot[1]), (bbplot[2], bbplot[3]), (0, 0, 255), 2)
                info_IOU = str(indexpredic[0])+" ("+str(round(IOU*100,2))+"%)"
                cv2.putText(image,info_IOU , (bbplot[0],bbplot[1]-2), self.font, 1, (0, 0, 255), 1, cv2.LINE_AA)
                # cv2.imshow("checkBB",image)
                # cv2.waitKey(100)

        ################################# Start confusion Matrix to list Threshold #################################
        #               truth
        #           -------------
        # predic   |    0   0   0
        #          |    0   0   0
        #          |    0   0   0

        print(listTruthClass)
        print(listPredicClass)

        if numberPredic > numberTruth:  # predic > truth : FP ++
            FP = numberPredic - numberTruth
            numListThreshold = numberTruth  # number for fet all maxtix element
        elif numberPredic < numberTruth:
            FN = numberTruth - numberPredic
            numListThreshold = numberPredic
        else:                                   # same predict object
            numListThreshold = numberPredic

        # print(matrix_IOU)
        for indexlist in range(numListThreshold):
            matrix_IOU = matrix_IOU * matrix_mask

            maxIOU = np.argmax(matrix_IOU)

            # ROW is index of predic
            # COL is index of truth

            maxrow = int(np.floor(maxIOU / len(truth)))
            # print(maxrow)
            maxcol = maxIOU % len(truth)
            # print(maxcol)
            matrix_mask[maxrow, :] = 0
            matrix_mask[:, maxcol] = 0
            # print(matrix_mask)
            listThreshold.append((matrix_IOU[maxrow][maxcol],listTruthClass[maxcol],listPredicClass[maxrow]))

        ######################################## Finish get to list threshold ############################################
        #print(listThreshold)

        for indexlist in range(len(listThreshold)):
            infoThreshold = listThreshold[indexlist]
            print(infoThreshold)
            if infoThreshold[0] >= thresholdIOU  and infoThreshold[1] == infoThreshold[2]:
                TP += 1
            else:
                FN += 1
                FP += 1
        return image,TP,FN,FP


if __name__=="__main__":
    annHelp = AnnotationHelper()
    ######### case 1 ####################
    truth = [['Hand01', [251.0, 208.0, 515.0, 666.0], -1, 800.0, 869.0]]
    predic = [['Hand03', [251, 208, 515, 600], 0.92],['Hand02', [251.0,  208, 515, 601],0.95],['Hand01', [251.0,  208, 500, 600], 0.54]]
    image = np.ones((800,800,3))
    matrixAP, AP, mAP = annHelp.eveluateOrderIOU(truth, predic, image)
    # print(TP,",",FN,",",FP)
    print("###################################")
    ######### case 2 ####################
    truth = [['Hand01', [251.0, 208.0, 515.0, 666.0], -1, 800.0, 869.0]]
    predic = [['Hand01', [25, 20, 51, 66], 0.92],['Hand02', [251, 208, 515, 600], 0.92]]
    image = np.ones((800,800,3))
    matrixAP, AP, mAP = annHelp.eveluateOrderIOU(truth, predic, image)
    # print(TP,",",FN,",",FP)
    print("###################################")
    ######### case 3 ####################
    truth = [['Hand01', [251.0, 208.0, 515.0, 666.0], -1, 800.0, 869.0]]
    predic = [['Hand02', [251, 208, 515, 666], 0.92]]
    image = np.ones((800,800,3))
    matrixAP,AP,mAP = annHelp.eveluateOrderIOU(truth, predic, image)
    # print(TP,",",FN,",",FP)
    print("###################################")

    ######### case 4 ####################
    truth = [['Hand01', [251.0, 208.0, 515.0, 666.0], -1, 800.0, 869.0]]
    predic = [['Hand01', [251, 208, 515, 666], 0.92]]
    image = np.ones((800,800,3))
    matrixAP, AP, mAP = annHelp.eveluateOrderIOU(truth, predic, image)
    # print(TP,",",FN,",",FP)
    print("###################################")

    ######### case 5 ####################
    truth = [['Hand01', [251.0, 208.0, 515.0, 666.0], -1, 800.0, 869.0]]
    predic = [['Hand01', [25, 20, 51, 66], 0.92]]
    image = np.ones((800,800,3))
    matrixAP, AP, mAP = annHelp.eveluateOrderIOU(truth, predic, image)
    # print(TP,",",FN,",",FP)
    print("###################################")

    ######### case 6 ####################
    truth = [['Hand01', [251.0, 208.0, 515.0, 666.0], -1, 800.0, 869.0]]
    predic = [['Hand01', [251, 208, 515, 666], 0.92],['Hand01', [251, 208, 515, 600], 0.92]]
    image = np.ones((800, 800, 3))
    matrixAP, AP, mAP = annHelp.eveluateOrderIOU(truth, predic, image)
    # print(TP, ",", FN, ",", FP)
    print("###################################")

    ######### case 7 ####################
    truth = [['Hand01', [251.0, 208.0, 515.0, 666.0], -1, 800.0, 869.0]]
    predic = [['Hand01', [25, 20, 51, 66], 0.92],['Hand01', [25, 20, 51, 66], 0.92]]
    image = np.ones((800, 800, 3))
    matrixAP, AP, mAP = annHelp.eveluateOrderIOU(truth, predic, image)
    # print(TP, ",", FN, ",", FP)
    print("###################################")

    ######### case 8 ####################
    truth = [['Hand01', [251.0, 208.0, 515.0, 666.0], -1, 800.0, 869.0]]
    predic = [['Hand02', [251, 208, 515, 666], 0.92]]
    image = np.ones((800, 800, 3))
    matrixAP, AP, mAP = annHelp.eveluateOrderIOU(truth, predic, image)
    # print(TP, ",", FN, ",", FP)
    print("###################################")


#Example Runtest
# if __name__ == "__main__":
#     AnnHelp = AnnotationHelper()
#     AnnPath = "../HandDataset/Hand_25Class/Annotation/xml/"
#     jsonPath = "./Dataset/out_Yolo_2_25Class/json/"
#     ImagePath = "../HandDataset/Hand_25Class/Annotation/imageresize/"
#     f = []
#     for (dirpath, dirnames, filenames) in walk(AnnPath):
#         f.extend(filenames)
#
#     # fig = plt.figure(1)
#     # ax1 = fig.add_subplot(1, 1, 1)
#
#     Missrate = []
#     FPI = []
#     evaluateFile = 'Evaluate_Missrate.npy'
#     ForceCal = True
#     if not ForceCal and os.path.exists(evaluateFile):
#         # Just change the True to false to force re-training
#         print('Loading Evaluate')
#         Missrate = np.load("Evaluate_Missrate.npy")
#         FPI = np.load("Evaluate_FPI.npy")
#     else:
#         # listThreshold = [0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#         for thresholdIOU in np.arange(0.7,1,0.1):
#             AllEvaluate = np.zeros((1, 3))
#             for thresholdnum in np.arange(0.1,1,0.1):
#                 print (thresholdnum)
#                 evaluate = np.zeros((1, 3))
#                 for file in f:
#                     filename,extfile = file.split(os.extsep)
#                     # print(filename)
#                     # print(extfile)
#
#                     truth = AnnHelp.convert_annotation(AnnPath+filename+".xml")
#                     predic = AnnHelp.convert_json(jsonPath+filename+".json",threshold=thresholdnum)
#                     imagefile = ImagePath+filename+".jpg"
#                     image = cv2.imread(imagefile)
#                     TP,FN,FP = AnnHelp.eveluate(truth,predic,image,thresholdIOU=thresholdIOU)
#                     # print('TP={} FN={} FP={}'.format(TP, FN, FP))
#                     evaluate = np.vstack((evaluate,[TP,FN,FP]))
#
#                 AllEvaluate = np.vstack((AllEvaluate,np.sum(evaluate,axis=0)))
#             if Missrate == []:
#                 #calculate Missrate  = TN/TP+FN  , FPI = FP/TP+FN
#                 Missrate = AllEvaluate[1:, 1] / (AllEvaluate[1:, 1] + AllEvaluate[1:, 0])
#                 FPI = AllEvaluate[1:, 2] / (AllEvaluate[1:, 1] + AllEvaluate[1:, 0])
#             else:
#                 Missrate = np.vstack((Missrate,AllEvaluate[1:, 1] / (AllEvaluate[1:, 1] + AllEvaluate[1:, 0])))
#                 FPI = np.vstack((FPI,AllEvaluate[1:, 2] / (AllEvaluate[1:, 1] + AllEvaluate[1:, 0])))
#         np.save("Evaluate_Missrate",Missrate)
#         np.save("Evaluate_FPI", FPI)
#
#     ## Finish count TP and FN
#     # Missrate = AllEvaluate[1:,1]/(AllEvaluate[1:,1]+AllEvaluate[1:,0])
#     # FPI = AllEvaluate[1:, 2] / (AllEvaluate[1:, 1] + AllEvaluate[1:, 0])
#     print(Missrate)
#     print(FPI)
#
#
#     fig, ax = plt.subplots()
#     ax.plot(FPI[0,:],Missrate[0,:], '-.', label='IOU = 0.5')
#     ax.plot(FPI[1,:],Missrate[1,:], '-.', label='IOU = 0.6')
#     ax.plot(FPI[2,:],Missrate[2,:], '-.,', label='IOU = 0.7')
#     ax.plot(FPI[3, :], Missrate[3, :], '-.', label='IOU = 0.8')
#     ax.plot(FPI[4, :], Missrate[4, :], '-.', label='IOU = 0.9')
#     ax.set_xlabel("FPI")
#     ax.set_ylabel("Missrate")
#     legend = ax.legend(loc='upper right', fontsize='x-small')
#
#
#     plt.show()



