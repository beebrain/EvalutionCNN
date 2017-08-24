'''Train a simple deep CNN on the CIFAR10 small images dataset.
Dataset
1. ImageDataset_Gray (64*64)
2. DetailDataset
3. Dataset.txt
'''

from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
from datetime import datetime
import Finger
from keras import backend as K
import os
import cv2
import matplotlib.pyplot as plt
import sys
import HandProcess as hp
from time import time as timer
print(sys.argv)

def checkAcc(YRealValue,YPredic):

    if len(YRealValue) != len(YPredic):
        return 0
    result = YPredic - YRealValue
    result = (result!=0)*1
    loss = sum(result)
    acc = len(YPredic) - loss
    print (acc,loss)
    return acc,loss

blocking = 3
person = 1
filemodel = "./1Block/"+str(person)+"/CNN_model_"+str(blocking)+"_"+str(filter1)+"_"+str(filter2)+"_"+str(filter3)
print(filemodel)
# input imag dimensions
img_rows, img_cols = 64, 64
# the CIFAR10 images are RGB
img_channels = 1

elapsed = int()
start = timer()

######################## Start load library ####################################
if os.path.isfile(filemodel+".json") :
    json_file = open(filemodel+".json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(filemodel + ".h5")
    print("Loaded model from disk")
else:
    print("Can't found Model file exit Program ")
    exit()

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
print(model.summary())
handPro = hp.HandProcess()
############################################### For Test Static Image #################################################
# X_test,Ytest = Finger.loadAllDataset(False)
# Y_test = np.reshape(Ytest, -1)
# print(X_test.shape)
#
# for index in range(len(X_test)):
#     if K.image_dim_ordering() == 'th':
#         input_X_test = X_test[index].reshape(1, img_channels, img_rows, img_cols)
#         input_shape = (img_channels, img_rows, img_cols)
#         print("theno")
#     else:
#         input_X_test = X_test[index].reshape(1, img_rows, img_cols, img_channels)
#         input_shape = (img_rows, img_cols, img_channels)
#         print("tersorflow")
#
#     cv2.imshow("RealImage",X_test[index])
#     cv2.waitKey()
#
#     input_X_test = input_X_test.astype('float32')
#     input_X_test /= 255
#     e = model.predict_classes(input_X_test)
#     print(e)
#     # acc,loss = checkAcc(Y_test, e)
#     # print(acc)
# print("Finished")
############################################### Finished Static Image #################################################

####################### finish Config camera ###################################
SaveVideo = False
CameraMode = False

if CameraMode:
    camera = cv2.VideoCapture(1)

else:
    camera = cv2.VideoCapture('output1.avi')

cv2.namedWindow('ca1', 0)
_, frame = camera.read()
height, width, _ = frame.shape

##################################### Save Video Section#########################
if SaveVideo:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = round(camera.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter('output.avi', fourcc, 29.0, (width,height))
################################################################################



ispauseVideo = False
while camera.isOpened():
    choice = cv2.waitKey(1)
    if choice == 27:
        break
    if choice == 32:
        ispauseVideo = not ispauseVideo
        print(ispauseVideo)
    if ispauseVideo:
        continue

    _, frame = camera.read()

    if frame is None:
        print('\nEnd of Video')
        break

    if SaveVideo:
        out.write(frame)

    ######### Extraction Process ##################

    ROI = handPro.process(frame)

    ROI_Gray =  cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY).astype("uint8")  # image size 200 x 200

    ROI64 = cv2.resize(ROI_Gray, (64, 64), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("ROI64",ROI64)
    if K.image_dim_ordering() == 'th':
        input_X_test = ROI64.reshape(1, img_channels, img_rows, img_cols)
        input_shape = (img_channels, img_rows, img_cols)
        # print("theno")
    else:
        input_X_test = ROI64.reshape(1, img_rows, img_cols, img_channels)
        input_shape = (img_rows, img_cols, img_channels)
        # print("tersorflow")

    input_X_test = input_X_test.astype('float32')
    input_X_test /= 255
    e = model.predict_classes(input_X_test,verbose=0)
    e = e[0]+1
    print(e)
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    cv2.putText(ROI_Gray, str(e), (40, 40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("200 ROI",ROI_Gray)

    cv2.putText(frame, str(e), (100, 100), font,2, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.resizeWindow('ca1', width, height)
    cv2.imshow("ca1",frame)
    # print(ROI64.shape)
    elapsed += 1
    if elapsed % 5 == 0:
        sys.stdout.write('\r')
        sys.stdout.write('{0:3.3f} FPS'.format(
            elapsed / (timer() - start)))
        sys.stdout.flush()



if SaveVideo:
    out.release()

sys.stdout.write('\n')

camera.release()
cv2.destroyAllWindows()