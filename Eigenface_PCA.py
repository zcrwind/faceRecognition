# !/usr/bin/python
# -*- coding: UTF-8 -*-

'''
Author: zhang chenrui
2017.5 @NWPU
'''
import os
import re
import time
import numpy as np
from numpy import linalg
import cv2
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import operator

class PCA_EigenFace(object):
    '''使用基于PCA的特征脸方法实现人脸识别'''
    def __init__(self):
        self.trainCnt = 0    # 训练集图像总数
        self.testCnt = 0     # 训练集图像总数
        self.imgW = 128
        self.imgH = 128
        self.totalFaceMat = np.mat(np.empty((0, self.imgH * self.imgW)))  # 训练集总体矩阵
        self.trainSet_path = 'F:/face_recognition/zcr/aligned/total_10/trainSet/detect'  # './trainSet/'
        self.testSet_path = 'F:/face_recognition/zcr/aligned/total_10/testSet/detect'    # './testSet/'
        self.savePath = './Image_results/'
        self.classLabel = dict()        # 字典的key为人名(也就是类别名),value为训练集图像对应的编号组成的列表
        self.select_Threshold = 0.9     # PCA确定主成分个数的衡量阈值
        self.avgFaceVec = np.array      # 训练集的平均人脸向量("大众脸")
        self.diffMat = np.mat           # 偏差矩阵
        self.eigenVects_of_CovarianceMat = np.mat # 协方差矩阵(即总体散布矩阵)的特征向量组成的矩阵(每一列是一个特征脸向量)
        self.pattern = re.compile(r'([\w-]+)_\d{4}.jpg')   # LFW的图像命名格式是：人名_XXXX.jpg(注意：人名中可能会有'-')
        self.rightRate = 0   # 预测正确率

    def load_trainSet(self):
        '''加载训练集图像，进行矩阵向量化处理，并得到训练集总体矩阵'''
        self.trainCnt = 0
        for root, dirs, files in os.walk(self.trainSet_path, topdown = False):
            for imgName in files:
                try:
                    personName = self.pattern.match(imgName).groups()[0]
                except Exception as e:
                    print(e)
                if self.classLabel.get(personName) is None:
                    self.classLabel[personName] = []
                self.classLabel[personName].append(self.trainCnt)
                imgPath = os.path.join(root, imgName)
                img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
                self.totalFaceMat = np.vstack((self.totalFaceMat, np.mat(img).flatten()))
                self.trainCnt += 1
        self.totalFaceMat = (self.totalFaceMat).T    # 转置之后，每列是一个向量化的图像

    def PCA_Recognition(self, select_Threshold = 0.9):
        '''计算平均人脸向量、偏差矩阵、协方差矩阵(即总体散布矩阵)的特征向量'''
        self.load_trainSet()                                # 得到包含所有人脸图像的大矩阵,每一列是一个人脸矩阵转化成的向量
        self.avgFaceVec = np.mean(self.totalFaceMat, 1)     # 计算平均人脸向量
        self.diffMat = self.totalFaceMat - self.avgFaceVec  # 计算偏差矩阵
        # eigenvalues, eigenvectors = linalg.eig(np.mat(self.diffMat * self.diffMat.T))  # 理论上这样计算协方差矩阵的特征值和特征向量，但是协方差矩阵太大,运行时出现MemoryError，故用下面的方法
        eigenvalues, eigenvectors = linalg.eig(np.mat((self.diffMat).T * self.diffMat))    # eigenvectors是diffMat.T * diffMat的特征向量，并不是最终协方差矩阵的特征向量
        eigSortIndex = np.argsort(-eigenvalues)
        # 通过计算阈值来计算维数，即保证所保留的特征向量对应的特征值之和和总体特征值之和的比值大于一定的阈值select_Threshold，这个select_Threshold一般取0.9
        for i in range(np.shape(self.totalFaceMat)[1]):
            if (eigenvalues[eigSortIndex[:i]] / eigenvalues.sum()).sum() >= select_Threshold:
                eigSortIndex = eigSortIndex[:i]
                break
        self.eigenVects_of_CovarianceMat = self.diffMat * eigenvectors[:, eigSortIndex]
        
    def predictNewFace(self, newImgVec, newImg_trueLabel):
        '''进行预测. newImg_trueLabel是待预测图像的真实的类别(也就是人名)'''
        isRight = False
        EigenFacesMat = self.eigenVects_of_CovarianceMat    # EigenFacesMat就是特征脸向量组成的矩阵(每一列是一个特征脸向量)
        diff = newImgVec.T - self.avgFaceVec                # diff是一个62500x1的矩阵，即为一个列向量
        weightVec = EigenFacesMat.T * diff                  # weight是一个cx1的列向量，c为PCA得出的特征向量的个数
        preID = 0     # 图像在LFW训练集中的编号
        distance = float('inf')   # 初始化为正无穷
        for i in range(self.trainCnt):
            trainVec = EigenFacesMat.T * self.diffMat[:,i]
            if  (np.array(weightVec - trainVec) ** 2).sum() < distance:  # 用欧氏距离计算相近程度
                preID = i
                distance = (np.array(weightVec - trainVec) ** 2).sum()
        try:
            if preID in self.classLabel[newImg_trueLabel]:
                isRight = True
        except KeyError as e:
            pass
        return isRight

    def evaluation(self):
        '''估计预测准确率、召回率等指标'''
        print("Start the Prediction...")
        rightCnt = 0
        self.testCnt = 0
        for root, dirs, files in os.walk(self.testSet_path, topdown = False):
            for imgName in files:
                try:
                    personName = self.pattern.match(imgName).groups()[0]
                except Exception as e:
                    pass
                imgPath = os.path.join(root, imgName)
                judgeImg = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
                self.testCnt += 1
                if self.predictNewFace(np.mat(judgeImg).flatten(), personName):
                    rightCnt += 1
        self.rightRate = float(rightCnt) / self.testCnt
        print('The Number of test sample is: %d' % self.testCnt)
        print('The Number of RIGHT instances is: %d' % rightCnt)
        print('The Number of WRONG instances is: %d' % (self.testCnt - rightCnt))
        print('Prediction Accuracy is %f' % self.rightRate)  # 求出正确率

    def drawSubplot_EigenFaces(self, title, imagesList, rowNum, colNum):
        '''画特征脸'''
        fig = plt.figure()
        fig.text(.5, .95, title, horizontalalignment = 'center') 
        for i in range(len(imagesList)):
            ax0 = fig.add_subplot(rowNum, colNum, (i + 1))
            plt.imshow(np.asarray(imagesList[i]), cmap = "gray")
            # plt.imshow(np.asarray(imagesList[i]), cmap = cm.jet)
            plt.xticks([]), plt. yticks([]) # 隐藏X、Y坐标
        plt.show()
    
    def drawAvgFace(self):
        avgFaceMat = self.avgFaceVec.reshape(self.imgH, self.imgW)
        cv2.imwrite('PCA_croped_AverageFace.png', avgFaceMat)

if __name__ == '__main__':
    startTime = time.time()
    print("Start at: %s" % time.ctime())
    pcaObj = PCA_EigenFace()
    # pcaObj.load_trainSet()
    pcaObj.PCA_Recognition()
    train_endTime = time.time()
    print('Time of training is: %lf' % (train_endTime - startTime))
    print(pcaObj.trainCnt)
    print(pcaObj.totalFaceMat)
    print(pcaObj.totalFaceMat.shape)
    classfy_startTime = time.time()
    pcaObj.evaluation()
    endTime = time.time()
    print('Time of calssfication is: %lf' % (endTime - classfy_startTime))
    print('Total Time: %lf' % (endTime - startTime))

    # for key,value in pcaObj.classLabel.items():
    #     print(key, value)

    # 绘制平均脸
    print('Start drawing the Average Face...')
    pcaObj.drawAvgFace()    # OK
    print('Done.')

    # 绘制特征脸
    Eigenfaces = []
    FaceVectors = pcaObj.eigenVects_of_CovarianceMat
    for i in range(FaceVectors.shape[1]):
        if i > 12 - 1:
            break
        Eigenface = FaceVectors[:, i].reshape(pcaObj.imgH, pcaObj.imgW)
        Eigenfaces.append(Eigenface)
    pcaObj.drawSubplot_EigenFaces("Eigenfaces", Eigenfaces, 3, 4)

