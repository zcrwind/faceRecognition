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

class LDA_Fisherface(object):
    '''LDA_Fisherface'''
    def __init__(self):
        self.trainCnt = 0
        self.testCnt = 0
        self.imgW = 128
        self.imgH = 128
        self.totalFaceMat = np.mat(np.empty((0, self.imgH * self.imgW)))  # 训练集总体矩阵
        self.trainSet_path = 'F:/face_recognition/zcr/aligned/total_10/trainSet/detect'  # './trainSet/'
        self.testSet_path = 'F:/face_recognition/zcr/aligned/total_10/testSet/detect'    # './testSet/'
        self.savePath = './results/'
        self.classCnt = 0
        self.classLabel = dict()        # 字典的key为人名(也就是类别名),value为训练集图像对应的编号组成的列表
        self.avgFaceVec = np.array      # 训练集的平均人脸向量("大众脸")
        self.diffMat = np.mat           # 偏差矩阵
        self.eigenvalues_of_CovarianceMat = []
        self.eigenVects_of_CovarianceMat = np.mat # 协方差矩阵(即总体散布矩阵)的特征向量组成的矩阵(每一列是一个特征脸向量)
        self.LDA_eigenvalues = np.array
        self.LDA_eigenvectors = np.array
        self.fisher_eigenvectors = np.array
        self.pattern = re.compile(r'([\w-]+)_\d{4}.jpg')   # LFW的图像命名格式是：人名_XXXX.jpg(注意：人名中可能会有'-')
        self.rightRate = 0   # 预测正确率

    def load_trainSet(self):
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
        self.classCnt = len(self.classLabel)
        
    def PCA(self, PC_num = 0):
        '''换一种方式：确定个PC个数而不是设定阈值，来确定特征向量的个数'''
        [N, img_len] = self.totalFaceMat.shape   # N是训练集样本个数，img_len是图像矩阵向量化之后的维数
        if (PC_num <= 0) or (PC_num > N):        # 换一种思路求特征向量：给定主成分个数(PC_num)
            PC_num = N
        self.avgFaceVec = np.asarray(self.totalFaceMat.mean(axis = 0)[0,:])[0]
        self.diffMat = self.totalFaceMat - self.avgFaceVec
        if N > img_len:
            self.CovarianceMat = np.dot(self.diffMat.T, self.diffMat)
            [eigenvalues, eigenvectors] = np.linalg.eigh(self.CovarianceMat)
        else:
            self.CovarianceMat = np.dot(self.diffMat, self.diffMat.T)
            [eigenvalues, eigenvectors] = np.linalg.eigh(self.CovarianceMat)
            eigenvectors = np.dot(self.diffMat.T, eigenvectors)
            for i in range(N):
                eigenvectors[:,i] = eigenvectors[:,i] / np.linalg.norm(eigenvectors[:,i])
        eigSortIndex = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[eigSortIndex]
        eigenvectors = eigenvectors[:,eigSortIndex]
        self.eigenvalues_of_CovarianceMat = eigenvalues[0:PC_num].copy()
        self.eigenVects_of_CovarianceMat = eigenvectors[:,0:PC_num].copy()

    def LDA(self, PC_num = 0):
        classNum = self.classCnt
        eigVects = self.eigenVects_of_CovarianceMat
        projectedMat = np.dot(self.totalFaceMat - self.avgFaceVec, eigVects)   # 执行投影
        [N,d] = projectedMat.shape
        if (PC_num <= 0) or (PC_num > (len(classNum) - 1)):
            PC_num = classNum - 1
        meanTotal = projectedMat.mean(axis = 0)
        # 初始化著名的Sw和Sb
        Sw = np.zeros((d, d), dtype = np.float64)
        Sb = np.zeros((d, d), dtype = np.float64)
        for key,value in self.classLabel.items():
            Mat_i = np.mat(np.empty((0, projectedMat.shape[1])))
            for ID in value:
                Mat_i = np.vstack((Mat_i, projectedMat[ID,:]))
            meanClass = Mat_i.mean(axis = 0)
            # 计算著名的Sw和Sb
            Sw = Sw + np.dot((Mat_i - meanClass).T, (Mat_i - meanClass))
            Sb = Sb + N * np.dot((meanClass - meanTotal).T, (meanClass - meanTotal))
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw) * Sb)
        eigSortIndex = np.argsort(-eigenvalues.real)
        eigenvalues, eigenvectors = eigenvalues[eigSortIndex], eigenvectors[:,eigSortIndex]
        self.LDA_eigenvalues = np.array(eigenvalues[0:PC_num].real, dtype = np.float64, copy = True)
        self.LDA_eigenvectors = np.array(eigenvectors[0:,0:PC_num].real, dtype = np.float64, copy = True)

    def FisherFace_main(self):
        PCA_eigenVects = self.eigenVects_of_CovarianceMat
        LDA_eigenVects = self.LDA_eigenvectors
        self.fisher_eigenvectors = np.dot(PCA_eigenVects, LDA_eigenVects)
        trainProjections = []
        for i in range(self.totalFaceMat.shape[0]):
            projection = np.dot(self.totalFaceMat[i,:] - self.avgFaceVec, self.fisher_eigenvectors)
            trainProjections.append(projection)
        self.trainProjections = trainProjections

    def predictNewFace(self, newImgVec, newImg_trueLabel):
        '''进行预测. newImg_trueLabel是待预测图像的真实的类别(也就是人名)'''
        isRight = False
        distance = float('inf')   # 初始化为正无穷
        newImgProjection = np.dot(newImgVec - self.avgFaceVec, self.fisher_eigenvectors)
        preID = 0       # 图像在LFW训练集中的编号
        for i in range(len(self.trainProjections)):
            newimg = np.asarray(newImgProjection).flatten()
            train_i = np.asarray(self.trainProjections[i]).flatten()
            dist = np.sqrt(np.sum(np.power((newimg - train_i), 2)))
            if dist < distance:
                distance = dist
                preID = i
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
            plt.xticks([]), plt. yticks([])
        plt.show()
        fig.savefig('Fisher_subplot.png')

    def drawAvgFace(self):
        avgFaceMat = self.avgFaceVec.reshape(self.imgH, self.imgW)
        cv2.imwrite('croped_Fisherface_AverageFace.png', avgFaceMat)


if __name__ == '__main__':
    startTime = time.time()
    print("Start at: %s" % time.ctime())
    ldaObj = LDA_Fisherface()
    ldaObj.load_trainSet()
    ldaObj.PCA()
    print(ldaObj.trainCnt)
    ldaObj.LDA()
    ldaObj.FisherFace_main()
    train_endTime = time.time()
    print('Time of training is: %lf' % (train_endTime - startTime))
    classfy_startTime = time.time()
    ldaObj.evaluation()
    endTime = time.time()
    print('Time of calssfication is: %lf' % (endTime - classfy_startTime))
    print('Total Time: %lf' % (endTime - startTime))

    # 绘制平均脸
    print('Start drawing the Average Face...')
    ldaObj.drawAvgFace()    # OK
    print('Done.')

    # 绘制特征脸
    Eigenfaces = []
    FaceVectors = ldaObj.fisher_eigenvectors
    for i in range(FaceVectors.shape[0]):
        if i > 12 - 1:
            break
        Eigenface = FaceVectors[:, i].reshape(ldaObj.imgH, ldaObj.imgW)
        Eigenfaces.append(Eigenface)
    ldaObj.drawSubplot_EigenFaces("LDA Eigenfaces", Eigenfaces, 3, 4)