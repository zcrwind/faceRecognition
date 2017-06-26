# !/usr/bin/python
# -*- coding: UTF-8 -*-

'''对LFW的数据进行预处理，选出其中照片数目>=filterNum的person'''

import os
import shutil

def filter(sourceDir, targetDir, filterNum):
    '''过滤原始的LFW数据集，找出其中照片数目大于filterNum的person，存放在targetDir下'''
    for root, dirs, files in os.walk(sourceDir, topdown = False):
        for subdir in dirs:
            source_subPath = os.path.join(root, subdir)
            imgfiles = os.listdir(source_subPath)
            if len(imgfiles) >= filterNum:
                target_subPath = os.path.join(targetDir, subdir)
                # print(target_subPath)
                if not os.path.exists(target_subPath):
                    os.makedirs(target_subPath)
                for imgfile in imgfiles:
                    print(imgfile)
                    sourceFilePath = os.path.join(source_subPath, imgfile)
                    targetFilePath = os.path.join(target_subPath, imgfile)
                    shutil.copyfile(sourceFilePath, targetFilePath)

def get_Train_TestSet(filteredDir, trainSaveDir, testSaveDir, ratio):
    '''根据设定的比例构造训练集和测试集'''
    for root, dirs, files in os.walk(filteredDir, topdown = False):
        for subdir in dirs:
            source_subPath = os.path.join(root, subdir)
            # print(source_subPath)
            imgfiles = os.listdir(source_subPath)
            # totalCnt = len(imgfiles)
            totalCnt = 20
            trainCnt = int(ratio * totalCnt)
            testCnt = totalCnt - trainCnt
            cnt = 0
            # print("total: %d  train: %d  test: %d" % (len(imgfiles), trainCnt, testCnt))
            for imgfile in imgfiles:
                sourceFilePath = os.path.join(source_subPath, imgfile)
                if cnt < trainCnt:
                    target_subPath = os.path.join(trainSaveDir, subdir)
                else:
                    target_subPath = os.path.join(testSaveDir, subdir)
                if not os.path.exists(target_subPath):
                    os.makedirs(target_subPath)
                targetFilePath = os.path.join(target_subPath, imgfile)
                shutil.copyfile(sourceFilePath, targetFilePath)
                cnt += 1
                if cnt >= totalCnt:
                    break

if __name__ == '__main__':
    # 执行filter
    # sourceDir = 'F:/face_recognition/lfw/'
    # targetDir = 'F:/face_recognition/zcr/LFW_Filtered_20/'
    # filter(sourceDir, targetDir, 20)

    # 构造训练集和测试集
    filteredDir = 'F:/face_recognition/zcr/LFW_Filtered_20/'
    trainSaveDir = 'F:/face_recognition/zcr/trainSet/'
    testSaveDir = 'F:/face_recognition/zcr/testSet/'
    ratio = 0.8     # ratio是训练集占比
    get_Train_TestSet(filteredDir, trainSaveDir, testSaveDir, ratio)