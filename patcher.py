from PIL import Image
import numpy as np
import cv2
import os.path as path
import os
import json
import random
def DelTree(treeName, isDelDir=False, isDelRoot=False):
    'delete a tree, recursively, it can be non empty!'
    if not path.exists(treeName):
        if not isDelDir and not isDelRoot:
            os.mkdir(treeName)
        return -1

    for root, dirs, files in os.walk(treeName, topdown=False):
        for name in files:
            os.remove(path.join(root, name))
            # print "deleting file %s" % name
        if isDelDir == True:
            for name in dirs:
                os.rmdir(path.join(root, name))
                # print "deleting folder %s" % name
    if isDelDir == True and isDelRoot == True:
        os.rmdir(treeName)

class Patcher():
    def __init__(self, provider):
        '''
        provider对象必须包含以下属性
            1. dctFiles: 指出数据集中的文件 
            2. MapFile: 返回经过映射的文件名，用于打包的数据集
        '''
        self.dctFiles = provider.dctFiles
        self.provider = provider

    def GetClusters(self, ndx, closeRate=0.5, isShow=True, allowedTags=['*'], maxPairs=200):
        '''
        GetClusters: 获取图片中人脸的群聚。closeRate表示bbox的面积之和除以这些bbox的总面积
        返回 [[二个脸的框], [三个脸的框], [多个脸的框]], opencv标注后的图
        '''
        strFile = list(self.dctFiles.keys())[ndx]
        item = self.dctFiles[strFile]
        bboxCnt = len(item['xywhs'])
        if len(item['xywhs']) < 4:
            return [[],[],[]],None
        
        def _getValues(b1):
            w1 = b1['w']
            h1 = b1['h']
            x1 = b1['x1']
            y1 = b1['y1']
            x2 = x1 + w1
            y2 = y1 + h1
            sqrtA = (w1 * h1) ** 0.5
            cx1 = [b1['x1'] + b1['w'] / 2]
            cy1 = [b1['y1'] + b1['h'] / 2] 
            return w1, h1, x1, y1, x2, y2, cx1, cy1, sqrtA
        # 先获取靠近的一对
        lstPairs = []
        for i in range(bboxCnt):
            b1 = item['xywhs'][i]
            if not b1['tag'] in allowedTags:
                if '*' != allowedTags[0]:
                    continue 
            wi, hi, x1i, y1i, x2i, y2i, cxi, cyi, sqrtAi = _getValues(b1)
            for j in range(i + 1, bboxCnt):
                b2 = item['xywhs'][j]
                if not b2['tag'] in allowedTags:
                    if '*' != allowedTags[0]:
                        continue
                wj, hj, x1j, y1j, x2j, y2j, cxj, cyj, sqrtAj = _getValues(b2)
                x1o = x1i if x1i < x1j else x1j
                y1o = y1i if y1i < y1j else y1j
                x2o = x2i if x2i > x2j else x2j
                y2o = y2i if y2i > y2j else y2j
                wo = x2o - x1o
                ho = y2o - y1o
                areaO = wo * ho
                areaIJ = wi * hi + wj * hj
                if areaIJ / areaO >= closeRate:
                    dctBbox = {
                        'x1' : x1o,
                        'y1' : y1o,
                        'w': x2o - x1o,
                        'h': y2o - y1o,
                    }
                    lstPairs.append([[b1, b2], (x1o, y1o), (x2o, y2o), dctBbox, areaIJ])
        # 再获取靠近的三元组
        if len(lstPairs) > maxPairs:
            random.shuffle(lstPairs)
            lstPairs = lstPairs[:maxPairs]
        lstTrints = []
        for i in range(len(lstPairs)):
            pair = lstPairs[i]
            b1 = pair[3]
            wi, hi, x1i, y1i, x2i, y2i, cxi, cyi, sqrtAi = _getValues(b1)
            for j in range(bboxCnt):
                b2 = item['xywhs'][j]
                if not b2['tag'] in allowedTags:
                    if '*' != allowedTags[0]:
                        continue                
                if b2 == pair[0][0] or b2 == pair[0][1]:
                    continue
                wj, hj, x1j, y1j, x2j, y2j, cxj, cyj, sqrtAj = _getValues(b2)
                x1o = x1i if x1i < x1j else x1j
                y1o = y1i if y1i < y1j else y1j
                x2o = x2i if x2i > x2j else x2j
                y2o = y2i if y2i > y2j else y2j
                wo = x2o - x1o
                ho = y2o - y1o
                areaO = wo * ho
                areaIJ = pair[4] + wj * hj
                if areaIJ / areaO >= closeRate * 2 / 3:
                    dctBbox = {
                        'x1' : x1o,
                        'y1' : y1o,
                        'w': x2o - x1o,
                        'h': y2o - y1o
                    }
                    tmp = lstPairs[i][0]  + [b2]
                    isRepeat = False
                    for ele in lstTrints:
                        if ele[1] == (x1o, y1o) or ele[2] == (x2o, y2o):
                            isRepeat = True
                            break
                    if isRepeat == False:               
                        lstTrints.append([tmp, (x1o, y1o), (x2o, y2o), dctBbox, areaIJ])                
        # 在三元组中合并交并比比较大的
        lstMulti = []
        for i in range(len(lstTrints)):
            b1 = lstTrints[i][3]
            wi, hi, x1i, y1i, x2i, y2i, cxi, cyi, sqrtAi = _getValues(b1)
            for j in range(i + 1, len(lstTrints)):
                b2 = lstTrints[j][3]
                wj, hj, x1j, y1j, x2j, y2j, cxj, cyj, sqrtAj = _getValues(b2)

                x1o = x1i if x1i < x1j else x1j
                y1o = y1i if y1i < y1j else y1j
                x2o = x2i if x2i > x2j else x2j
                y2o = y2i if y2i > y2j else y2j
                # 相交区
                x1It = x1i if x1i > x1j else x1j
                y1It = y1i if y1i > y1j else y1j
                x2It = x2i if x2i < x2j else x2j
                y2It = y2i if y2i < y2j else y2j
                if x1It >= x2It or y1It >= y2It:
                    continue
                itArea = (x2It - x1It) * (y2It - y1It)
                unArea = (x2o - x1o) * (y2o - y1o)
                iou = itArea / unArea
                if iou < 0.5 or iou > 0.99:
                    continue
                lstItBoxes = []
                # 去重
                lstTmp = lstTrints[i][0] + lstTrints[j][0]
                lstItBox = []
                areaIJ = 0
                for ii in range(6):
                    isRepeat = False
                    for jj in range(ii + 1, 6):
                        if jj == ii:
                            continue
                        if lstTmp[ii] == lstTmp[jj]:
                            isRepeat = True
                            break
                    if isRepeat:
                        continue
                    areaIJ += lstTmp[ii]['w'] * lstTmp[ii]['h']
                    lstItBox.append(lstTmp[ii])
                dctBbox = {
                    'x1' : x1o,
                    'y1' : y1o,
                    'w': x2o - x1o,
                    'h': y2o - y1o
                }
                isRepeat = False
                for ele in lstMulti:
                    if ele[1] == (x1o, y1o) or ele[2] == (x2o, y2o):
                        isRepeat = True
                        break
                if isRepeat == False:
                    areaO = dctBbox['w'] * dctBbox['h']
                    if areaIJ / areaO >= closeRate:
                        lstMulti.append([lstItBox, (x1o, y1o), (x2o, y2o), dctBbox, areaIJ])
        self.provider.MapFile(strFile)
        image = Image.open(self.provider.MapFile(strFile))
        img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        for clst in lstTrints:
            cv2.rectangle(img, clst[1], clst[2], (255, 255, 255), 2, 4)       
        for clst in lstPairs:
            cv2.rectangle(img, clst[1], clst[2], (0, 255, 0), 2, 4)
        for clst in lstMulti:
            cv2.rectangle(img, clst[1], clst[2], (255, 0, 0), 2, 4)     
        if isShow:
            cv2.imshow("OpenCV",img)
            cv2.waitKey()
        lstRet = [lstPairs, lstTrints, lstMulti]
        return lstRet, np.array(img)
    
    def ShowClusterRandom(self, isShow=True):
        while True:
            ndx = np.random.randint(len(self.dctFiles))
            lstRet, img = self.GetClusters(ndx, isShow=False)
            if len(lstRet[2]) > 0:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
                cv2.imshow("OpenCV",img)
                cv2.waitKey()                   
                break
    '''
    CutPatches: 切割出包含一簇人脸的图片
    '''
    def CutClusterPatches(self, strOutFolder, patchNdx, scalers=[0.95, 0.7], outSize = [96, 128], \
            ndx=-1, strFile='', maxObjPerCluster=5, closeRatio=0.333, lstClusters=[], maxPatchPerImg=16, allowedTags=['*']):
        if ndx >= 0:
            strFile = list(self.dctFiles.keys())[ndx]
        lstRet, img = self.GetClusters(ndx, isShow=False, allowedTags=allowedTags)
        lstOriPats = lstRet[0] + lstRet[1] + lstRet[2]
        # item = self.dctFiles[strFile]
        if len(lstOriPats) < 1:
            return patchNdx, []
        sMainName = os.path.splitext(os.path.split(strFile)[1])[0]

        image = Image.open(self.provider.MapFile(strFile))
        lstPatches = []
        wVsH = outSize[0] / outSize[1]
        random.shuffle(lstOriPats)
        # pat打包格式：[[子框],(x1, y1),(x2,y2), bbox, 面积]
        for (i, pat) in enumerate(lstOriPats):
            if len(pat[0]) > maxObjPerCluster:
                continue
            bbox = pat[3]
            w  = bbox['w']
            h = bbox['h']
            x1 = bbox['x1']
            y1 = bbox['y1']
            cx = x1 + w // 2
            cy = y1 + h // 2
            rand_clip_retry = 30
            scaler = 0.95
            for retryNdx in range(rand_clip_retry):
                # w2, h2,cx2, cy2, x12, x22, y12, y22表示输出patch的几何信息
                w2 = w / scaler
                h2 = h / scaler
                if w2 / h2 < wVsH:
                    w2 = h2 * wVsH
                else:
                    h2 = w2 / wVsH
                w2 = int(w2 + 0.5)
                h2 = int(h2 + 0.5)
                # 随机水平移动
                xOfsRange = w2 * (1-scaler) / 2
                yOfsRange = h2 * (1-scaler) / 2
                xOfs = np.random.rand() * xOfsRange - xOfsRange / 2
                yOfs = np.random.rand() * yOfsRange - yOfsRange / 2
                cx2 = cx + xOfs
                cy2 = cy + yOfs
                # 必须确保长宽比
                if cx2 - w2 / 2 < 0:
                    continue
                if cy2 - h2 / 2 < 0:
                    continue
                x12 = int(cx2 - w2 // 2 + 0.5)
                y12 = int(cy2 - h2 // 2 + 0.5)
                x22 = int(cx2 + w2 // 2 + 0.5)
                y22 = int(cy2 + h2 // 2 + 0.5)
                if x22 >= image.width or y22 >= image.height:
                    continue

                areaO = w2 * h2
                if pat[4] / areaO < closeRatio:
                    break

                # patchImg = image[x12:x22, y12:y22]
                cropped = image.crop((x12, y12, x22, y22))

                sOutFileName = '%s/%s_%05d_%d.png' % (strOutFolder, sMainName, patchNdx, int(scaler * 100))
                dct = {'filename' : sOutFileName}
                lstBBxyxys = []
                for subBox in pat[0]:
                    ptx1 = subBox['x1'] - x12
                    pty1 = subBox['y1'] - y12
                    ptx2 = ptx1 + subBox['w']
                    pty2 = pty1 + subBox['h']
                    if ptx1 < 0 or pty1 < 0:
                        continue
                    img = cv2.cvtColor(np.asarray(cropped),cv2.COLOR_RGB2BGR)
                    img = cv2.resize(img,(outSize[0], outSize[1]), interpolation=cv2.INTER_LINEAR)
                    resizeRatio = outSize[0] / w2
                    [ptx1, ptx2] = [int(x * outSize[0] / w2 + 0.5) for x in [ptx1, ptx2]]
                    [pty1, pty2] = [int(x * outSize[1] / h2 + 0.5) for x in [pty1, pty2]]
                    # cv2.rectangle(img, (ptx1, pty1), (ptx2, pty2), (0,255,0), 1, 4)
                    cv2.imwrite(sOutFileName, img)
                    lstBBxyxys.append([ptx1, pty1, ptx2, pty2, subBox['tag']])
                dct['xyxys'] = lstBBxyxys
                patchNdx += 1
                lstPatches.append(dct)                
                break
            if i >= maxPatchPerImg:
                break
        return patchNdx, lstPatches        

    '''
    CutPatches: 切割出只包含一个人脸GT框的图片
    '''
    def CutPatches(self, strOutFolder, patchNdx, scalers=[0.8, 0.6, 0.45], outSize = [96,128], ndx=-1, strFile='', \
        maxPatchPerImg=16, allowedTags = ['*']):
        if ndx >= 0:
            strFile = list(self.dctFiles.keys())[ndx]
        item = self.dctFiles[strFile]
        lstPatches = []
        if len(item['xywhs']) < 1:
            return patchNdx, []
        sMainName = os.path.splitext(os.path.split(strFile)[1])[0]
        image = Image.open(self.provider.MapFile(strFile))
        
        wVsH = outSize[0] / outSize[1]
        lstXYWH = item['xywhs'].copy()
        if len(lstXYWH) > maxPatchPerImg:
            random.shuffle(lstXYWH)
            lstXYWH = lstXYWH[:maxPatchPerImg]
        for bbox in lstXYWH:
            if not bbox['tag'] in allowedTags:
                if '*' != allowedTags[0]:
                    continue            
            w  = bbox['w']
            h = bbox['h']
            x1 = bbox['x1']
            y1 = bbox['y1']
            tag = bbox['tag']
            cx = x1 + w // 2
            cy = y1 + h // 2
            for scaler in scalers:
                # 为每一个放大尺度都做一个patch，并且添加随机移动效果
                rand_clip_retry = 30
                for retryNdx in range(rand_clip_retry):
                    # w2, h2,cx2, cy2, x12, x22, y12, y22表示输出patch的几何信息
                    w2 = w / scaler
                    h2 = h / scaler
                    if w2 / h2 < wVsH:
                        w2 = h2 * wVsH
                    else:
                        h2 = w2 / wVsH
                    w2 = int(w2 + 0.5)
                    h2 = int(h2 + 0.5)
                    # 随机水平移动
                    xOfsRange = w2 * (1-scaler) / 2
                    yOfsRange = h2 * (1-scaler) / 2
                    xOfs = np.random.rand() * xOfsRange - xOfsRange / 2
                    yOfs = np.random.rand() * yOfsRange - yOfsRange / 2
                    cx2 = cx + xOfs
                    cy2 = cy + yOfs
                    # 必须确保长宽比
                    if cx2 - w2 / 2 < 0:
                        continue
                    if cy2 - h2 / 2 < 0:
                        continue
                    x12 = int(cx2 - w2 // 2 + 0.5)
                    y12 = int(cy2 - h2 // 2 + 0.5)
                    x22 = int(cx2 + w2 // 2 + 0.5)
                    y22 = int(cy2 + h2 // 2 + 0.5)
                    if x22 >= image.width or y22 >= image.height:
                        continue
                    # patchImg = image[x12:x22, y12:y22]
                    cropped = image.crop((x12, y12, x22, y22))
                    # cropped.save('patch_%d.png' % (patchNdx))
                    # ptx1, ptx2, pty1, pty2表示标注框
                    ptx1 = x1 - x12
                    pty1 = y1 - y12
                    ptx2 = ptx1 + w
                    pty2 = pty1 + h
                    if ptx1 < 0 or pty1 < 0:
                        continue
                    img = cv2.cvtColor(np.asarray(cropped),cv2.COLOR_RGB2BGR)
                    
                    sOutFileName = '%s/%s_%05d_%d.png' % (strOutFolder, sMainName, patchNdx, int(scaler * 100))
                 
                    img = cv2.resize(img,(outSize[0], outSize[1]), interpolation=cv2.INTER_LINEAR)
                    resizeRatio = outSize[0] / w2
                    [ptx1, ptx2] = [int(x * outSize[0] / w2 + 0.5) for x in [ptx1, ptx2]]
                    [pty1, pty2] = [int(x * outSize[1] / h2 + 0.5) for x in [pty1, pty2]]
                    # cv2.rectangle(img, (ptx1, pty1), (ptx2, pty2), (0,255,0), 1, 4)
                    cv2.imwrite(sOutFileName, img)
                    dct = {
                        'filename' : sOutFileName,
                        'xyxys' : [[ptx1, pty1, ptx2, pty2, tag]]
                    }
                    lstPatches.append(dct)
                    patchNdx += 1
                    break
        return patchNdx, lstPatches

    def ShowImage(self, ndx, isShow, allowedTags = ['*']):
        strFile = list(self.dctFiles.keys())[ndx]
        item = self.dctFiles[strFile]
        image = Image.open(self.provider.MapFile(strFile))
        imgWH = (image.width, image.height)
        width = 1 if imgWH[0] < 480 else 2
        img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        for bbox in item['xywhs']:
            if allowedTags[0] == '*' or bbox['tag'] in allowedTags:
                pt1 = (bbox['x1'], bbox['y1'])
                pt2 = (bbox['x1'] + bbox['w'] , bbox['y1'] + bbox['h'])
                col = (bbox['isOverIllumination'] * 255,191,bbox['occlusion'] * 255)
                cv2.rectangle(img, pt1, pt2, col, width, 4)
                pt1 = (pt1[0], pt1[1]+15)
                cv2.putText(img, 'pose:%d,occl:%d' % (bbox['isAtypicalPose'],  bbox['occlusion']) \
                    , pt1, cv2.FONT_HERSHEY_PLAIN, width, col)
        if isShow:
            cv2.imshow("OpenCV",img)
            cv2.waitKey()
        return [np.array(img), ndx, strFile]

    def ShowRandom(self, isShow=True, allowedTags = ['*']):
        ndx = np.random.randint(len(self.dctFiles))
        return self.ShowImage(ndx, isShow, allowedTags)


    def ShowRandomValidate(self, strOutFolder):
        with open("./%s/bboxes.json" % (strOutFolder)) as fd:
            lst = json.load(fd)
        cnt = len(lst)
        ndx = np.random.randint(cnt)
        item = lst[ndx]
        strFile = item['filename']
        image = Image.open(strFile)
        img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        for bbox in item['xyxys']:
            pt1 = (bbox[0], bbox[1])
            pt2 = (bbox[2], bbox[3])
            col = (0,255,0)
            cv2.rectangle(img, pt1, pt2, col, 1, 4)
        return np.array(img)

if __name__ == '__main__':
    exit(-1)