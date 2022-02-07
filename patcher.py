from PIL import Image
import numpy as np
import cv2
import os.path as path
import os
import math
import json
import random
import plugins_dsread.abstract_utils as abstract_utils
import traceback
from numpy.lib.type_check import isreal
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

def XYXY_CalcIoU(xyxy1, xyxy2):
    x1i,y1i,x2i,y2i= xyxy1[0],xyxy1[1],xyxy1[2],xyxy1[3]
    x1j,y1j,x2j,y2j= xyxy2[0],xyxy2[1],xyxy2[2],xyxy2[3]

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
        # 不相交的两个框
        iou = 0
    else:
        itArea = (x2It - x1It) * (y2It - y1It)
        unArea = (x2o - x1o) * (y2o - y1o)
        iou = itArea / unArea                        
    return iou

def XYXY_CalcIoU_indep(x1i,y1i,x2i,y2i,x1j,y1j,x2j,y2j):

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
        # 不相交的两个框
        iou = 0
    else:
        itArea = (x2It - x1It) * (y2It - y1It)
        unArea = (x2o - x1o) * (y2o - y1o)
        iou = itArea / unArea                        
    return iou    

class Patcher():
    def __init__(self, provider:abstract_utils.AbstractUtils):
        '''
        provider对象必须包含以下属性
            1. dctFiles: 指出数据集中的文件 
            2. MapFile: 返回经过映射的文件名，用于打包的数据集
        '''
        self.dctFiles = provider.dctFiles
        self.provider = provider

        # 子块包含的物体越多，对minCloseRate的要求就按比例降低，现在是按2**0.5
        # lstClsoeDecay[0] 用于含有3个物体的框, 依此类推
        self.lstCloseDecay = [1/math.log(x+3) for x in range(1, 20)]   
        bkpt = 0

    def GetClusters(self, ndx, minClose=0.5, isDraw=False, isShow=True, maxObjPerCluster=10, outWvsH=1, allowedTags=['*'], maxPairs=100):
        '''
        GetClusters: 获取图片中人脸的群聚。minClose表示bbox的面积之和除以这些bbox的总面积
        返回 [[二个脸的框], [三个脸的框], [多个脸的框]], opencv标注后的图
        '''
        strFile = list(self.dctFiles.keys())[ndx]
        item = self.dctFiles[strFile]

        if allowedTags[0] != '*':
            itemNew = item.copy()
            itemNew['xywhs'] = []
            for b1 in item['xywhs']:
                if b1['tag'] in allowedTags:
                    itemNew['xywhs'].append(b1)
            itemNew['cnt'] = len(itemNew['xywhs'])
            item = itemNew
        bboxCnt = len(item['xywhs'])
     
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

        def _DelSubsets(lstLess, lstMore):
            lstNewLess = []
            for less in lstLess:
                setLess = set(less[6])
                isMerged = False
                for more in lstMore:
                    setMore = set(more[6])
                    if setLess.issubset(setMore):
                        isMerged = True
                        break
                if isMerged == False:
                    lstNewLess.append(less)             
            return lstNewLess


        def _DelSubsetsFromSameList(lstIn0:list):
            if len(lstIn0) < 2:
                return lstIn0
            # 物体个数从多到少排序
            lstIn = lstIn0
            lstIn.sort(key=lambda x: len(x[6]), reverse=True)
            lstLeft = lstIn[:1]
            i = 0
            while(True):
                cnt = len(lstIn)                
                delMask = np.zeros(cnt, dtype='int32')
                while i < cnt:
                    setOuter = set(lstIn[i][6])
                    delCnt = 0
                    for j in range(cnt-1, i, -1):
                        setInner = set(lstIn[j][6])
                        if setInner.issubset(setOuter):
                            delMask[j] = 1
                            delCnt += 1
                    i += 1
                    if delCnt != 0:
                        lstLeft = []
                        for j in range(cnt):
                            if delMask[j] == 0:
                                lstLeft.append(lstIn[j])
                        lstIn = lstLeft  
                        break
                
                if delCnt == 0:
                    break
                delCnt = 0
            return lstLeft

        def XYXY_CalcIoU(xyxy1, xyxy2):
            x1i,y1i,x2i,y2i= xyxy1[0],xyxy1[1],xyxy1[2],xyxy1[3]
            x1j,y1j,x2j,y2j= xyxy2[0],xyxy2[1],xyxy2[2],xyxy2[3]

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
                # 不相交的两个框
                iou = 0
            else:
                itArea = (x2It - x1It) * (y2It - y1It)
                unArea = (x2o - x1o) * (y2o - y1o)
                iou = itArea / unArea                        
            return iou

        def XYXY_CalcIoU_indep(x1i,y1i,x2i,y2i,x1j,y1j,x2j,y2j):

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
                # 不相交的两个框
                iou = 0
            else:
                itArea = (x2It - x1It) * (y2It - y1It)
                unArea = (x2o - x1o) * (y2o - y1o)
                iou = itArea / unArea                        
            return iou            

        def _multiMerge(lstIn:list, maxObjPerCluster, minClsoeRate, outWvsH, lstCloseDecay):
            lstMulti = []
            lstLeft = []
            lstConsumed = []

            for i in range(len(lstIn)):
                multiIn = lstIn[i]
                isMerged = False
                if len(multiIn[6]) < maxObjPerCluster:
                    b1 = multiIn[3]
                    wi, hi, x1i, y1i, x2i, y2i, cxi, cyi, sqrtAi = _getValues(b1)                    
                    for j in range(i + 1, len(lstIn)):
                    
                        b2 = lstIn[j][3]
                        wj, hj, x1j, y1j, x2j, y2j, cxj, cyj, sqrtAj = _getValues(b2)
                        iou = XYXY_CalcIoU_indep(x1i,y1i,x2i,y2i,x1j,y1j,x2j,y2j)
                        if iou < 0.05:
                            continue
                        dctBbox = {
                            'x1' : x1o,
                            'y1' : y1o,
                            'w': x2o - x1o,
                            'h': y2o - y1o
                        }
                        # 检查dctBbox是不是已经出现在已有的里了
                        isRepeat = False
                        for tmp in lstMulti:
                            bB1 = tmp[3]
                            if bB1['x1'] == x1o and bB1['y1'] == y1o and bB1['w'] == x2o-x1o and bB1['h'] == y2o - y1o:
                                isRepeat = True
                                break
                        if isRepeat:
                            continue
                        areaO = dctBbox['w'] * dctBbox['h']
                        wVSh = dctBbox['w'] / dctBbox['h']
                        # 找到原始框标号的并集
                        set1 = set(multiIn[6])
                        set2 = set(lstIn[j][6])
                        set3 = set1.union(set2)
                        lstTmp = list(set3)
                        if len(lstTmp) > maxObjPerCluster:
                            random.shuffle(lstTmp)
                            lstTmp = lstTmp[:maxObjPerCluster]
                            lstTmp.sort()
                        # 找到物体框的并集
                        lstUn = [item['xywhs'][x] for x in lstTmp]
                        areaIJ = 0
                        for xywh in lstUn:
                            areaIJ += xywh['w'] * xywh['h']
                        closeRate = areaIJ / areaO
                        # 后面的步骤中，对于不符合输出高宽比的图像会尝试膨胀，相当于有效closeRate变低
                        aspectErr = wVSh / outWvsH if wVSh > outWvsH else outWvsH / wVSh
                        effCloseRate = closeRate / aspectErr
                        
                        decay = lstCloseDecay[len(lstTmp) - 3]
                        if effCloseRate >= minClsoeRate * decay:
                            lstMulti.append([lstUn, (x1o, y1o), (x2o, y2o), dctBbox, areaIJ, closeRate, lstTmp])
                        if False:
                            isMerged = True
                            if lstIn[j] not in lstConsumed:
                                lstConsumed.append(lstIn[j])
                
                
            lstLeft = _DelSubsets(lstIn, lstMulti)
            lstMulti= _DelSubsetsFromSameList(lstMulti)
            
            if False:
                if isMerged == True and multiIn not in lstConsumed:
                    lstConsumed.append(multiIn)
                if isMerged == False and multiIn not in lstConsumed:
                    lstLeft.append(multiIn)
            return lstMulti, lstLeft

        img = None
        if len(item['xywhs']) < 2:
            lstSolo = []
            b1 = item['xywhs'][0]
            wi, hi, x1i, y1i, x2i, y2i, cxi, cyi, sqrtAi = _getValues(item['xywhs'][0])
            dctBbox = {
                'x1' : x1i,
                'y1' : y1i,
                'w': x2i - x1i,
                'h': y2i - y1i,
            }
            if isDraw:
                image = Image.open(self.provider.MapFile(strFile))
                img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
            lstSolo.append([[b1], (x1i, y1i), (x2i, y2i), dctBbox, b1['w'] * b1['h'], 1.0, [0]])
            return [lstSolo,[],[]],np.array(img)

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
                closeRate = areaIJ / areaO
                wVSh = wo / ho
                aspectErr = wVSh / outWvsH if wVSh > outWvsH else outWvsH / wVSh
                # 当仅做2个的聚类时，closeRate在后面的步骤中处理
                if maxObjPerCluster == 2 or closeRate / aspectErr >= minClose:
                    dctBbox = {
                        'x1' : x1o,
                        'y1' : y1o,
                        'w': x2o - x1o,
                        'h': y2o - y1o,
                    }
                    lstPairs.append([[b1, b2], (x1o, y1o), (x2o, y2o), dctBbox, areaIJ, closeRate, [i, j]])
        lstPairs.sort(key=lambda x: x[5], reverse=True)
        lstTrints = []
        lstNewPairs = lstPairs # lstNewPairs后面存储没有被合并到三元组的对子
        lstNewTrints = []  # lstNewTrints 后面存储没有被合并到多元组的三元组
        lstMulti = []
        if maxObjPerCluster >= 3:
            # lstPairs.sort(key=lambda x:x[5], reverse=True)
            # 再获取靠近的三元组
            if len(lstPairs) > maxPairs:
                lstPairs = lstPairs[:maxPairs]
            
            lstNewPairs = []
            setIDs = set()
            for i in range(len(lstPairs)):
                pair = lstPairs[i]
                b1 = pair[3]
                wi, hi, x1i, y1i, x2i, y2i, cxi, cyi, sqrtAi = _getValues(b1)
                isMerged = False
                for j in range(bboxCnt):
                    if j == pair[6][0] or j == pair[6][1]:
                        # 重复的原始框
                        continue
                    if False:
                        setPair = set(pair[6])
                        for trint in lstTrints:
                            setTrint = set(trint[6])
                            if setPair.issubset(setTrint):
                                isMerged = True
                                break
                        if isMerged:
                            break
                    # 计算每个三元组的唯一ID。要求图片中原始框的数量不能超过1024
                    lstKeys=[pair[6][0], pair[6][1], j]
                    lstKeys.sort()
                    id = lstKeys[0] + (lstKeys[1] << 10) + (lstKeys[2] << 20)
                    if id in setIDs:
                        continue
                    setIDs.add(id)
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
                    areaIJ = pair[4] + wj * hj
                    closeRate = areaIJ / areaO
                    wVSh = wo / ho
                    aspectErr = wVSh / outWvsH if wVSh > outWvsH else outWvsH / wVSh
                    if closeRate / aspectErr >= minClose * self.lstCloseDecay[0]:
                        dctBbox = {
                            'x1' : x1o,
                            'y1' : y1o,
                            'w': x2o - x1o,
                            'h': y2o - y1o
                        }
                        lstTmp = pair[6].copy()
                        lstTmp.append(j)
                        lstTmp.sort()
                        lstTrints.append([pair[0] + [b2], (x1o, y1o), (x2o, y2o), dctBbox, areaIJ,closeRate, lstTmp])
                        isMerged = True
            lstTrints.sort(key=lambda x:x[5], reverse=True)
            lstNewPairs = _DelSubsets(lstPairs, lstTrints)
            lstNewTrints = lstTrints                

            # random.shuffle(lstTrints)

            if maxObjPerCluster >= 4 and len(lstTrints) > 0:
                lstTrints.sort(key=lambda x:x[5], reverse=True)
                if len(lstTrints) > maxPairs:
                    lstTrints = lstTrints[:maxPairs]        
                # 在三元组中合并交并比适中的
                lstMulti, lstLeft = _multiMerge(lstTrints, maxObjPerCluster, minClose, outWvsH, self.lstCloseDecay)
                # 最后的合并
                if len(lstMulti) >= 2:
                    lstMulti.sort(key=lambda x:x[5], reverse=False)
                    for mergeCnt in range(3):
                        if len(lstMulti) > maxPairs:
                            lstMulti = lstMulti[:maxPairs]
                        oldLen = len(lstMulti)
                        lstMulti, lstLeft = _multiMerge(lstMulti + lstLeft, maxObjPerCluster, minClose, outWvsH, self.lstCloseDecay)
                        newLen = len(lstMulti)
                        if newLen == oldLen or newLen == 1:
                            break
                lstNewTrints = lstLeft
        if isDraw:
            image = Image.open(self.provider.MapFile(strFile))
            img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
            if isDraw:
                for clst in lstNewPairs:
                    cv2.rectangle(img, clst[1], clst[2], (0, 0, 255), 2, 4)       
                for clst in lstNewTrints:
                    cv2.rectangle(img, clst[1], clst[2], (0, 255, 255), 2, 4)
                for clst in lstMulti:
                    cv2.rectangle(img, clst[1], clst[2], (255, 255, 255), 2, 4)     
                if isShow:
                    cv2.imshow("OpenCV",img)
                    cv2.waitKey()
        
        
        lstRet = [lstNewPairs, lstNewTrints, lstMulti]
        lstRet = [lstPairs, lstTrints, lstMulti]
        
        if len(lstRet) > 10:
            i = 0
        return lstRet, np.array(img)
    
    def ShowClusterRandom(self, isShow=True):
        while True:
            ndx = np.random.randint(len(self.dctFiles))
            lstRet, img = self.GetClusters(ndx, isShow=False, isDraw=True)
            if len(lstRet[2]) > 0:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
                cv2.imshow("OpenCV",img)
                cv2.waitKey()                   
                break
    '''
    CutPatches: 切割出包含一簇人脸的图片
    '''
    def CutClusterPatches(self, strOutFolder, patchNdx, scalers=[1.00, 0.8], outWH = [96, 128], \
            ndx=-1, strFile='', maxObjPerCluster=10, isAllowMorePerPatch=True, isSkipDirtyPatch=True, 
            minCloseRate=0.333, areaRateRange=[0,1], maxPatchPerImg=25, allowedTags=['*'], dbgSkips=[]):
        if ndx >= 0:
            strFile = list(self.dctFiles.keys())[ndx]
        wVsH = outWH[0] / outWH[1]
        lstRet, _ = self.GetClusters(ndx, minClose=minCloseRate, maxObjPerCluster=maxObjPerCluster, outWvsH=wVsH, allowedTags=allowedTags)
        if maxObjPerCluster < 3:
            lstOriPats = lstRet[0]
        elif maxObjPerCluster < 4:
            lstOriPats = lstRet[1] + lstRet[0]
        else:
            # 把含有更物体的框排在前面
            lstOriPats = lstRet[2] + lstRet[1] + lstRet[0]
        # item = self.dctFiles[strFile]
        if len(lstOriPats) < 1:
            return patchNdx, []
        sMainName = os.path.splitext(os.path.split(strFile)[1])[0]

        image = Image.open(self.provider.MapFile(strFile))
        lstPatches = []
        
        random.shuffle(lstOriPats)
        # pat打包格式：[[子框],(x1, y1),(x2,y2), bbox, 面积]
        newPatchCnt = 0
        skipBadAspectCnt = 0
        skipOutBoundCnt = 0
        skipBadSizeCnt = 0
        skipNotCloseCnt = 0
        skipTooDenseCnt = 0
        skipTooManyCnt = 0
        totalCnt = 0

        strFile = list(self.dctFiles.keys())[ndx]
        item = self.dctFiles[strFile]
        bboxCnt = len(item['xywhs'])
        lstOut4Filter = []
        gtUsedCnts = np.zeros(len(item['xywhs']))
        for (ip, pat) in enumerate(lstOriPats):
            totalCnt += len(pat[0])
            if len(pat[0]) > maxObjPerCluster:
                skipTooDenseCnt += len(pat[0])
                continue            
            if newPatchCnt >= maxPatchPerImg:
                skipTooManyCnt += len(pat[0])
                continue

            def _GetXYXY(xywh, wMax, hMax, scaler, wVsH, isAdaptiveExpand=True):
                w  = xywh['w']
                h = xywh['h']
                allowed_scaler = max(scaler, max(w/wMax, h/hMax))
                if scaler < allowed_scaler:
                    scaler = allowed_scaler
                x1 = xywh['x1']
                y1 = xywh['y1']
                cx = x1 + w // 2
                cy = y1 + h // 2 
                w2 = w / scaler
                h2 = h / scaler
                if isAdaptiveExpand == True:
                    if w2 / h2 < wVsH:
                        w2 = h2 * wVsH
                    else:
                        h2 = w2 / wVsH
                else:
                    if w2 / h2 > wVsH:
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
                    cx2 = w2 / 2
                if cy2 - h2 / 2 < 0:
                    cy2 = h2 / 2
                if cx2 + w2 / 2 > wMax:
                    cx2 = wMax - w2 / 2
                if cy2 + h2 / 2 > hMax:
                    cy2 = hMax - h2 / 2
                x12 = int(cx2 - w2 / 2 + 0.5)
                y12 = int(cy2 - h2 / 2 + 0.5)
                x22 = int(cx2 + w2 / 2 - 1 + 0.5)
                y22 = int(cy2 + h2 / 2 - 1 + 0.5)

                return cx, cy, w, h, x12, y12, x22, y22, w2, h2             
            
            bbox = pat[3]
                       
            aspectErr = bbox['w']/ bbox['h'] / wVsH
            if aspectErr > 25.0 or aspectErr < 0.04:
                skipBadAspectCnt += len(pat[0])
                continue
            # 留下比较大margin的scaler多试几次，每次都有随机性
            scalers = [0.8]*6 + [0.9]*3 + [1]
            outOriNdc = []
            isAbandonThisPatch = False
            for scaler in scalers:
                if isAbandonThisPatch:
                    break
                newSkipBadSizeCnt = 0
                newSkipOutBoundCnt = 0
                newSkipNotCloseCnt = 0
                # w2, h2,cx2, cy2, x12, x22, y12, y22表示输出patch的几何信息

                # 当子块的长宽比不符合输出的长宽比时，先投机地尝试扩大子块范围以符合长宽比要求
                cx, cy, w, h, x12,y12,x22,y22,w2,h2 = _GetXYXY(bbox, image.width, image.height, scaler, wVsH, True)
                if x22 >= image.width or y22 >= image.height:
                    
                    # 若扩大子块后导致它超过原图的边界，则老实地剪切子块中超出的部分
                    cx, cy, w, h, x12,y12,x22,y22,w2,h2 = _GetXYXY(bbox, image.width, image.height, scaler, wVsH, False)
                # 上面的操作导致需要重新计算closeRate
                
                # 跳过太小的图像
                if w2/outWH[0] < 0.4 or h2 / outWH[1] < 0.4:
                    newSkipBadSizeCnt += len(pat[0])
                    break
                
                # x12, y12 表示子块在原图的左上角坐标
                # x22, y22 表示子块在原图的右下角坐标
                
                # 若出现负坐标，则随机地均匀化两边的纯色边
                if x12 < 0:
                    dlt = int(-x12 * np.random.rand())
                    x12 += dlt
                    x22 += dlt
                if y12 < 0:
                    dlt = int(-y12 * np.random.rand())
                    y12 += dlt
                    y22 += dlt                
                cropped = image.crop((x12, y12, x22, y22))
                outXYXY = [x12, y12, x22, y22]
                

                sOutFileName = '%s/%s_%05d_%d.png' % ('.' + strOutFolder[6:], sMainName, patchNdx, int(scaler * 100))
                dct = {'filename' : sOutFileName}
                lstBBxyxys = []
                areaIJ = 0
                areaO = w2 * h2
                boxCnt = 0
                gtNewUsedCnts = np.zeros(len(item['xywhs']))
                for (i, subBox) in enumerate(item['xywhs']):
                    if not subBox['tag'] in allowedTags:
                        if '*' != allowedTags[0]:
                            continue
                   
                    # 去除经过剪裁后已经位于外面的物体框
                    gtW = subBox['w']
                    gtH = subBox['h']
                    ptx1 = subBox['x1'] - x12
                    pty1 = subBox['y1'] - y12
                    ptx2 = ptx1 + gtW
                    pty2 = pty1 + gtH
                    clipW = gtW
                    clipH = gtH
                    if ptx1 >= w2 or pty1 >= h2 or ptx2 < 0 or pty2 < 0:
                        newSkipOutBoundCnt += 1 
                        continue
                    if ptx1 < 0 or pty1 < 0 or ptx2 >= w2 or pty2 >= h2:
                        # 检查出界程度
                        isClipFromHead = False
                        if ptx1 < 0:
                            clipW = min(w2, gtW + ptx1)
                            ptx1 = 0
                        if pty1 < 0:
                            isClipFromHead = True
                            clipH = min(h2, gtH + pty1)
                            pty1 = 0
                        if ptx2 >= w2:
                            clipW = min(w2, gtW - (ptx2 + 1 - w2))
                            ptx2 = w2 - 1
                        if pty2 >= h2:
                            clipH = min(h2, gtH - (pty2 + 1 - h2))
                            pty2 = h2 - 1
                        iouMin = 0.6 if isClipFromHead == False else 0.8
                        if clipW * clipH / gtW / gtH < iouMin:
                            if clipW * clipH / w2 / h2 >= 0.18:
                                # 被剪裁的物体太大，很可能有大部分残留在子块区域中，
                                # 会对训练产生明显不良影响，所以宁可放弃这个子块
                                lstBBxyxys = []
                                # isAbandonThisPatch = True
                                break
                            newSkipOutBoundCnt += 1 
                            continue
                    # 落入子块中的物体，检查是不是dirty的
                    if isSkipDirtyPatch and subBox['dirty'] != 0:
                        lstBBxyxys = []
                        # isAbandonThisPatch = True
                        break                         
                    areaRate = clipW * clipH / w2 / h2
                    if areaRate < areaRateRange[0] or areaRate > areaRateRange[1]:
                        newSkipBadSizeCnt += 1
                        continue

                    if not isAllowMorePerPatch and boxCnt >= len(pat[0]):
                        lstBBxyxys = []
                        break

                    areaIJ += clipW * clipH
                    [ptx1, ptx2] = [int(x * outWH[0] / w2 + 0.5) for x in [ptx1, ptx2]]
                    [pty1, pty2] = [int(x * outWH[1] / h2 + 0.5) for x in [pty1, pty2]]
                    if ptx2 >= outWH[0]:
                        ptx2 = outWH[0] - 1
                    if pty2 >= outWH[1]:
                        pty2 = outWH[1] - 1
                    # cv2.rectangle(img, (ptx1, pty1), (ptx2, pty2), (0,255,0), 1, 4)
               
                    # 每个GT框最多使用3次
                    if gtUsedCnts[i] >= 3:
                        lstBBxyxys = []
                        gtNewUsedCnts = 0
                        break
                    gtNewUsedCnts[i] = 1        
                    lstBBxyxys.append([ptx1, pty1, ptx2, pty2, subBox['tag']])
                    boxCnt += 1
                    outOriNdc.append(i)
                if not isAllowMorePerPatch and boxCnt > len(pat[0]):
                    continue
                closeRate = areaIJ / areaO
                if closeRate < minCloseRate * self.lstCloseDecay[len(pat[0])]:
                    newSkipNotCloseCnt = len(pat[0])
                    continue
                
                skipBadSizeCnt += newSkipBadSizeCnt
                skipOutBoundCnt += newSkipOutBoundCnt
                if len(lstBBxyxys) > 0:
                    # 检查和基于本张图的现有子块是否过于重叠
                    isReject = False                    
                    for j, outChk in enumerate(lstOut4Filter):
                        iou = XYXY_CalcIoU(outChk[0], outXYXY)
                        if iou >=0.5:
                            set1, set2 = set(outOriNdc), set(outChk[1])
                            if set1 == set2:
                                isReject = True
                                break
                            else:
                                un = set1.union(set2)
                                it = set1.intersection(set2)
                                if len(it) / len(un) >= 0.8:
                                    isReject = True
                                break
                    if isReject == True:
                        continue
                    gtUsedCnts += gtNewUsedCnts
                    img = cv2.cvtColor(np.asarray(cropped),cv2.COLOR_RGB2BGR)
                    img = cv2.resize(img,(outWH[0], outWH[1]), interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite('./outs/' + sOutFileName[2:], img)
                    dct['xyxys'] = lstBBxyxys
                    patchNdx += 1
                    newPatchCnt += 1
                    lstPatches.append(dct)
                    lstOut4Filter.append([outXYXY, outOriNdc])
                break
            skipNotCloseCnt += newSkipNotCloseCnt

        if isinstance(dbgSkips, list) and len(dbgSkips) >= 6:
            dbgSkips[0] += totalCnt
            dbgSkips[1] += skipBadAspectCnt
            dbgSkips[2] += skipNotCloseCnt
            dbgSkips[3] += skipOutBoundCnt
            dbgSkips[4] += skipBadSizeCnt
            dbgSkips[5] += skipTooDenseCnt
            dbgSkips[6] += skipTooManyCnt
        return patchNdx, lstPatches

    '''
    CutPatches: 切割出只包含一个人脸GT框的图片
    '''
    def CutPatches(self, strOutFolder, patchNdx, scalers=[0.8, 0.6, 0.45], areaRateRange=[0, 1], outWH = [96,128], \
        ndx=-1, strFile='', isSkipDirtyPatch=True, maxPatchPerImg=32, allowedTags = ['*'], dbgSkips=[]):
        if ndx >= 0:
            strFile = list(self.dctFiles.keys())[ndx]
        item = self.dctFiles[strFile]
        lstPatches = []
        if len(item['xywhs']) < 1:
            return patchNdx, []
        sMainName = os.path.splitext(os.path.split(strFile)[1])[0]
        image = Image.open(self.provider.MapFile(strFile))
        
        wVsH = outWH[0] / outWH[1]
        lstXYWH = item['xywhs'].copy()
        skipOutBoundCnt = 0
        skipBadSizeCnt = 0
        if len(lstXYWH) > maxPatchPerImg:
            random.shuffle(lstXYWH)
            lstXYWH = lstXYWH[:maxPatchPerImg]
        totalCnt = 0
        for bbox in lstXYWH:
            if not bbox['tag'] in allowedTags:
                if '*' != allowedTags[0]:
                    continue
            if isSkipDirtyPatch == True and bbox['dirty'] != 0:
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
                totalCnt += 1
                rand_clip_retry = 30
                for retryNdx in range(rand_clip_retry):
                    newSkipOutBoundCnt = 0
                    newSkipBadSizeCnt = 0
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
                        cx2 = w / 2
                    if cy2 - h2 / 2 < 0:
                        cy2 = h / 2
                    x12 = int(cx2 - w2 // 2 + 0.5)
                    y12 = int(cy2 - h2 // 2 + 0.5)
                    x22 = int(cx2 + w2 // 2 + 0.5)
                    y22 = int(cy2 + h2 // 2 + 0.5)
                    if x22 >= image.width or y22 >= image.height:
                        newSkipOutBoundCnt += 1
                        continue
                    # patchImg = image[x12:x22, y12:y22]
                    cropped = image.crop((x12, y12, x22, y22))
                    # cropped.save('patch_%d.png' % (patchNdx))
                    # ptx1, ptx2, pty1, pty2表示标注框
                    ptx1 = x1 - x12
                    pty1 = y1 - y12
                    ptx2 = ptx1 + w
                    pty2 = pty1 + h
                    if ptx1 < 0 or pty1 < 0 or ptx2 >= w2 or pty2 >= h2:
                        newSkipOutBoundCnt += 1
                        continue
                    areaRate = w * h / w2 / h2
                    if areaRate < areaRateRange[0] or areaRate > areaRateRange[1]:
                        newSkipBadSizeCnt += 1
                        continue
                    img = cv2.cvtColor(np.asarray(cropped),cv2.COLOR_RGB2BGR)
                    
                    sOutFileName = '%s/%s_%05d_%d.png' % (strOutFolder, sMainName, patchNdx, int(scaler * 100))
                 
                    img = cv2.resize(img,(outWH[0], outWH[1]), interpolation=cv2.INTER_LINEAR)
                    resizeRatio = outWH[0] / w2
                    [ptx1, ptx2] = [int(x * outWH[0] / w2 + 0.5) for x in [ptx1, ptx2]]
                    [pty1, pty2] = [int(x * outWH[1] / h2 + 0.5) for x in [pty1, pty2]]
                    # cv2.rectangle(img, (ptx1, pty1), (ptx2, pty2), (0,255,0), 1, 4)
                    cv2.imwrite(sOutFileName, img)
                    dct = {
                        'filename' : '.' + sOutFileName[6:],
                        'xyxys' : [[ptx1, pty1, ptx2, pty2, tag]]
                    }
                    lstPatches.append(dct)
                    patchNdx += 1
                    break
                skipOutBoundCnt += newSkipOutBoundCnt
                skipBadSizeCnt += newSkipBadSizeCnt
        if isinstance(dbgSkips, list) and len(dbgSkips) >= 6:
            dbgSkips[0] += totalCnt
            dbgSkips[3] += skipOutBoundCnt
            dbgSkips[4] += skipBadSizeCnt
        return patchNdx, lstPatches

    def ShowImageFile(self, fileKey, isShow, allowedTags = ['*']):
        item = self.dctFiles[fileKey]
        image = Image.open(self.provider.MapFile(fileKey))
        imgWH = (image.width, image.height)
        width = 1 if imgWH[0] < 480 else 2
        img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        drawedCnt = 0
        for (i, bbox) in enumerate(item['xywhs']):
            if allowedTags[0] == '*' or bbox['tag'] in allowedTags:
                drawedCnt += 1
                pt1 = (bbox['x1'], bbox['y1'])
                pt2 = (bbox['x1'] + bbox['w'] , bbox['y1'] + bbox['h'])

                bgr = (bbox['isOverIllumination'] * 255,223,bbox['occlusion'] * 255)
                if bbox['dirty'] != 0:
                    bgr = (bgr[0]/2, bgr[1]/2, bgr[2]/2)
                cv2.rectangle(img, pt1, pt2, bgr, width, 4)
                pt1 = (pt1[0], pt1[1]+15)
                cv2.putText(img, '%s,%d' % (bbox['tag'],  i) \
                    , pt1, cv2.FONT_HERSHEY_PLAIN, width, bgr)
        if isShow:
            cv2.imshow("OpenCV",img)
            cv2.waitKey()
        return [np.array(img), fileKey, drawedCnt]

    def ShowImage(self, ndx, isShow, allowedTags = ['*']):
        strFile = list(self.dctFiles.keys())[ndx]
        ret = self.ShowImageFile(strFile, isShow, allowedTags)
        return ret + [ndx]
    def ShowRandom(self, isShow=True, allowedTags = ['*']):
        while True:
            ndx = np.random.randint(len(self.dctFiles))
            lstRet = self.ShowImage(ndx, isShow, allowedTags)
            if lstRet[2] > 0:
                break
        return lstRet
    def ShowAt(self, ndx, isShow=True, allowedTags = ['*']):
        return self.ShowImage(ndx, isShow, allowedTags)

    def ShowRandomValidate(self, strOutFolder=''):
        try:
            with open("%s/bboxes.json" % (strOutFolder)) as fd:
                lst = json.load(fd)
        except:
            return None, -1, None
        cnt = len(lst)
        if cnt == 0:
            return None, -1, None
        ndx = np.random.randint(cnt)
        item = lst[ndx]
        strFile = './outs/' + item['filename'][2:]
        image = Image.open(strFile)
        img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        for bbox in item['xyxys']:
            pt1 = (bbox[0], bbox[1])
            pt2 = (bbox[2], bbox[3])
            col = (0,255,0)
            cv2.rectangle(img, pt1, pt2, col, 1, 4)
            pt1 = (pt1[0], pt1[1]+15)
            cv2.putText(img, '%s' % (bbox[4]) \
                , pt1, cv2.FONT_HERSHEY_PLAIN, 1, col)
        return np.array(img), ndx, item
    def FilterTags(self, tagsToDel:list):
        ret = self.provider.DelTags(tagsToDel)
        self.dctFiles = self.provider.dctFiles
        return ret                    
if __name__ == '__main__':
    exit(-1)