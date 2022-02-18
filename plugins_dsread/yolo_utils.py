try:
    import abstract_utils
except:
    import plugins_dsread.abstract_utils as abstract_utils
import os.path as path
import json
import glob
import zipfile
import os
import random
import time
from PIL import Image

def GetUtilClass():
    return YOLOUtils
def GetDSTypeName():
    return "YOLO"


class YOLOUtils(abstract_utils.AbstractUtils):

    def ParseAnno(self, lstAnnLines, imgW, imgH, dctNewCfg):
        lstAnns = []
        for sAnn in lstAnnLines:
            lstAnn = sAnn.split(' ')
            if len(lstAnn) == 5:
                lstAnns.append([float(x) for x in lstAnn])
        
        lstXywhs = []
        for yoloAnn in lstAnns:
            
            tag = 'tag%d' % (int(yoloAnn[0]))
            x = int((yoloAnn[1] - yoloAnn[3] / 2) * imgW + 0.5)
            y = int((yoloAnn[2] - yoloAnn[4] / 2) * imgH + 0.5)
            w = int(yoloAnn[3] * imgW + 0.5)
            h = int(yoloAnn[4] * imgH + 0.5)
            hVSw = h / w
            dirty = 0
            if hVSw < dctNewCfg['minHvsW'] or hVSw > dctNewCfg['maxHvsW']:
                if dctNewCfg['isSkipDirtyImg']:
                    return []
                dirty = 1
            dctItem = {
                'x1' : x,
                'y1' : y,
                'w': w,
                'h': h,
                'tag': tag,
                'blur': 0,
                'isOverIllumination': 0,
                'dirty' : dirty,
                'difficult' : 0,
                'pose' : 0,
                'occlusion' : 0
            }
            lstXywhs.append(dctItem)
        return lstXywhs

    def __init__(self, dsFolder = '.', setSel='train', dctCfg={}, callback=None):
        super(YOLOUtils, self).__init__(dsFolder, setSel, dctCfg, callback)
        self.dsFolder = dsFolder
        self.dctTags = dict()
        self.dctFiles = dict()
        self.setSel = setSel
        self.dctCfg = {}
        self.isTarMode = False
        self.tarRoots = []
        isSkipDirtyImg = False
        minHvsW, maxHvsW = 0.1, 10.0
        minGTPerImg, maxGTPerImg = 1, 50
        try:
            minHvsW = dctCfg['minHvsW']
            maxHvsW = dctCfg['maxHvsW']
            minGTPerImg = dctCfg['minGTPerImg']
            maxGTPerImg = dctCfg['maxGTPerImg']
            isSkipDirtyImg = dctCfg['isSkipDirtyImg']
        except:
            pass

        dctNewCfg = {
            'minHvsW' : minHvsW,
            'maxHvsW' : maxHvsW,
            'isSkipDirtyImg' : isSkipDirtyImg
        }
        t1 = time.time()
        # 扫描所有数据集
        pgs = 0
        cnt = 0
        for root, dirs, files in os.walk(dsFolder, topdown=False):
            root = root.replace('\\', '/')
            i = 0
            fileCnt = len(files)
            if root.find(setSel) < 0 and setSel != 'any':
                continue
            for name in files:
                if name[-3:] in ['jpg', 'png', 'peg']:
                    annoFile = root.replace('images', 'labels') + '/' + name[:-3] + 'txt'
                    imgFile = root + '/' + name
                    image = Image.open(imgFile)
                    imgW, imgH = image.width, image.height
                    with open(annoFile) as fd:
                        lstAnnLines = fd.read().split('\n')
                    lstBBoxes = self.ParseAnno(lstAnnLines, imgW, imgH, dctNewCfg)

                    if len(lstBBoxes) >= minGTPerImg and len(lstBBoxes) <= maxGTPerImg:
                        fileKey = imgFile.replace('/', '@')
                        if fileKey[1] == ':':
                            fileKey = fileKey[0] + '~' + fileKey[2:]                        
                        for xywh in lstBBoxes:
                            tag = xywh['tag']
                            if not tag in self.dctTags.keys():
                                self.dctTags[tag] = 1
                            else:
                                self.dctTags[tag] += 1                     
                        self.dctFiles[fileKey] = {
                            'cnt0' : len(lstBBoxes),
                            'cnt' : len(lstBBoxes),
                            'xywhs' : lstBBoxes
                        }
                    if i % 100 == 0:
                        pgs = 100 * (i / fileCnt)
                        if callback is not None:
                            callback(pgs)
                    i += 1
                    cnt += 1
        
        k2 = sorted(self.dctFiles.keys())
        dctRet = {}
        for (i,k) in enumerate(k2):
            dctRet[k] = self.dctFiles[k]
        self.dctFiles = dctRet

        t2 = time.time()
        dt = (t2 - t1)
        bkpt = 0

    def DelTags(self, lstTags:list):
            newDctFiles = dict()
            for key in self.dctFiles.keys():
                item = self.dctFiles[key]
                newXywhs = []            
                for xywh in item['xywhs']:
                    tag = xywh['tag']
                    if not tag in lstTags:
                        newXywhs.append(xywh)
                    else:
                        self.dctTags[tag] -= 1
                if len(newXywhs) != 0:
                    item['cnt'] = len(newXywhs)
                    item['xywhs'] = newXywhs
                    newDctFiles[key] = item
                else:
                    # 删光了
                    pass
            self.dctFiles = newDctFiles
            # 删除数量为0的tags
            newDictTags = dict()
            for k in self.dctTags.keys():
                if self.dctTags[k] != 0:
                    newDictTags[k] = self.dctTags[k]
            self.dctTags = newDictTags
                
    def CanDelTags(self):
        return True

    def IsSupportGTPerImg(self):
        return True

    def MapFile(self, strFileKey:str):
        if self.isTarMode:
            ret = None
        else:
            ret = strFileKey.replace('@', '/')
            if ret[1] == '~':
                ret = ret[0] + ':' + ret[2:]            
            ret = ret
        return ret


    def GetTagDict(self):
        return self.dctTags

    '''
        根据 fileKey反查在 dctFiles中的key
    '''
    def MapFileKey(self, fileKey):
        for ext in ['.jpg', '.png', '.jpeg']:
            key2 = fileKey + ext
            if key2 in self.dctFiles.keys():
                return key2
            
        return ''