import tarfile
import os.path as path
from typing import DefaultDict
import zipfile
import io
import glob
import json
import traceback
import time
try:
    import abstract_utils
except:
    import plugins_dsread.abstract_utils as abstract_utils
def GetDSTypeName():
    return "Sub dataset"

def GetUtilClass():
    return SubdatasetUtils


class SubdatasetUtils(abstract_utils.AbstractUtils):

    def __init__(self, dsFolder = '.', setSel='train', dctCfg={}, callback=None):
        super(SubdatasetUtils, self).__init__(dsFolder, setSel, dctCfg, callback)
        self.dsFolder = dsFolder
        self.dctTags = dict()
        self.dctFiles = dict()
        self.setSel = setSel
        self.dctCfg = {}

        minHvsW, maxHvsW = 0.1, 10.0
        minGTPerImg, maxGTPerImg = 1, 50
        try:
            minHvsW = dctCfg['minHvsW']
            maxHvsW = dctCfg['maxHvsW']
            minGTPerImg = dctCfg['minGTPerImg']
            maxGTPerImg = dctCfg['maxGTPerImg']
        except:
            pass

        dctNewCfg = {
            'minHvsW' : minHvsW,
            'maxHvsW' : maxHvsW
        }

        if path.exists(dsFolder):
            with open(dsFolder + '/bboxes.json') as fd:
                lstIn = json.load(fd)
            dctIn = dict()
            for item in lstIn:
                in_xywhs = [[x[0], x[1], x[2] - x[0], x[3] - x[1], x[4]] for x in item['xyxys']]
                xywhs = filter(lambda x: x[3] / x[2] >= minHvsW and x[3] / x[2] <= maxHvsW, in_xywhs)
                xywhs = filter(lambda x: len(x) >= minGTPerImg and len(x) <= maxGTPerImg, xywhs)
                xywhs = list(xywhs)
                for xywh in xywhs:
                    try:
                        self.dctTags[xywh[4]] += 1
                    except:
                        self.dctTags[xywh[4]] = 1
                fileKey = item['filename'].split('/')[-1]
                xywhs = [{'x1': x[0], 'y1': x[1], 'w': x[2], 'h': x[3], 'tag': x[4],
                        'blur': 0,
                        'isExaggerate': 0,
                        'isOverIllumination': 0,
                        'occlusion' : 0,
                        'pose': 0,
                        'isInvalid' : 0,
                } for x in xywhs]
                self.dctFiles[fileKey] = {
                    'cnt0': len(in_xywhs),
                    'cnt': len(xywhs),
                    'xywhs' : xywhs
                }
        bkpt = 0
    
    def IsSupportGTPerImg(self):
        return True

    def MapFile(self, strFileKey:str):
        ret = self.dsFolder + '/' + strFileKey
        return ret        

    
    def GetTagDict(self):
        return self.dctTags

    '''
        根据 fileKey反查在 dctFiles中的key
    '''
    def MapFileKey(self, fileKey):
        if fileKey in self.dctFiles.keys():
            return fileKey
        return ''
if __name__ == '__main__':
    try:
        import patcher
    except:
        import sys
        import importlib
        newPath = path.abspath('./')
        if not path.exists(newPath + '/patcher.py'):
            newPath = path.abspath('../')
        sys.path.append(newPath)
        patcher = importlib.import_module('patcher')
    obj = patcher.Patcher(SubdatasetUtils(dsFolder = 'Q:/gitrepos/subdataset/outs/out_train_multi',setSel='any'))
    print(len(obj.dctFiles))
    obj.ShowClusterRandom()
    print('done!')

