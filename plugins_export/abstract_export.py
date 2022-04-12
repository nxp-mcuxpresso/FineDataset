# Copyright 2016-2022 NXP
# SPDX-License-Identifier: MIT
from abc import ABC, abstractmethod
import os.path as path
import glob

def GetUtilClass():
    return None
def GetDSTypeName():
    return None
class AbstractExport(ABC):
    @abstractmethod
    def __init__(self, setSel='train', subsetSel='single', strRootPath = './outs'):
        self.setSel = setSel
        self.subsetSel = subsetSel
        
        lst = glob.glob('%s/out_%s_%s*' % (strRootPath, setSel, subsetSel))
        lst = [x.replace('\\', '/') for x in lst]
        lst = list(filter(lambda x: path.isdir(x) == True, lst))
        self.lstInPaths = lst
        
        self.strRootPath = strRootPath

    def IsFixedSizeImg(self):
        return False, {'w':0, 'h':0}

    @abstractmethod
    def Export(self, maxCnt=1E7, callback=None, isTarOnly=True):
        return 0


if __name__ == '__main__':
    print('Not meant to run as main! done!')

