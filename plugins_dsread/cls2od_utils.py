try:
    import abstract_utils
except:
    import plugins_dsread.abstract_utils as abstract_utils
import os.path as path
import json
import glob
import zipfile
import io
import random
import time

def GetUtilClass():
    return None #[Cls2OD_Folder]
def GetDSTypeName():
    return None #['Cls2OD - by folder']
class InternalCls2ODUtils(abstract_utils.AbstractUtils):
    def __init__(self, dsFolder = '.', setSel='train', dctCfg = {}, callback=None, maxCnt=50000, isShuffle=True, isFineCls=True):
        super(InternalCls2ODUtils, self).__init__(dsFolder, setSel, dctCfg, callback, maxCnt, isShuffle)


class Cls2OD_Folder(InternalCls2ODUtils):
    def __init__(self, dsFolder = '.', setSel='train', dctCfg = {}, callback=None, maxCnt=50000, isShuffle=True):
        for root, dirs, files in os.walk(dsFolder, topdown=False):
            root = root.replace('\\', '/')
            for name in files:
                print(root, name)

        super(Cls2OD_Folder, self).__init__(dsFolder, setSel, dctCfg, callback, maxCnt, isShuffle, False)


class Fruits360Utils(Cls2OD_Folder):
    def __init__(self, dsFolder = '.', setSel='train', dctCfg = {}, callback=None, maxCnt=50000, isShuffle=True):
        super(Fruits360Utils, self).__init__(dsFolder, setSel, dctCfg, callback, maxCnt, isShuffle, False)
    @classmethod
    def GetClassName(coarse:str, fine:str):
        if ' ' in fine:
            pass