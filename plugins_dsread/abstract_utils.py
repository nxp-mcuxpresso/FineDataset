from abc import ABC, abstractmethod

def GetUtilClass():
    return None
def GetDSTypeName():
    return None


''' COCO数据集的要求
    图片文件：{setSel}{yyy}.zip
    标注文件：
        annotations_{包含setSel的字符串}{yyy}.zip
        其中，里面必须包含如下结构：
            annotations/instances_{setSel}{yyy}.json
'''
class AbstractUtils(ABC):
    @abstractmethod
    def __init__(self, dsFolder = '.', setSel='train', dctCfg = {}, callback=None, maxCnt=50000, isShuffle=True):
        self.dctFiles = dict()
        bkpt = 0

    def CanDelTags(self):
        return False
        
    def DelTags(self, lstTags:list):
        return False
    
    def IsFixedSizeImg(self):
        return False, {'w':0, 'h':0}

    def IsSupportGTPerImg(self):
        return False

    @abstractmethod
    def MapFile(self, strFile:str):
        return strFile

    @abstractmethod
    def GetTagDict(self):
        return dict()
    
    '''
        根据 fileKey反查在 dctFiles中的key
    '''
    @abstractmethod
    def MapFileKey(self, fileKey):
        return ''


if __name__ == '__main__':
    print('Not meant to run as main! done!')

