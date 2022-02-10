from abc import ABC, abstractmethod

def GetUtilClass():
    return None
def GetDSTypeName():
    return None

class AbstractExport(ABC):
    @abstractmethod
    def __init__(self, setSel='train', subsetSel='single', strRootPath = './outs'):
        self.dctFiles = dict()
        bkpt = 0

    def IsFixedSizeImg(self):
        return False, {'w':0, 'h':0}

    @abstractmethod
    def Export(self, maxCnt=1E7, callback=None, isTarOnly=True):
        return 0


if __name__ == '__main__':
    print('Not meant to run as main! done!')

