import os.path as path
import zipfile
import io
class WFUtils():
    def __init__(self, dsFolder = '.', setSel='train', dctCfg={}, callback=None):
        self.dsFolder = dsFolder
        self.setSel = setSel
        self.dctCfg = {}
        self.pathBBox = '%s/wider_face_split/wider_face_%s_bbx_gt.txt' % (dsFolder, setSel)
        self.dctFiles = None
        self.zfDataFile = None
        self.dctTags = {'face':0}
        self.isZipMode = False

        lstLines = []
        if setSel == 'any':
            lstSetSel = ['train', 'val', 'test']
        else:
            lstSetSel = [setSel]
        for setSel in lstSetSel:
            if path.exists(self.pathBBox):
                fd = open(self.pathBBox)
                lstLines = fd.readlines()
                fd.close()
            else:
                self.zipAnnoPath = '%s/wider_face_split.zip' % (dsFolder)
                zf = zipfile.ZipFile(self.zipAnnoPath)
                fd = zf.open('wider_face_split/wider_face_train_bbx_gt.txt')
                lstLines = fd.readlines()
                fd.close() 
                lstLines = [str(x, encoding='utf-8')[:-1] for x in lstLines]
            if len(lstLines) > 0:
                break

        STATE_WANT_FILENAME = 0
        STATE_WANT_BBOX_CNT = 1
        STATE_WANT_BBOX_ITEM = 2
        st = STATE_WANT_FILENAME
        bboxRem = 0
        lstBBoxes = []
        self.dctFiles = {}
        self.dctFileKeyMapper = {}
        print('scaning')
        cnt = len(lstLines)
        for (i, strLine) in enumerate(lstLines):
            strLine = strLine.strip()
            if st == STATE_WANT_FILENAME:
                if i % 100 == 0:
                    pgs = 100 * i / cnt
                    if callback is not None:
                        callback(pgs)
                strFileName = '%s/WIDER_%s/images/%s' % (dsFolder, setSel, strLine)
                fileKey = strLine.split('/')[-1]
                self.dctFileKeyMapper[fileKey] = strFileName
                st = STATE_WANT_BBOX_CNT
            elif st == STATE_WANT_BBOX_CNT:
                nBBoxCnt = int(strLine)
                st = STATE_WANT_BBOX_ITEM
                bboxRem = nBBoxCnt
                lstBBoxes = []
            elif st == STATE_WANT_BBOX_ITEM:
                if nBBoxCnt > 0:
                    lstVals = [int(x) for x in strLine.split(' ')]
                    dctItem = {
                        'x1' : lstVals[0],
                        'y1' : lstVals[1],
                        'w': lstVals[2],
                        'h': lstVals[3],
                        'tag': 'face',
                        'blur': lstVals[4],
                        'isExaggerate': lstVals[5],
                        'isOverIllumination': lstVals[6],
                        'occlusion' : lstVals[8],
                        'isAtypicalPose': lstVals[9],
                        'isInvalid' : lstVals[7],
                    }
                    if dctItem['isInvalid'] == 0 and dctItem['blur'] < 2 and dctItem['isAtypicalPose'] == 0 and dctItem['occlusion'] < 1:
                        if lstVals[2] * lstVals[3] >= 36*36 and lstVals[2] != 0 and lstVals[3] / lstVals[2] < 2.0:
                            lstBBoxes.append(dctItem)
                            self.dctTags['face'] += 1
                    bboxRem -= 1
                    if bboxRem == 0:
                        st = STATE_WANT_FILENAME
                        if len(lstBBoxes) > 0:
                            self.dctFiles[fileKey] = {
                                'cnt0' : nBBoxCnt,
                                'cnt' : len(lstBBoxes),
                                'xywhs' : lstBBoxes
                            }
                else:
                    st = STATE_WANT_FILENAME
        k2 = sorted(self.dctFiles.keys())

        dctRet = {}
        for (i,k) in enumerate(k2):
            dctRet[k] = self.dctFiles[k]
        self.dctFiles = dctRet
    
    def MapFile(self, strFile:str):
        sFilePath = self.dctFileKeyMapper[strFile]
        if path.exists(sFilePath):
            return sFilePath
        if self.zfDataFile is not None:
            self.zfDataFile = zipfile('%s/WIDER_%s.zip' % (self.dsFolder, self.setSel))
        fd = self.zfDataFile.open(strFile)
        data = fd.read()
        fd.close()
        ret = io.BytesIO(data)
        return ret        

    
    def GetTagDict(self):
        return self.dctTags

    '''
        根据 fileKey反查在 dctFiles中的key
    '''
    def MapFileKey(self, fileKey):
        mapped  = fileKey
        exts = ['.jpg', '.jpeg']
        for ext in exts:
            sKey = mapped + ext
            if  sKey in self.dctFiles.keys():
                return mapped + ext
        return ''

if __name__ == '__main__':
    import patcher
    obj = patcher.Patcher(WFUtils(dsFolder = 'q:/datasets/wider_face/',setSel='train'))
    print(len(obj.dctFiles))
    obj.ShowClusterRandom()
    print('done!')

