import os.path as path
import zipfile
import io
try:
    import abstract_utils
except:
    import plugins_dsread.abstract_utils as abstract_utils
import glob
def GetDSTypeName():
    return "Wider Face"

def GetUtilClass():
    return WFUtils
class WFUtils(abstract_utils.AbstractUtils):

    def _Init_ProcAnno(self, preNdx, isZip, lstLines, dsFolder, setSel,  dctCfg:dict, callback):
        STATE_WANT_FILENAME = 0
        STATE_WANT_BBOX_CNT = 1
        STATE_WANT_BBOX_ITEM = 2
        st = STATE_WANT_FILENAME
        bboxRem = 0
        lstBBoxes = []
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
        
        def default_callback(pgs, msg, in_callback):
            print(pgs, msg)
            if in_callback is not None:
                in_callback(pgs, msg)
        print('scaning')
        cnt = len(lstLines)
        dirtyCnt = 0
        for (i, strLine) in enumerate(lstLines):
            strLine = strLine.strip()
            if st == STATE_WANT_FILENAME:
                if i % 500 == 0:
                    pgs = 100 * i / cnt
                    default_callback(pgs, '扫描%s' % (setSel), callback)
                
                strFileName = 'WIDER_%s/images/%s' % (setSel, strLine)
                if isZip == False:
                    strFileName = dsFolder + '/' + strFileName
                fileKey = '{:02}_'.format(preNdx) + strLine.split('/')[-1]
                self.dctFileKeyMapper[fileKey] = strFileName
                st = STATE_WANT_BBOX_CNT
            elif st == STATE_WANT_BBOX_CNT:
                nBBoxCnt = int(strLine)
                st = STATE_WANT_BBOX_ITEM
                bboxRem = nBBoxCnt
                lstBBoxes = []
                dirtyCnt = 0
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
                        'pose': lstVals[9],
                        'isInvalid' : lstVals[7],
                        'dirty':0
                    }

                    if dctItem['isInvalid'] == 0 and dctItem['blur'] < 2:# and dctItem['pose'] == 0 and dctItem['occlusion'] < 1:
                        if lstVals[2] * lstVals[3] >= 16*16 and lstVals[2] != 0:
                            hVsW = lstVals[3] / lstVals[2]
                            if hVsW >= minHvsW and hVsW <= maxHvsW:
                                lstBBoxes.append(dctItem)
                            else:
                                if isSkipDirtyImg == False:
                                    dctItem['dirty'] = 1
                                    lstBBoxes.append(dctItem)
                                else:
                                    dirtyCnt += 1                               
                    bboxRem -= 1
                    if bboxRem == 0:
                        st = STATE_WANT_FILENAME
                        if len(lstBBoxes) >= minGTPerImg and len(lstBBoxes) <= maxGTPerImg :
                            if isSkipDirtyImg == False or dirtyCnt == 0:
                                self.dctTags['face'] += len(lstBBoxes)
                                self.dctFiles[fileKey] = {
                                    'cnt0' : nBBoxCnt,
                                    'cnt' : len(lstBBoxes),
                                    'xywhs' : lstBBoxes
                                }
                else:
                    st = STATE_WANT_FILENAME
        bkpt = 0
    
    def __init__(self, dsFolder = '.', setSel='train', dctCfg={}, callback=None):
        super(WFUtils,self).__init__(dsFolder, setSel, dctCfg, callback)
        self.dsFolder = dsFolder
        self.setSel = setSel
        self.dsFolerLen = len(dsFolder)
        self.dctCfg = dict()
        self.dctFiles = dict()
        self.dctFileKeyMapper = {}        

        # 数据源列表，指出目录或者压缩文件
        self.lstSrcs = list()

        self.zfDataFile = None
        self.dctTags = {'face':0}
        self.isZipMode = False

        lstLines = []
        if setSel == 'any':
            lstSetSel = ['train', 'val']
        else:
            lstSetSel = [setSel]
        for (i, curSet) in enumerate(lstSetSel):
            pathBBox = '%s/wider_face_split/wider_face_%s_bbx_gt.txt' % (dsFolder, curSet)
            if path.exists(pathBBox):
                fd = open(pathBBox)
                lstLines = fd.readlines()
                fd.close()
                self._Init_ProcAnno(i, False, lstLines, dsFolder, curSet, dctCfg, callback)
                self.lstSrcs.append('%s/WIDER_%s' % (dsFolder, curSet))
            else:
                # zip 模式
                zipAnnoPath = '%s/wider_face_split.zip' % (dsFolder)
                zfAn = zipfile.ZipFile(zipAnnoPath)
                fd = zfAn.open('wider_face_split/wider_face_%s_bbx_gt.txt' % (curSet))
                lstLines = fd.readlines()
                fd.close() 
                
                datFilePath = glob.glob('%s/WIDER_%s*.zip' % (dsFolder, curSet))
                if len(datFilePath) == 0:
                    continue
                datFilePath = datFilePath[0]
                zfImg = zipfile.ZipFile(datFilePath)
                self.lstSrcs.append(zfImg)
                lstLines = [str(x, encoding='utf-8')[:-1] for x in lstLines]

            self._Init_ProcAnno(i, True, lstLines, dsFolder, curSet, dctCfg, callback)

        k2 = sorted(self.dctFiles.keys())

        dctRet = {}
        for (i,k) in enumerate(k2):
            dctRet[k] = self.dctFiles[k]
        self.dctFiles = dctRet
    
    def MapFile(self, strFileKey:str):

        srcNdx = int(strFileKey[:2])
        sFilePath = self.dctFileKeyMapper[strFileKey]
        if isinstance(self.lstSrcs[0], str):
            sFilePath = '%s/' % (self.dsFolder) + self.dctFileKeyMapper[strFileKey]
            if path.exists(sFilePath):
                return sFilePath
        else:
            zf = self.lstSrcs[srcNdx]
            fd = zf.open(sFilePath)
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
    obj = patcher.Patcher(WFUtils(dsFolder = 'q:/datasets/wider_face',setSel='any'))
    print(len(obj.dctFiles))
    obj.ShowClusterRandom()
    print('done!')

