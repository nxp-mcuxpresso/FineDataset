import tarfile
import os.path as path
import zipfile
import io
import glob
import xmltodict
class VOCUtils():

    def ParseAnno(self, an, dctNewCfg):

        imgWH = [int(an['size']['width']), int(an['size']['height'])]
        imgArea = imgWH[0] * imgWH[1]
        lstXywhs = []
        for obj in an['object']:
            bndBox = obj['bndbox']
            x = int(bndBox['xmin'])
            y = int(bndBox['ymin'])
            w = int(bndBox['xmax']) - x
            h = int(bndBox['ymax']) - y
            hVSw = h / w
            if hVSw < dctNewCfg['minHvsW'] or hVSw > dctNewCfg['maxHvsW']:
                continue
            self.setTags.add(obj['name'])
            dctItem = {
                'x1' : x,
                'y1' : y,
                'w': w,
                'h': h,
                'tag': obj['name'],
                'blur': 0,
                'difficult' : int(obj['difficult']),
                'isOverIllumination': 0,
                'occlusion' : int(obj['truncated']),
                'isAtypicalPose': 1 if obj['pose'] != 'Frontal' else 0,
                'isInvalid' : 0,
            }
            lstXywhs.append(dctItem)
        return lstXywhs

    def __init__(self, dsFolder = '.', setSel='train', dctCfg={}):
        self.dsFolder = dsFolder
        self.setTags = set()
        self.dctFiles = dict()
        self.setSel = setSel
        self.dctCfg = {}
        self.isTarMode = True
        self.tarRoot = ''
        minHvsW, maxHvsW = 0.1, 10.0
        try:
            minHvsW = dctCfg['minHvsW']
            maxHvsW = dctCfg['maxHvsW']
        except:
            pass
        dctNewCfg = {
            'minHvsW' : minHvsW,
            'maxHvsW' : maxHvsW
        }
        # 扫描所有数据集
        if path.exists(dsFolder + '/Annotations') and path.exists(dsFolder + '/JPEGImages'):
            self.isTarMode = False
            lstFiles = glob.glob(dsFolder + '/Annotations/*.xml')
            lstFiles = [x.replace('\\', '/') for x in lstFiles]
            for sFile in lstFiles:
                fd = open(sFile)
                an = xmltodict.parse(fd.read())['annotation']
                fd.close()
                if isinstance(an['object'],list) == False:
                    an['object'] = [an['object']]
                lstBBoxes = self.ParseAnno(an, dctNewCfg)
                if len(lstBBoxes) > 0:
                    fileKey = path.splitext(path.split(sFile)[-1])[0]
                    self.dctFiles[fileKey] = {
                        'cnt0' : len(an['object']),
                        'cnt' : len(lstBBoxes),
                        'xywhs' : lstBBoxes
                    }

        else:
            # 扫描所有符合setSel要求的VOC tar文件
            lstFiles = list(set(glob.glob('%s/*voc*.tar' % (dsFolder))).union(glob.glob('%s/*VOC*.tar' % (dsFolder))))
            if len(lstFiles) == 0:
                lstFiles = list(set(glob.glob('%s/*.tar' % (dsFolder))).union(glob.glob('%s/*.tar' % (dsFolder))))                
            if setSel != 'any':
                lstFiles = list(filter(lambda x: setSel in x, lstFiles))
            lstFiles = [ x.replace('\\', '/') for x in lstFiles ]
            self.lstTars = [tarfile.open(x) for x in lstFiles]
            if len(self.lstTars) == 0:
                return
            for tar in self.lstTars:
                tar.getmembers

            #查前缀
            for info in self.lstTars[0].getmembers():
                if info.name[-3:] == 'jpg':
                    self.tarRoot = path.split(info.name)[0]
                    break

            for (tarNdx, tar) in enumerate(self.lstTars):
                lstTarInfos = tar.getmembers()
                for info in lstTarInfos:
                    if info.name[-3:] == 'xml':
                        fd = tar.extractfile(info)
                        an = xmltodict.parse(fd.read())['annotation']
                        fd.close()
                        if isinstance(an['object'],list) == False:
                            an['object'] = [an['object']]
                        lstBBoxes = self.ParseAnno(an, dctNewCfg)
                        if len(lstBBoxes) > 0:
                            fileKey = '{:02}_'.format(tarNdx) + info.name.split('/')[-1][:-4]
                            self.dctFiles[fileKey] = {
                                'cnt0' : len(an['object']),
                                'cnt' : len(lstBBoxes),
                                'xywhs' : lstBBoxes
                            }                        

        bkpt = 0
        
    def MapFile(self, strFileKey:str):
        if self.isTarMode:
            ndx = int(strFileKey[:2])
            strFileKey = strFileKey[3:]
            tarf = self.lstTars[ndx]
            strFile = self.tarRoot + '/' + strFileKey + '.jpg'
            fd = tarf.extractfile(strFile)        
            data = fd.read()
            fd.close()
            ret = io.BytesIO(data)
        else:
            ret = self.dsFolder + '/JPEGImages/' + strFileKey + '.jpg'
        return ret

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

    
    def GetTagSet(self):
        return self.setTags

    '''
        根据 fileKey反查在 dctFiles中的key
    '''
    def MapFileKey(self, fileKey):
        if fileKey in self.dctFiles.keys():
            return fileKey
        return ''
if __name__ == '__main__':
    import patcher
    obj = patcher.Patcher(VOCUtils(dsFolder = 'Q:/datasets/voc07+12/training/VOC2007/mini_voc07',setSel='train'))
    print(len(obj.dctFiles))
    obj.ShowClusterRandom()
    print('done!')

