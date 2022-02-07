import tarfile
import os.path as path
import zipfile
import io
import glob
import xmltodict
import xml.dom.minidom
import traceback
import time
try:
    import abstract_utils
except:
    import plugins_dsread.abstract_utils as abstract_utils
def GetDSTypeName():
    return "VOC"

def GetUtilClass():
    return VOCUtils

def ParseNode(node):
    if node.firstChild == node.lastChild:
        return node.firstChild.nodeValue
    else:
        dct2 = dict()
        for child in node.childNodes:
            keyName = child.localName
            if keyName is None:
                continue
            dct2[keyName] = ParseNode(child)
        return dct2

def Xml2Dict(xmlData):
    dom1=xml.dom.minidom.parseString(xmlData)
    root=dom1.documentElement
    dct = dict()
    for node in root.childNodes:
        if node.nodeType == 3 and node.firstChild is None:
            # 字符串节点
            if node.localName is None:
                # 缩进节点
                continue
        elif node.nodeType == 1:
            # element节点
            key = node.localName
            parsed = ParseNode(node)
            if key in dct.keys():
                if isinstance(dct[key], list):
                    dct[key].append(parsed)
                else:
                    dct[key] = [dct[key], parsed]
            else:
                dct[node.localName] = parsed
    return dct


class VOCUtils(abstract_utils.AbstractUtils):

    def ParseAnno(self, an, dctNewCfg):

        imgWH = [int(an['size']['width']), int(an['size']['height'])]
        imgArea = imgWH[0] * imgWH[1]
        lstXywhs = []
        for obj in an['object']:
            bndBox = obj['bndbox']
            x = int(float(bndBox['xmin']))
            y = int(float(bndBox['ymin']))
            w = int(float(bndBox['xmax'])) - x
            h = int(float(bndBox['ymax'])) - y
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
                'tag': obj['name'],
                'blur': 0,
                'isOverIllumination': 0,
                'dirty' : dirty,
            }
            optionalKeyMap = ['difficult', 'pose', 'occlusion']
            for (i, optionalKey) in enumerate(['difficult', 'pose', 'truncated']):
                dctItem[optionalKeyMap[i]] = 0
                try:
                    if optionalKey in obj.keys():
                        dctItem[optionalKeyMap[i]] = int(obj[optionalKey])
                except:
                    pass
            lstXywhs.append(dctItem)
        return lstXywhs

    def __init__(self, dsFolder = '.', setSel='train', dctCfg={}, callback=None):
        super(VOCUtils, self).__init__(dsFolder, setSel, dctCfg, callback)
        self.dsFolder = dsFolder
        self.dctTags = dict()
        self.dctFiles = dict()
        self.setSel = setSel
        self.dctCfg = {}
        self.isTarMode = True
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
        if path.exists(dsFolder + '/Annotations') and path.exists(dsFolder + '/JPEGImages'):
            self.isTarMode = False
            lstFiles = glob.glob(dsFolder + '/Annotations/*.xml')
            lstFiles = [x.replace('\\', '/') for x in lstFiles]
            cnt = len(lstFiles)
            for (i, sFile) in enumerate(lstFiles):
                if i % 100 == 0:
                    if callback is not None:
                        callback(i*100/cnt)
                fd = open(sFile)
                an = xmltodict.parse(fd.read())['annotation']
                fd.close()
                if isinstance(an['object'],list) == False:
                    an['object'] = [an['object']]
                lstBBoxes = self.ParseAnno(an, dctNewCfg)
                if len(lstBBoxes) >= minGTPerImg and len(lstBBoxes) <= maxGTPerImg:
                    fileKey = path.splitext(path.split(sFile)[-1])[0]
                    for xywh in lstBBoxes:
                        tag = xywh['tag']
                        if not tag in self.dctTags.keys():
                            self.dctTags[tag] = 1
                        else:
                            self.dctTags[tag] += 1                     
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
            try:
                tarCnt = len(self.lstTars)
                for (tarNdx, tar) in enumerate(self.lstTars):
                    lstTarInfos = tar.getmembers()
                    #查前缀
                    for info in lstTarInfos:
                        if info.name[-3:] == 'jpg':
                            self.tarRoots.append(path.split(info.name)[0])
                            break
                    infoCnt = len(lstTarInfos) // 2 + 1 # 一半是jpg, 一半是xml
                    for (i,info) in enumerate(lstTarInfos):
                        if info.name[-3:] == 'xml':
                            fd = tar.extractfile(info)
                            datBlock = fd.read()
                            an = xmltodict.parse(datBlock)['annotation']                         
                            #an = Xml2Dict(datBlock)
                            fd.close()
                            if isinstance(an['object'],list) == False:
                                an['object'] = [an['object']]
                            lstBBoxes = self.ParseAnno(an, dctNewCfg)
                            if len(lstBBoxes) >= minGTPerImg and len(lstBBoxes) <= maxGTPerImg:
                                fileKey = '{:02}_'.format(tarNdx) + info.name.split('/')[-1][:-4]
                                for xywh in lstBBoxes:
                                    tag = xywh['tag']
                                    if not tag in self.dctTags.keys():
                                        self.dctTags[tag] = 1
                                    else:
                                        self.dctTags[tag] += 1                                    
                                self.dctFiles[fileKey] = {
                                    'cnt0' : len(an['object']),
                                    'cnt' : len(lstBBoxes),
                                    'xywhs' : lstBBoxes
                                }
                            if i % 100 == 0:
                                pgs = 100 * (tarNdx / tarCnt + i / infoCnt / tarCnt)
                                if callback is not None:
                                    callback(pgs)
            except Exception as e:
                print(e)
                traceback.print_exc()
                raise e

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
            ndx = int(strFileKey[:2])
            strFileKey = strFileKey[3:]
            tarf = self.lstTars[ndx]
            strFile = self.tarRoots[ndx] + '/' + strFileKey + '.jpg'
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
    import patcher
    obj = patcher.Patcher(VOCUtils(dsFolder = 'Q:/gitrepos',setSel='any'))
    print(len(obj.dctFiles))
    obj.ShowClusterRandom()
    print('done!')

