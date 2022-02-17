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
    return [COCOCoarseUtils, COCOFineUtils]
def GetDSTypeName():
    return ['COCO-Coarse', 'COCO-Fine']


''' COCO数据集的要求
    图片文件：{setSel}{yyy}.zip
    标注文件：
        annotations_{包含setSel的字符串}{yyy}.zip
        其中，里面必须包含如下结构：
            annotations/instances_{setSel}{yyy}.json
'''
class InternalCOCOUtils(abstract_utils.AbstractUtils):
    def __init__(self, dsFolder = '.', setSel='train', dctCfg = {}, callback=None, maxCnt=50000, isShuffle=True, isFineCls=True):
        super(InternalCOCOUtils, self).__init__(dsFolder, setSel, dctCfg, callback, maxCnt, isShuffle)
        self.dctFiles = {}
        self.setSel = setSel
        self.dctFile2Zf = {}
        self.lstZfs = []
        self.lstFiles = []
        self.lstAnns = []
        self.dctTagToWholeTag = dict()
        self.dctTags = dict()
        dctFiles = dict()
        minHvsW, maxHvsW = 0.1, 10.0
        minGTPerImg, maxGTPerImg = 1, 50
        isSkipDirtyImg = False
        try:
            minHvsW = dctCfg['minHvsW']
            maxHvsW = dctCfg['maxHvsW']
            # coco格式是每个标注作为一个单元，在遍历全部标注前不保证能算出一个图有多少个GT
            # 标注出现的顺序并不是某种图片的排列顺序。
            minGTPerImg = dctCfg['minGTPerImg']
            maxGTPerImg = dctCfg['maxGTPerImg']
            isSkipDirtyImg = dctCfg['isSkipDirtyImg']
        except:
            pass

        def default_callback(pgs, msg, in_callback):
            print(pgs, msg)
            if in_callback is not None:
                in_callback(pgs, msg)

        lstLines = []
        if setSel == 'any':
            lstSetSel = ['train', 'val', 'test']
        else:
            lstSetSel = [setSel]
        setFiles = set()
        print('扫描zip数据集的索引')
        for curSet in lstSetSel:
            globKey = '%s/*%s*.zip' % (dsFolder, curSet)
            globRet = set(glob.glob(globKey))
            setFiles = setFiles.union(globRet)
        lstFiles = list(setFiles)
        
        if len(lstFiles) == 0:
            raise ValueError('%s目录里未找到符合条件%s的COCO zip包！' % (dsFolder, setSel))
        # 查找训练文件
        dctFileMap = dict()
        datFileCnt = 0
        for sFile in lstFiles:
            if sFile.lower().find('annotations_') >= 0:
                zf = zipfile.ZipFile(sFile)
                isZFUsed = False
                if zf.filelist[0].filename.lower().find('annotations/') == 0:
                    lstInst = list()
                    labelPrefix = 'annotations/instances_'
                    prefixLen = len(labelPrefix)
                    for sItem in zf.filelist:
                        fileName = sItem.filename
                        if fileName[-5:].lower() == '.json' and fileName[:prefixLen].lower() == labelPrefix:                            
                            datFileMainName = fileName[prefixLen:-5]
                            if setSel == 'any' or datFileMainName.lower().find(setSel) >= 0:
                                isZFUsed = True
                                datFilePath = '%s/%s.zip' % (dsFolder, datFileMainName)
                                dctFileMap[datFilePath] = {'annZipContainer':sFile, 'annFile':fileName, 'zfAnn':zf}
                                datZF = zipfile.ZipFile(datFilePath)
                                dctFileMap[datFilePath]['imgRoot'] = datZF.filelist[0].filename
                                dctFileMap[datFilePath]['zfDat'] = datZF
                                datFileCnt += 1
                if isZFUsed == False:
                    zf.close()

        print('分析COCO标注:')
        dctTagIDs = dict()
        dctWholeTagIDs = dict()
        dctImgIDs = dict()
        
        for (zfNdx, x) in enumerate(dctFileMap.keys()) : 
            dctX = dctFileMap[x]
            self.lstZfs.append(dctX)
            default_callback(100*zfNdx / datFileCnt + 1, x + ', 分析json标注, 请耐心等待...', callback)

            jsonFile = dctX['zfAnn'].open(dctX['annFile'])
            jsonData = json.load(jsonFile)
            jsonFile.close()
            lstAnn = jsonData['annotations']
            lstImg = jsonData['images']
            lstCls = jsonData['categories']
            # COCO标注中，list序号和对象id并不对应，需要做成dict以快速搜索
            for (i, dct) in enumerate(lstCls):
                # coco分大类和小类
                dctWholeTagIDs[dct['id']] = dct['supercategory'] if isFineCls == False else \
                    dct['name'] + '`' + dct['supercategory']
                dctTagIDs[dct['id']] = dct['supercategory'] if isFineCls == False else \
                    dct['name']
                key1 = dctTagIDs[dct['id']]
                self.dctTagToWholeTag[key1] = dctWholeTagIDs[dct['id']]

            for (i, dct) in enumerate(lstImg):
                dctImgIDs[dct['id']] = {'id':dct['id'],'file_name':dct['file_name'], 
                'w': dct['width'], 'h':dct['height']}
                if i % 1000 == 0:
                    default_callback(i*10/len(lstImg)/datFileCnt + 100 * zfNdx / datFileCnt,
                    '扫描image id',callback)
            

            for (i, dct) in enumerate(lstAnn):
                if i % 1000 == 0:
                    default_callback((10 + i*90/len(lstAnn))/datFileCnt + 100 * zfNdx / datFileCnt
                    ,'扫描bbox',callback)
                try:
                    imgInfo = dctImgIDs[dct['image_id']]
                except:
                    continue
                imgKey = '%02d_' % (zfNdx) + imgInfo['file_name']
                bbox_in = dct['bbox']
                x1 =  int(bbox_in[0] + 0.5)
                y1 =  int(bbox_in[1] + 0.5)
                w = int(bbox_in[2] + 0.5)                
                h = int(bbox_in[3] + 0.5)
                if w < 8 or h < 8:
                    continue
                aspect = h / w
                badAspectFlag = 1 if aspect < minHvsW or aspect > maxHvsW else 0
                dctItem = {
                    'x1': x1,
                    'y1': y1,
                    'w': w,
                    'h': h,
                    'area': w*h,
                    'tag': dctWholeTagIDs[dct['category_id']],
                    'blur': 0,
                    'isExaggerate': 0,
                    'isOverIllumination': 0,
                    'occlusion' : 0,
                    'pose': 0,
                    'difficult': 0,
                    'isInvalid' : 0,
                    'dirty': badAspectFlag
                }
                try:
                    dctItem['difficult'] = dct['iscrowd']
                except:
                    pass
                
                tag = dctItem['tag']
                
                if imgKey in dctFiles:                    
                        item = dctFiles[imgKey]
                        if item['cnt'] < maxGTPerImg:
                            item['cnt'] += 1
                            item['cnt0'] += 1
                            item['xywhs'].append(dctItem)
                else:
                    dctFiles[imgKey] = {
                        'cnt': 1,
                        'cnt0': 1,
                        'xywhs': [dctItem]
                    }

        dctFilt = dict()
        t0 = time.time()
        for (i, itemKey) in enumerate(dctFiles.keys()):
            isToSkipImg = False
            if i % 1000 == 0:
                default_callback(i*100/len(dctFiles), '根据物体数量约束筛选', callback)            
            item = dctFiles[itemKey]
            cnt = item['cnt']
            # 检查是否含有脏标注
            if isSkipDirtyImg == True:
                for xywh in item['xywhs']:
                    if xywh['dirty'] > 0:
                        isToSkipImg = True
                        break

            if isToSkipImg == True:
                continue
            if cnt >= minGTPerImg and cnt <= maxGTPerImg:
                dctFilt[itemKey] = item
                for xywh in item['xywhs']:
                    tag = xywh['tag']
                    # 两者性能似乎没有显著差异
                    if False:
                        if not tag in self.dctTags.keys():
                            self.dctTags[tag] = 1
                        else:
                            self.dctTags[tag] += 1
                    else:
                        try:
                            self.dctTags[tag] += 1
                        except:
                            self.dctTags[tag] = 1                         

        dt = time.time() - t0
        self.dctFiles = dctFilt

        k2 = sorted(self.dctFiles.keys())
        dctRet = {}
        for (i,k) in enumerate(k2):
            dctRet[k] = self.dctFiles[k]
        self.dctFiles = dctRet

        bkpt = 0
    def DelTag(self, sTag:str):
        for dctItem in self.dctFiles:
            bkpt = 0
        
    def MapFile(self, strFile:str):
        dct = self.lstZfs[int(strFile[:2])]
        zf = dct['zfDat']
        imgKey = dct['imgRoot'] + strFile[3:]
        fd = zf.open(imgKey)
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
        exts = ['.jpg', '.jpeg']
        for ext in exts:
            if fileKey + ext in self.dctFiles.keys():
                return fileKey + ext
        return ''

class COCOCoarseUtils(InternalCOCOUtils):
    def __init__(self, dsFolder = '.', setSel='train', dctCfg = {}, callback=None, maxCnt=50000, isShuffle=True):
        super(COCOCoarseUtils, self).__init__(dsFolder, setSel, dctCfg, callback, maxCnt, isShuffle, False)

class COCOFineUtils(InternalCOCOUtils):
    def __init__(self, dsFolder = '.', setSel='train', dctCfg = {}, callback=None, maxCnt=50000, isShuffle=True):
        super(COCOFineUtils, self).__init__(dsFolder, setSel, dctCfg, callback, maxCnt, isShuffle, True)
    def TranslateTag(self, tagIn):
        ret = tagIn.split('`')[0]
        return ret

if __name__ == '__main__':
    coco = COCOFineUtils('q:/datasets/COCO_2017', 'val')
    print('Not meant to run as main! done!')

