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
    return ['COCO-大类', 'COCO-小类']


''' COCO数据集的要求
    图片文件：{setSel}{yyy}.zip
    标注文件：
        annotations_{包含setSel的字符串}{yyy}.zip
        其中，里面必须包含如下结构：
            annotations/instances_{setSel}{yyy}.json
'''
class InternalCOCOUtils():
    def __init__(self, dsFolder = '.', setSel='train', dctCfg = {}, callback=None, maxCnt=50000, isShuffle=True, isFineCls=True):
        self.pathBBox = '%s/annotation_%s.odgt' % (dsFolder, setSel)
        self.dctFiles = {}
        self.setSel = setSel
        self.dctFile2Zf = {}
        self.lstZfs = []
        self.lstFiles = []
        self.lstAnns = []
        
        self.dctTags = dict()

        minHvsW, maxHvsW = 1.0/6.0, 6.0
        try:
            minHvsW = dctCfg['minHvsW']
            maxHvsW = dctCfg['maxHvsW']
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
        for setSel in lstSetSel:
            globKey = '%s/*%s*.zip' % (dsFolder, setSel)
            globRet = set(glob.glob(globKey))
            setFiles = setFiles.union(globRet)
        lstFiles = list(setFiles)
        
        if len(lstFiles) == 0:
            raise ValueError('%s目录里未找到符合条件%s的COCO zip包！' % (dsFolder, setSel))
        # 查找训练文件
        dctFileMap = dict()
        
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
                                dctFileMap[datFilePath] = {'zipfile':sFile, 'annotationfile':fileName, 'zfobj':zf}
                if isZFUsed == False:
                    zf.close()

        print('分析COCO标注:')
        dctTagIDs = dict() 
        dctImgIDs = dict()
        
        for (zfNdx, x) in enumerate(dctFileMap.keys()) : 
            dctX = dctFileMap[x]
            self.lstZfs.append(dctX['zfobj'])
            default_callback(1, x + ', 分析json标注, 请耐心等待...', callback)

            jsonFile = dctX['zfobj'].open(dctX['annotationfile'])
            jsonData = json.load(jsonFile)
            jsonFile.close()
            lstAnn = jsonData['annotations']
            lstImg = jsonData['images']
            lstCls = jsonData['categories']
            # COCO标注中，list序号和对象id并不对应，需要做成dict以快速搜索
            for (i, dct) in enumerate(lstCls):
                dctTagIDs[dct['id']] = dct['supercategory'] if isFineCls == False else \
                    dct['supercategory'] + '/' + dct['name']

            for (i, dct) in enumerate(lstImg):
                dctImgIDs[dct['id']] = {'id':dct['id'],'file_name':dct['file_name'], 
                'w': dct['width'], 'h':dct['height']}
                if i % 500 == 0:
                    default_callback(i*20/len(lstImg),'扫描image id',callback)
            

            for (i, dct) in enumerate(lstAnn):
                if i % 500 == 0:
                    default_callback(20 + i*80/len(lstAnn),'扫描bbox',callback)
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
                aspect = h / w
                if aspect < minHvsW or aspect > maxHvsW:
                    continue               
                dctItem = {
                    'x1': x1,
                    'y1': y1,
                    'w': w,
                    'h': h,
                    'area': w*h,
                    'tag': dctTagIDs[dct['category_id']],
                    'blur': 0,
                    'isExaggerate': 0,
                    'isOverIllumination': 0,
                    'occlusion' : 0,
                    'pose': 0,
                    'difficult': 0,
                    'isInvalid' : 0                   
                }
                try:
                    dctItem['difficult'] = dct['iscrowd']
                except:
                    pass
                
                tag = dctItem['tag']
                try:
                    self.dctTags[tag] += 1
                except:
                    self.dctTags[tag] = 1
                
                if imgKey in self.dctFiles:
                    item = self.dctFiles[imgKey]
                    item['cnt'] += 1
                    item['cnt0'] += 1
                    item['xywhs'].append(dctItem)
                else:
                    self.dctFiles[imgKey] = {
                        'cnt': 1,
                        'cnt0': 1,
                        'xywhs': [dctItem]
                    }

           
        bkpt = 0
        
    def MapFile(self, strFile:str):
        zf = self.dctFile2Zf[strFile]
        fd = zf.open(strFile)
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
        mapped = 'Images/' + fileKey
        exts = ['.jpg', '.jpeg']
        for ext in exts:
            if mapped + ext in self.dctFiles.keys():
                return mapped + ext
        return ''

class COCOCoarseUtils(InternalCOCOUtils):
    def __init__(self, dsFolder = '.', setSel='train', dctCfg = {}, callback=None, maxCnt=50000, isShuffle=True):
        super(COCOCoarseUtils, self).__init__(dsFolder, setSel, dctCfg, callback, maxCnt, isShuffle, False)

class COCOFineUtils(InternalCOCOUtils):
    def __init__(self, dsFolder = '.', setSel='train', dctCfg = {}, callback=None, maxCnt=50000, isShuffle=True):
        super(COCOFineUtils, self).__init__(dsFolder, setSel, dctCfg, callback, maxCnt, isShuffle, True)


if __name__ == '__main__':
    coco = InternalCOCOUtils('q:/datasets/COCO_2017', 'val')
    print('Not meant to run as main! done!')

