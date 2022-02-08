from scipy.io import loadmat
from collections import defaultdict
import tarfile
import os.path as path
import io, os, struct, json, shutil
import zipfile
import glob
import xmltodict
import xml.dom.minidom
import traceback
import time
from PIL import Image
try:
    import abstract_utils
except:
    import plugins_dsread.abstract_utils as abstract_utils
def GetDSTypeName():
    return "Seq-Vbb"

def GetUtilClass():
    return SeqVbbUtils

def read_header(ifile):
    feed = ifile.read(4)
    norpix = ifile.read(24)
    version = struct.unpack('@i', ifile.read(4))
    length = struct.unpack('@i', ifile.read(4))
    assert(length != 1024)
    descr = ifile.read(512)
    params = [struct.unpack('@i', ifile.read(4))[0] for i in range(9)]
    fps = struct.unpack('@d', ifile.read(8))
    ifile.read(432)
    image_ext = {100: 'raw', 102: 'jpg', 201: 'jpg', 1: 'png', 2: 'png'}
    return {'w': params[0], 'h': params[1], 'bdepth': params[2],
            'ext': image_ext[params[5]], 'format': params[5],
            'size': params[4], 'true_size': params[8],
            'num_frames': params[6]}

def read_seq(inPath):
    ifile = open(inPath, 'rb')
    params = read_header(ifile)
    bytes = open(inPath, 'rb').read()
    lstImgMetas = []
    # imgs = []
    extra = 8
    s = 1024
    for i in range(params['num_frames']):
        tmp = struct.unpack_from('@I', bytes[s:s + 4])[0]
        lstImgMetas.append([inPath, s + 4, tmp - 4, params['w'], params['h'], params['ext']])
        img = bytes[s + 4:s + tmp]
        s += tmp + extra
        if i == 0:
            val = struct.unpack_from('@B', bytes[s:s + 1])[0]
            if val != 0:
                s -= 4
            else:
                extra += 8
                s += 8
        # imgs.append(I)

    return lstImgMetas, params

def read_img(fileKey):
    lstImgMeta = fileKey.split('?')
    
    fd = open(lstImgMeta[0], 'rb')
    fd.seek(lstImgMeta[1])
    bytes = fd.read(lstImgMeta[2])
    imgIO = io.BytesIO(bytes)
    img = Image.open(imgIO)
    img.show()
    return img

def read_vbb(inPath):
    assert inPath[-3:] == 'vbb'
    try:
        vbb = loadmat(inPath)
    except:
        return None
    nFrame = int(vbb['A'][0][0][0][0][0])
    objLists = vbb['A'][0][0][1][0]
    maxObj = int(vbb['A'][0][0][2][0][0])
    objInit = vbb['A'][0][0][3][0]
    objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]
    objStr = vbb['A'][0][0][5][0]
    objEnd = vbb['A'][0][0][6][0]
    objHide = vbb['A'][0][0][7][0]
    altered = int(vbb['A'][0][0][8][0][0])
    log = vbb['A'][0][0][9][0]
    logLen = int(vbb['A'][0][0][10][0][0])

    data = {}
    data['nFrame'] = nFrame
    data['maxObj'] = maxObj
    data['log'] = log.tolist()
    data['logLen'] = logLen
    data['altered'] = altered
    data['frames'] = defaultdict(list)

    for frame_id, obj in enumerate(objLists):
        if len(obj) > 0:
            try:
                for id, pos, occl, lock, posv in zip(obj['id'][0],
                                                    obj['pos'][0],
                                                    obj['occl'][0],
                                                    obj['lock'][0],
                                                    obj['posv'][0]):
                    keys = obj.dtype.names
                    id = int(id[0][0]) - 1  # MATLAB is 1-origin
                    p = pos[0].tolist()
                    pos = [p[0] - 1, p[1] - 1, p[2], p[3]]  # MATLAB is 1-origin
                    occl = int(occl[0][0])
                    lock = int(lock[0][0])
                    posv = posv[0].tolist()

                    datum = dict(zip(keys, [id, pos, occl, lock, posv]))
                    datum['lbl'] = str(objLbl[datum['id']])
                    # MATLAB is 1-origin
                    datum['str'] = int(objStr[datum['id']]) - 1
                    # MATLAB is 1-origin
                    datum['end'] = int(objEnd[datum['id']]) - 1
                    datum['hide'] = int(objHide[datum['id']])
                    datum['init'] = int(objInit[datum['id']])

                    data['frames'][frame_id].append(datum)
            except:
                continue

    return data

class SeqVbbUtils(abstract_utils.AbstractUtils):
    def __init__(self, dsFolder = '.', setSel='train', dctCfg={}, callback=None):
        super(SeqVbbUtils, self).__init__(dsFolder, setSel, dctCfg, callback)
        self.dsFolder = dsFolder
        self.dctFDs = dict()
        self.dctTags = dict()
        self.dctFiles = dict()
        self.setSel = setSel
        self.dctCfg = {}
        self.isTarMode = True
        self.tarRoots = []
        self.lstSeqFiles = []
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

        dctNewCfg = {
            'minHvsW' : minHvsW,
            'maxHvsW' : maxHvsW,
            'isSkipDirtyImg' : isSkipDirtyImg
        }
        default_callback(1, 'scan seq and vbb',callback)
        # scan for seq files that have vbb files in the same dir
        imgCnt = 0
        fileCnt = 0
        for root, dirs, files in os.walk(dsFolder, topdown=False):
            root = root.replace('\\', '/')
            for name in files:
                if name[-3:] == 'seq':
                    vbbPath = path.join(root, name[:-3] + 'vbb').replace('\\', '/').replace('images', 'annotations')
                    if path.exists(vbbPath):
                        default_callback(1, 'scaning %s' % (root + '/' + name), callback)
                        vbbAnno = read_vbb(vbbPath)
                        if vbbAnno is not None:
                            self.lstSeqFiles.append([root + '/' + name , vbbPath, vbbAnno])                        
                            imgCnt += vbbAnno['nFrame']
                            fileCnt += 1
                        else:
                            print('Can\'t parse %s!' % (vbbPath))

        self.lstImgMetas = []
        cnt = 0
        grpSize = max(100, imgCnt // 100)
        for [seq, _, vbbAnno] in self.lstSeqFiles:
            imgMetas, imgHdr = read_seq(seq)
            fd = open(seq, 'rb')
            self.dctFDs[seq] = fd
            imgWH = (imgHdr['w'], imgHdr['h'])
            for i in range(imgHdr['num_frames']):
                boxes = vbbAnno['frames'][i]
                lstBBoxes = []                
                for box in boxes:                    
                    xywh = box['pos']
                    if xywh[2] * 50 < imgWH[0] or xywh[3] * 50 < imgWH[1]:
                        continue
                    hVSw = xywh[3] / xywh[2]
                    dirty = 0
                    if hVSw < minHvsW or hVSw > maxHvsW:
                        dirty = 1                    
                    if isSkipDirtyImg and dirty != 0:
                        lstBBoxes = []
                        break
                    tag = box['lbl']
                    dctItem = {
                        'x1': min(int(xywh[0] + 0.5), imgWH[0]),
                        'y1': min(int(xywh[1] + 0.5), imgWH[1]),
                        'w': min(int(xywh[2] + 0.5), imgWH[0]),
                        'h': min(int(xywh[3] + 0.5), imgWH[1]),
                        'tag': tag,
                        'occlusion' : box['occl'],
                        'str' : box['str'],
                        'end' : box['end'],
                        'hide': box['hide'],
                        'init': box['init'],
                        'pose': box['posv'],
                        'isOverIllumination' : 0,
                        'dirty' : dirty
                    }
                    if not tag in self.dctTags.keys():
                        self.dctTags[tag] = 1
                    else:
                        self.dctTags[tag] += 1 
                    dctItem['difficult'] = dctItem['hide'] + dctItem['occlusion']
                    lstBBoxes.append(dctItem)
                if len(lstBBoxes) < 0:
                    continue
                seqNameEncoded = imgMetas[i][0].replace('/', '@')
                fileKey = '%s@%d@%d' % (seqNameEncoded[:-4], imgMetas[i][1], imgMetas[i][2])
                if fileKey[1] == ':':
                    fileKey = fileKey[0] + '~' + fileKey[2:]
                self.dctFiles[fileKey] = {
                    'cnt0' : len(lstBBoxes),
                    'cnt' : len(lstBBoxes),
                    'xywhs' : lstBBoxes
                }
                cnt += 1
                if cnt % grpSize == 1:
                    default_callback(100 * cnt / imgCnt, 'reading dataset', callback)
            self.lstImgMetas += imgMetas

        k2 = sorted(self.dctFiles.keys())
        dctRet = {}
        for (i,k) in enumerate(k2):
            dctRet[k] = self.dctFiles[k]
        self.dctFiles = dctRet

        bkpt = 0

    def CanDelTags(self):
        return True

    def IsSupportGTPerImg(self):
        return True

    def MapFile(self, fileKey:str):
        lstImgMeta = fileKey.split('@')
        seqFile = '/'.join(lstImgMeta[:-2]) + '.seq'
        if seqFile[1] == '~':
            seqFile = seqFile[0] + ':' + seqFile[2:]
        lstImgMeta[1] = int(lstImgMeta[-2])
        lstImgMeta[2] = int(lstImgMeta[-1])
        fd = self.dctFDs[seqFile]
        fd.seek(lstImgMeta[1])
        bytes = fd.read(lstImgMeta[2])
        #imgIO = io.BytesIO(bytes)
        #img = Image.open(imgIO)
        #img.show()
        return io.BytesIO(bytes)
    
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

    
    pathIn = 'Q:/datasets/pedestrain_Daimler'
    obj = SeqVbbUtils(pathIn)
    fileKey = list(obj.dctFiles.keys())[2305]
    img = Image.open(obj.MapFile(fileKey))
    img.show()
    exit(0)