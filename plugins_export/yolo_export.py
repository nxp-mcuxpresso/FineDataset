try:
    import abstract_export
except:
    import plugins_export.abstract_export as abstract_export

import json
import shutil

import os.path as path
import os
# from wf_utils import DelTree
from shutil import rmtree
import cv2
import tarfile

def GetDSTypeName():
    return "(ongoing) YOLO"

def GetUtilClass():
    return YOLOExport

class YOLOExport(abstract_export.AbstractExport):
    def __init__(self, setSel='train', subsetSel='single', strRootPath = './outs'):
        super(YOLOExport, self).__init__(setSel, subsetSel, strRootPath)

    def _doMakeYOLO(self, strInPath, maxCnt=1E7, callback=None, isTarOnly=True):
        
        if path.exists(strInPath):
            with open(strInPath + '/bboxes.json') as fd:
                self.lstBBoxes = json.load(fd)
        else:
            return -1

        cur_path = os.getcwd()
        t1 = os.stat(strInPath).st_mtime
        
        os.chdir('./outs')
        sTarFolder = 'yolo_%s' % (strInPath.split('/')[-1])
        if path.exists(sTarFolder + '.tar'):
            t0 = os.stat(sTarFolder + '.tar').st_mtime
        else:
            t0 = 0
    
        if t1 < t0:
            # tar文件比目录时间要新，无需制作
            os.chdir(cur_path)         
            return 0

        tarf = tarfile.TarFile(sTarFolder + '.tar', 'w')

        os.chdir(cur_path)
        
        outPath = './outs/yolo_%s' % (strInPath.split('/')[-1])
        if path.exists(outPath):
            rmtree(outPath)
        os.makedirs(outPath + '/data')        
        cnt = 0
        total = len(self.lstBBoxes)
        dctTag2Num = dict()
        lstOutJpgPath = []
        for item in self.lstBBoxes:
            sFileName = path.split(item['filename'])[1]
            sMainName = path.splitext(sFileName)[0]
            img = cv2.imread(self.strRootPath + '/' + item['filename'])
            sOutJpgPath = '%s/data/%s.jpg' % (outPath, sMainName)

            cv2.imwrite(sOutJpgPath, img, [int(cv2.IMWRITE_JPEG_QUALITY),70])
            
            os.chdir('./outs')
            tarf.add('%s/data/%s.jpg' % (sTarFolder, sMainName))
            os.chdir(cur_path)
            if isTarOnly:
                os.remove(sOutJpgPath)
            tagCnt = 0
            lstYoloGTs = []
            for xyxy in item['xyxys']:
                tag = xyxy[4]
                try:
                    tagNdx = dctTag2Num[tag]
                except:
                    dctTag2Num[tag] = tagCnt
                    tagNdx = tagCnt
                    tagCnt += 1
                gtW = xyxy[2] - xyxy[0]
                gtH = xyxy[3] - xyxy[1]
                
                imgW = img.shape[0]
                imgH = img.shape[1]
                ccwh = [(xyxy[0] + gtW / 2) / imgW, (xyxy[1] + gtH / 2) / imgH, gtW / imgW, gtH / imgH]
                strYoloGT = str(tagNdx) + ' ' + ' '.join(['%.4f' % (x) for x in ccwh])
                lstYoloGTs.append(strYoloGT)
            

            
            sOutTxtPath = '%s/data/%s.txt' % (outPath, sMainName)
            with open(sOutTxtPath, 'w') as fd:
                fd.write('\n'.join(lstYoloGTs))
            os.chdir('./outs')
            tarf.add('%s/data/%s.txt' % (sTarFolder, sMainName))
            os.chdir(cur_path)
            if isTarOnly:
                os.remove(sOutTxtPath)
            cnt += 1
            if callback is not None:
                callback(100 * cnt / total, 'processing %s' % strInPath)
            lstOutJpgPath.append('./data/' + sOutJpgPath.split('/')[-1])
            if cnt >= maxCnt:
                break

        with open(path.join(outPath, 'list.txt'), 'w') as lstf:
            lstf.write('\n'.join(lstOutJpgPath))

        os.chdir('./outs')
        tarf.add(sTarFolder + '/list.txt')
        os.chdir(cur_path)
        tarf.close()
        if isTarOnly:
            shutil.rmtree(outPath)
        return cnt

    def Export(self, maxCnt=1E7, callback=None, isTarOnly=True):
        for strPath in self.lstInPaths:
            self._doMakeYOLO(strPath, maxCnt, callback, isTarOnly)

if __name__ == '__main__':
    curDir = os.getcwd()
    if path.split(curDir)[-1].startswith('plugin'):
        os.chdir('../')
    print(os.getcwd())    
    tester = YOLOExport('train','multi')
    tester.Export(50)