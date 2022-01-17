import json
import shutil

strPatten = '''
<annotation>
	<folder>wf_voc_%s_%s</folder>
	<filename>%s.jpg</filename>
	<source>
		<database>The VOC2007 Database</database>
		<annotation>PASCAL VOC2007</annotation>
		<image>flickr</image>
		<flickrid>325443404</flickrid>
	</source>
	<owner>
		<flickrid>autox4u</flickrid>
		<name>Perry Aidelbaum</name>
	</owner>
	<size>
		<width>%d</width>
		<height>%d</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
'''

strObj = '''
	<object>
		<name>face</name>
		<pose>Right</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>%d</xmin>
			<ymin>%d</ymin>
			<xmax>%d</xmax>
			<ymax>%d</ymax>
		</bndbox>
	</object>
'''
import os.path as path
import os
# from wf_utils import DelTree
from shutil import rmtree
import cv2
import tarfile
class WF2VOC():
    def __init__(self, setSel='train', cntSel='single', strRootPath = './outs'):
        self.setSel = setSel
        self.cntSel = cntSel
        self.strOutPath = './outs/wf_voc_%s_%s' % (setSel, cntSel)
        self.strInPath = '%s/out_%s_%s' % (strRootPath, setSel, cntSel)
        self.strRootPath = strRootPath

    def MakeVOC(self, maxCnt=1E7, callback=None, isTarOnly=True):
        if path.exists(self.strInPath):
            with open(self.strInPath + '/bboxes.json') as fd:
                self.lstBBoxes = json.load(fd)
        else:
            return -1
        if path.exists(self.strOutPath):
            rmtree(self.strOutPath)
        os.makedirs(path.join(self.strOutPath, "JPEGImages"))
        os.makedirs(path.join(self.strOutPath, "Annotations"))
        cur_path = os.getcwd()
        
        cnt = 0
        total = len(self.lstBBoxes)

        os.chdir('./outs')
        sTarFolder = 'wf_voc_%s_%s' % (self.setSel, self.cntSel)
        tarf = tarfile.TarFile(sTarFolder + '.tar', 'w')
        os.chdir(cur_path)

        for item in self.lstBBoxes:
            sFileName = path.split(item['filename'])[1]
            sMainName = path.splitext(sFileName)[0]
            img = cv2.imread(self.strRootPath + '/' + item['filename'])
            sOutJpgPath = '%s/JPEGImages/%s.jpg' % (self.strOutPath, sMainName)
            
            cv2.imwrite(sOutJpgPath, img, [int(cv2.IMWRITE_JPEG_QUALITY),70])
            
            os.chdir('./outs')
            tarf.add('%s/JPEGImages/%s.jpg' % (sTarFolder, sMainName))
            os.chdir(cur_path)
            if isTarOnly:
                os.remove(sOutJpgPath)
            strOutFrame = strPatten % (self.setSel, self.cntSel, sFileName, img.shape[0], img.shape[1])
            for xyxy in item['xyxys']:
                strVocBox = strObj % (xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                strOutFrame += strVocBox
            strOutFrame += '</annotation>\n'
            
            sOutXmlPath = '%s/Annotations/%s.xml' % (self.strOutPath, sMainName)
            with open(sOutXmlPath, 'w') as fd:
                fd.write(strOutFrame)
            os.chdir('./outs')
            tarf.add('%s/Annotations/%s.xml' % (sTarFolder, sMainName))
            os.chdir(cur_path)
            if isTarOnly:
                os.remove(sOutXmlPath)
            cnt += 1
            if callback is not None:
                callback(100 * cnt / total)
            if cnt >= maxCnt:
                break
       
        tarf.close()
        if isTarOnly:
            shutil.rmtree(self.strOutPath)
        return cnt

    def ScanAndDelInvalidBBoxEntries(self, maxCnt=1E7, callback=None):
        if path.exists(self.strInPath):
            with open(self.strInPath + '/bboxes.json') as fd:
                self.lstBBoxes = json.load(fd)
        else:
            return -1

        cnt = 0
        total = len(self.lstBBoxes)

        lstNewBBoxes = list()
        for item in self.lstBBoxes:
            imgFile = self.strRootPath + '/' + item['filename']
            if path.exists(imgFile):
                lstNewBBoxes.append(item)
        if len(lstNewBBoxes) < len(self.lstBBoxes):
            with open(self.strInPath + '/bboxes.json', 'w') as fd:
                self.lstBBoxes = lstNewBBoxes
                json.dump(lstNewBBoxes, fd)
        return cnt

if __name__ == '__main__':
    tester = WF2VOC('train','multi')
    tester.MakeVOC(50)