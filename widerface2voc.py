import json

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
class WF2VOC():
    def __init__(self, setSel='train', cntSel='single', strRootPath = './wf_voc'):
        self.setSel = setSel
        self.cntSel = cntSel
        self.strOutPath = './outs/wf_voc_%s_%s' % (setSel, cntSel)
        self.strInPath = './outs/out_%s_%s' % (setSel, cntSel)
    def MakeVOC(self, maxCnt=1E7, callback=None):
        if path.exists(self.strInPath):
            with open(self.strInPath + '/bboxes.json') as fd:
                self.lstBBoxes = json.load(fd)
        else:
            return -1
        if path.exists(self.strOutPath):
            rmtree(self.strOutPath)
        os.makedirs(path.join(self.strOutPath, "JPEGImages"))
        os.makedirs(path.join(self.strOutPath, "Annotations"))
        cnt = 0
        total = len(self.lstBBoxes)
        for item in self.lstBBoxes:
            sFileName = path.split(item['filename'])[1]            
            img = cv2.imread(item['filename'])
            sOutJpgPath = '%s/JPEGImages/%s.jpg' % (self.strOutPath, path.splitext(sFileName)[0])
            cv2.imwrite(sOutJpgPath, img, [int(cv2.IMWRITE_JPEG_QUALITY),70])
            strOutFrame = strPatten % (self.setSel, self.cntSel, sFileName, img.shape[0], img.shape[1])
            for xyxy in item['xyxys']:
                strVocBox = strObj % (xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                strOutFrame += strVocBox
            strOutFrame += '</annotation>\n'
            
            sOutXmlPath = '%s/Annotations/%s.xml' % (self.strOutPath, path.splitext(sFileName)[0])
            with open(sOutXmlPath, 'w') as fd:
                fd.write(strOutFrame)
            cnt += 1
            if callback is not None:
                callback(100 * cnt / total)
            if cnt >= maxCnt:
                break
        return cnt
if __name__ == '__main__':
    tester = WF2VOC('train','multi')
    tester.MakeVOC(30)