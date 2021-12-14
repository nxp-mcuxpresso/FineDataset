import sys
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtCore import Qt
import widertools
import wf_utils
import json
import os.path as path
import os
import numpy as np

def DelTree(treeName, isDelDir=False, isDelRoot=False):
    'delete a tree, recursively, it can be non empty!'
    if not path.exists(treeName):
        if not isDelDir and not isDelRoot:
            os.mkdir(treeName)
        return -1

    for root, dirs, files in os.walk(treeName, topdown=False):
        for name in files:
            os.remove(path.join(root, name))
            # print "deleting file %s" % name
        if isDelDir == True:
            for name in dirs:
                os.rmdir(path.join(root, name))
                # print "deleting folder %s" % name
    if isDelDir == True and isDelRoot == True:
        os.rmdir(treeName)


class MainAppLogic():
    def __init__(self, ui:widertools.Ui_MainWindow):
        self.ui = ui
        self.prevImgItem = None
        ui.cmbMain.addItems(['train','val','test'])
        ui.cmbMaxFacesPerCluster.addItems(['6','5', '4', '3', '2'])
        ui.cmbCloseRatio.addItems(['0.5', '0.4', '0.32', '0.25', '0.2', '0.16', '0.125', '0.1', '0.08'])
        ui.cmbCloseRatio.setCurrentIndex(2)
        #ui.cmbMain.currentIndexChanged.connect(lambda: LoadDataset(ui))
        ui.cmbMain.textActivated.connect(lambda: self.LoadDataset())
        #ui.cmbMain.highlighted.connect(lambda: LoadDataset(ui)) 
        self.LoadDataset()
        ui.btnRandom.clicked.connect(self.OnClicked_Random)
        ui.btnSplitSingle.clicked.connect(self.OnClicked_SplitSingle)
        ui.btnGenSingleFaceDataSet.clicked.connect(self.OnClicked_GenSingleFaceDataset)
        ui.btnValidateSingleFaceDataSet.clicked.connect(self.OnClicked_ValidateSingleFaceDataset)
        ui.btnValidateMultiFaceDataSet.clicked.connect(self.OnClicked_ValidateMultiFaceDataset)        
        ui.btnGenMultiFaceDataSet.clicked.connect(self.OnClicked_GenMultiFaceDataset)
        #ui.btnValidateMultiFaceDataSet.clicked.connect(self.OnClicked_ValidateSingleFaceDataset)

        # ui.btnGenMultiFaceDataSet.setEnabled(False)
        # ui.btnValidateMultiFaceDataSet.setEnabled(False)
        # ui.cmbMaxFacesPerCluster.setEnabled(False)
        ui.btnSaveOriBBoxes.clicked.connect(self.OnClicked_SaveOriBBoxes)
        self.patchNdx = 0
        self.lstPatches = []            
    
    def ShowImage(self, table, ndx, strKey):
        item = self.dataObj.dctFiles[strKey]
        self.prevImgItem = [table, ndx, strKey, item]
        c = table.shape
        qImg = QtGui.QImage(bytearray(table), c[1], c[0], c[1]*3, QtGui.QImage.Format_BGR888)
        pix = QPixmap(QPixmap.fromImage(qImg))
        
        #pix2 = pix.scaled(32,32, Qt.KeepAspectRatio)#, Qt.SmoothTransformation)
        pix3 = pix.scaled(1024,1024, Qt.KeepAspectRatio)#, Qt.SmoothTransformation)
        
        # ui.imgWnd.setPixmap(pix2)
        ui.lblImg.setPixmap(pix3)

    def OnClicked_SaveOriBBoxes(self):
        sOutFile = 'labels_%s.json' % (self.ui.cmbMain.currentText())
        with open(sOutFile, 'w') as fd:
            json.dump(self.dataObj.dctFiles, fd, indent=4)
        QMessageBox.information(None,'box', '已保存到%s' % sOutFile)

    def OnClicked_Random(self):
        [table,ndx, strKey] = self.dataObj.ShowRandom(False)
        self.ShowImage(table, ndx, strKey)

    def OnClicked_SplitSingle(self):
        if self.prevImgItem is None:
            return
        outX = int(self.ui.txtOutX.text())
        outY = int(self.ui.txtOutY.text())
        self.patchNdx, lstPatches = self.dataObj.CutPatches(self.patchNdx, ndx=self.prevImgItem[1], outSize=[outX, outY])
        self.lstPatches += lstPatches
        with open('bboxes.json', 'w', encoding='utf-8') as fd:
            json.dump(self.lstPatches, fd, indent=4)

    def OnClicked_GenMultiFaceDataset(self):
        cnt = len(self.dataObj.dctFiles.keys())
        ndc = np.arange(cnt)
        np.random.shuffle(ndc)
        strOutFolder = './out_%s_multi' % (self.ui.cmbMain.currentText())
        DelTree(strOutFolder, True)
        outX = int(self.ui.txtOutX.text())
        outY = int(self.ui.txtOutY.text())        
        self.patchNdx = 0
        self.lstPatches = []
        dsSize = int(self.ui.txtDatasetSize.text())
        if not path.exists(strOutFolder):
            os.makedirs(strOutFolder)
        maxObjPerCluster = int(self.ui.cmbMaxFacesPerCluster.currentText())
        closeRatio = float(self.ui.cmbCloseRatio.currentText())
        for ndx in ndc:
            #[table,ndx, strKey] = self.dataObj.ShowImage(ndx, False)
            #self.ShowImage(table, ndx, strKey)            
            self.patchNdx, lstPatches = self.dataObj.CutClusterPatches(strOutFolder, \
                self.patchNdx, ndx=ndx, closeRatio=closeRatio, maxObjPerCluster=maxObjPerCluster, outSize=[outX, outY])
            self.lstPatches += lstPatches
            if self.patchNdx >= dsSize:
                break
            print('%d/%d completed' % (self.patchNdx, dsSize))
        with open('%s/bboxes.json' % strOutFolder, 'w', encoding='utf-8') as fd:
            json.dump(self.lstPatches, fd, indent=4)
        QMessageBox.information(None,'box', '制作完成')

    def OnClicked_GenSingleFaceDataset(self):
        cnt = len(self.dataObj.dctFiles.keys())
        ndc = np.arange(cnt)
        np.random.shuffle(ndc)
        strOutFolder = './out_%s_single' % (self.ui.cmbMain.currentText())
        DelTree(strOutFolder, True)
        outX = int(self.ui.txtOutX.text())
        outY = int(self.ui.txtOutY.text())        
        self.patchNdx = 0
        self.lstPatches = []
        dsSize = int(self.ui.txtDatasetSize.text())
        if not path.exists(strOutFolder):
            os.makedirs(strOutFolder)
        for ndx in ndc:
            #[table,ndx, strKey] = self.dataObj.ShowImage(ndx, False)
            #self.ShowImage(table, ndx, strKey)            
            self.patchNdx, lstPatches = self.dataObj.CutPatches(strOutFolder, self.patchNdx, ndx=ndx, outSize=[outX, outY])
            self.lstPatches += lstPatches
            if self.patchNdx >= dsSize:
                break
            print('%d/%d completed' % (self.patchNdx, dsSize))
        with open('%s/bboxes.json' % (strOutFolder), 'w', encoding='utf-8') as fd:
            json.dump(self.lstPatches, fd, indent=4)
        QMessageBox.information(None,'box', '制作完成')


    def OnClicked_ValidateFaceDataset(self, strSel='single'):
        strOutFolder = './out_%s_%s' % (self.ui.cmbMain.currentText(), strSel)
        table = self.dataObj.ShowRandomValidate(strOutFolder)
        c = table.shape
        qImg = QtGui.QImage(bytearray(table), c[1], c[0], c[1]*3, QtGui.QImage.Format_BGR888)
        pix = QPixmap(QPixmap.fromImage(qImg))
        
        #pix2 = pix.scaled(32,32, Qt.KeepAspectRatio)#, Qt.SmoothTransformation)
        pix3 = pix.scaled(128,128, Qt.KeepAspectRatio)#, Qt.SmoothTransformation)
        
        # ui.imgWnd.setPixmap(pix2)
        self.ui.lblImg.setPixmap(pix3)             

    def OnClicked_ValidateSingleFaceDataset(self, strSel='single'):
        self.OnClicked_ValidateFaceDataset('single')

    def OnClicked_ValidateMultiFaceDataset(self, strSel='single'):
        self.OnClicked_ValidateFaceDataset('multi')        
              
    def LoadDataset(self):
        # QMessageBox.information(None,'box',ui.cmbMain.currentText())
        # ui.cmbMain.currentData
        self.dataObj = wf_utils.WFUtils(self.ui.cmbMain.currentText())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = widertools.Ui_MainWindow()
    ui.setupUi(MainWindow)

    MainWindow.show()
    mainLogic = MainAppLogic(ui)
    app.exec()
    # sys.exit(app.exec_())
