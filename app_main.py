import sys
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QStatusBar
from PyQt5.QtCore import Qt
import widertools
import wf_utils
import json
import os.path as path
import os
import numpy as np
import widerface2voc as w2v
from wf_utils import DelTree

class MainAppLogic():
    def __init__(self, ui:widertools.Ui_MainWindow, mainWindow):
        self.ui = ui
        self.mainWindow = mainWindow
        self.prevImgItem = None
        ui.cmbMain.addItems(['train','val'])
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
        ui.btnToVoc.clicked.connect(self.OnClicked_ToVOC)
        ui.pgsBar.setVisible(False)
        ui.tmrToHidePgsBar = QtCore.QTimer(self.mainWindow)
        ui.tmrToHidePgsBar.setInterval(300)
        ui.tmrToHidePgsBar.setSingleShot(True)
        ui.tmrToHidePgsBar.timeout.connect(self.OnTimeout_tmrToHidePgsBar)
        ui.statusBar = QStatusBar()
        mainWindow.setStatusBar(ui.statusBar)
        ui.statusBar.addPermanentWidget(ui.pgsBar)
        #ui.btnValidateMultiFaceDataSet.clicked.connect(self.OnClicked_ValidateSingleFaceDataset)

        # ui.btnGenMultiFaceDataSet.setEnabled(False)
        # ui.btnValidateMultiFaceDataSet.setEnabled(False)
        # ui.cmbMaxFacesPerCluster.setEnabled(False)
        ui.btnSaveOriBBoxes.clicked.connect(self.OnClicked_SaveOriBBoxes)
        self.patchNdx = 0
        self.lstPatches = []
    def OnTimeout_tmrToHidePgsBar(self):
        self.ui.pgsBar.setVisible(False)

    def ShowImage(self, table, ndx, strKey):
        item = self.dataObj.dctFiles[strKey]
        self.prevImgItem = [table, ndx, strKey, item]
        c = table.shape
        qImg = QtGui.QImage(bytearray(table), c[1], c[0], c[1]*3, QtGui.QImage.Format_BGR888)
        pix = QPixmap(QPixmap.fromImage(qImg))
        
        #pix2 = pix.scaled(32,32, Qt.KeepAspectRatio)#, Qt.SmoothTransformation)
        pix3 = pix.scaled(640,640, Qt.KeepAspectRatio)#, Qt.SmoothTransformation)
        
        # ui.imgWnd.setPixmap(pix2)
        ui.lblImg.setPixmap(pix3)

    def OnClicked_ToVOC(self):
        def callback(pgs):
            self.ui.pgsBar.setValue(pgs)
            QApplication.processEvents()
        
        self.ui.pgsBar.setVisible(True)
        for setSel in ['train', 'val', 'test']:
            for cntSel in ['single', 'multi']:
                self.ui.pgsBar.setValue(1)
                self.ui.statusBar.showMessage('正在转换%s %s' % (setSel, cntSel), 60000)
                QApplication.processEvents()
                voc = w2v.WF2VOC(setSel, cntSel)
                voc.MakeVOC(callback=callback)
        self.ui.statusBar.showMessage('转换完成', 5000)
        self.ui.tmrToHidePgsBar.start()

    def OnClicked_SaveOriBBoxes(self):
        self.ui.statusBar.showMessage('正在保存', 60000)
        QApplication.processEvents()
        sOutFile = 'labels_%s.json' % (self.ui.cmbMain.currentText())
        with open(sOutFile, 'w') as fd:
            json.dump(self.dataObj.dctFiles, fd, indent=4)
        # QMessageBox.information(None,'box', '已保存到%s' % sOutFile)
        self.ui.statusBar.showMessage('已保存到%s' % sOutFile, 5000)

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
        self.ui.pgsBar.setValue(1)
        self.ui.pgsBar.setVisible(True)        
        dsSize = int(self.ui.txtDatasetSize.text())
        if not path.exists(strOutFolder):
            os.makedirs(strOutFolder)
        maxObjPerCluster = int(self.ui.cmbMaxFacesPerCluster.currentText())
        closeRatio = float(self.ui.cmbCloseRatio.currentText())
        self.ui.pgsBar.setValue(1)
        self.ui.pgsBar.setVisible(True)        
        self.ui.tmrToHidePgsBar.stop()
        for ndx in ndc:
            #[table,ndx, strKey] = self.dataObj.ShowImage(ndx, False)
            #self.ShowImage(table, ndx, strKey)
            self.ui.statusBar.showMessage('制作中, 图片%d' % ndx, 3600000)
            self.patchNdx, lstPatches = self.dataObj.CutClusterPatches(strOutFolder, \
                self.patchNdx, ndx=ndx, closeRatio=closeRatio, maxObjPerCluster=maxObjPerCluster, outSize=[outX, outY])
            self.lstPatches += lstPatches
            if self.patchNdx >= dsSize:
                break
            pgs = 100 * self.patchNdx / dsSize
            self.ui.pgsBar.setValue(pgs)
            QApplication.processEvents()
            print('%d/%d completed' % (self.patchNdx, dsSize))
        with open('%s/bboxes.json' % strOutFolder, 'w', encoding='utf-8') as fd:
            json.dump(self.lstPatches, fd, indent=4)
        self.ui.pgsBar.setValue(100)
        self.ui.statusBar.showMessage('制作了%d/%d张图片于%s' % (self.patchNdx, dsSize, strOutFolder), 5000)
        self.ui.tmrToHidePgsBar.start()
        #self.ui.pgsBar.setVisible(False)

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
        self.ui.pgsBar.setValue(1)
        self.ui.pgsBar.setVisible(True)
        self.ui.tmrToHidePgsBar.stop()
        for ndx in ndc:
            #[table,ndx, strKey] = self.dataObj.ShowImage(ndx, False)
            #self.ShowImage(table, ndx, strKey)            
            self.ui.statusBar.showMessage('制作中, 图片%d' % ndx, 3600000)
            self.patchNdx, lstPatches = self.dataObj.CutPatches(strOutFolder, self.patchNdx, ndx=ndx, outSize=[outX, outY])
            self.lstPatches += lstPatches
            if self.patchNdx >= dsSize:
                break
            pgs = 100 * self.patchNdx / dsSize
            self.ui.pgsBar.setValue(pgs)
            print('%d/%d completed' % (self.patchNdx, dsSize))
            QApplication.processEvents()
        self.ui.pgsBar.setValue(100)
        with open('%s/bboxes.json' % (strOutFolder), 'w', encoding='utf-8') as fd:
            json.dump(self.lstPatches, fd, indent=4)
        self.ui.statusBar.showMessage('制作了%d/%d张图片于%s' % (self.patchNdx, dsSize, strOutFolder), 5000)
        self.ui.tmrToHidePgsBar.start()
        # self.ui.pgsBar.setVisible(False)

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
    mainLogic = MainAppLogic(ui, MainWindow)
    app.exec()
    # sys.exit(app.exec_())
