import sys
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QCheckBox, QWidget, QApplication, QMainWindow, QMessageBox, QStatusBar, QFileDialog
from PyQt5.QtCore import Qt
import widertools
import json
import os.path as path
import os
import numpy as np
import widerface2voc as w2v
import time
import shutil
import wf_utils
import crowdhuman_utils as ch_utils
from patcher import DelTree
import patcher
class MainAppLogic():
    def __init__(self, ui:widertools.Ui_MainWindow, mainWindow):
        self.ui = ui
        self.mainWindow = mainWindow
        self.prevImgItem = None
        ui.cmbDSType.addItems(['wider_face', 'crowd_human', 'voc', 'coco'])
        ui.cmbDSType.setCurrentIndex(1)
        ui.cmbSubSet.addItems(['train','val'])
        ui.cmbMaxFacesPerCluster.addItems(['10', '9', '8', '7', '6','5', '4', '3', '2'])
        ui.cmbCloseRatio.addItems(['0.5', '0.4', '0.32', '0.25', '0.2', '0.16', '0.125', '0.1', '0.08'])
        ui.cmbCloseRatio.setCurrentIndex(2)
        #ui.cmbSubSet.currentIndexChanged.connect(lambda: LoadDataset(ui))
        ui.cmbSubSet.textActivated.connect(lambda: self.LoadDataset())
        #ui.cmbSubSet.highlighted.connect(lambda: LoadDataset(ui)) 
        ui.btnRandom.clicked.connect(self.OnClicked_Random)
        ui.btnSplitSingle.clicked.connect(self.OnClicked_SplitSingle)
        ui.btnGenSingleFaceDataSet.clicked.connect(self.OnClicked_GenSingleFaceDataset)
        ui.btnValidateSingleFaceDataSet.clicked.connect(self.OnClicked_ValidateSingleFaceDataset)
        ui.btnValidateMultiFaceDataSet.clicked.connect(self.OnClicked_ValidateMultiFaceDataset)        
        ui.btnGenMultiFaceDataSet.clicked.connect(self.OnClicked_GenMultiFaceDataset)
        ui.btnDSFolder.clicked.connect(self.OnClicked_DSFolder)
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
        self.strOutFolder = ''
        self.chkTags = []
        self.LoadDataset('q:/datasets/dataset_crowd/')

    def OnTimeout_tmrToHidePgsBar(self):
        self.ui.pgsBar.setVisible(False)

    def ShowImage(self, table, ndx, strKey):
        item = self.dataObj.dctFiles[strKey]
        self.prevImgItem = [table, ndx, strKey, item]
        c = table.shape
        qImg = QtGui.QImage(bytearray(table), c[1], c[0], c[1]*3, QtGui.QImage.Format_BGR888)
        pix = QPixmap(QPixmap.fromImage(qImg))
        
        #pix2 = pix.scaled(32,32, Qt.KeepAspectRatio)#, Qt.SmoothTransformation)
        rect = ui.lblImg.rect()
        pix3 = pix.scaled(rect.width(),rect.height(), Qt.KeepAspectRatio)     #, Qt.SmoothTransformation)   

        
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
        sOutFile = 'labels_%s.json' % (self.ui.cmbSubSet.currentText())
        with open(sOutFile, 'w') as fd:
            json.dump(self.dataObj.dctFiles, fd, indent=4)
        # QMessageBox.information(None,'box', '已保存到%s' % sOutFile)
        self.ui.statusBar.showMessage('已保存到%s' % sOutFile, 5000)

    def OnClicked_Random(self):
        [table,ndx, strKey] = self.dataObj.ShowRandom(False, allowedTags=self.GetAllowedTags())
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

    def OnClicked_DSFolder(self):
        dir_choose = QFileDialog.getExistingDirectory(MainWindow,  
                                    "选取文件夹",  
                                    './') # 起始路径

        if dir_choose == "":
            print("\n取消选择")
            return

        print("\n你选择的文件夹为:")
        print(dir_choose)
        self.LoadDataset(dir_choose)

    def GetNextFreeFolder(self, strPrimary, isReplace=True):
        sTry = strPrimary
        if isReplace == True:
            if path.exists(sTry):
                shutil.rmtree(sTry)
        else:
            ndx = 1
            while path.exists(sTry):
                now = int(time.time())
                #转换为其他日期格式,如:"%Y-%m-%d %H:%M:%S"
                timeArray = time.localtime(now)
                otherStyleTime = time.strftime("%Y-%m-%d-%H-%M-%S", timeArray)            
                sTry = '%s_%s' % (strPrimary, otherStyleTime)
                ndx += 1
        return sTry

    def OnClicked_GenMultiFaceDataset(self):
        cnt = len(self.dataObj.dctFiles.keys())
        ndc = np.arange(cnt)
        np.random.shuffle(ndc)
        strOutFolder = './out_%s_multi' % (self.ui.cmbSubSet.currentText())
        strOutFolder = self.GetNextFreeFolder(strOutFolder)
        self.strOutFolder = strOutFolder
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
        lstAllowed = self.GetAllowedTags()        
        self.ui.tmrToHidePgsBar.stop()
        for ndx in ndc:
            #[table,ndx, strKey] = self.dataObj.ShowImage(ndx, False)
            #self.ShowImage(table, ndx, strKey)
            self.ui.statusBar.showMessage('制作中, 图片%d' % ndx, 3600000)
            self.patchNdx, lstPatches = self.dataObj.CutClusterPatches(strOutFolder, self.patchNdx, ndx=ndx, 
                closeRatio=closeRatio, maxObjPerCluster=maxObjPerCluster, outSize=[outX, outY], allowedTags=lstAllowed)
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
        strOutFolder = './out_%s_single' % (self.ui.cmbSubSet.currentText())
        strOutFolder = self.GetNextFreeFolder(strOutFolder)
        self.strOutFolder = strOutFolder
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
        lstAllowed = self.GetAllowedTags()
        for ndx in ndc:
            #[table,ndx, strKey] = self.dataObj.ShowImage(ndx, False)
            #self.ShowImage(table, ndx, strKey)            
            self.ui.statusBar.showMessage('制作中, 图片%d' % ndx, 3600000)
            self.patchNdx, lstPatches = self.dataObj.CutPatches(
                strOutFolder, self.patchNdx, ndx=ndx, outSize=[outX, outY], allowedTags=lstAllowed)
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
        strOutFolder = './out_%s_%s' % (self.ui.cmbSubSet.currentText(), strSel)
        table = self.dataObj.ShowRandomValidate(strOutFolder)
        c = table.shape
        qImg = QtGui.QImage(bytearray(table), c[1], c[0], c[1]*3, QtGui.QImage.Format_BGR888)
        pix = QPixmap(QPixmap.fromImage(qImg))

        rect = self.ui.lblImg.rect()
        pix3 = pix.scaled(rect.width(),rect.height(), Qt.KeepAspectRatio)
        # ui.imgWnd.setPixmap(pix2)
        self.ui.lblImg.setPixmap(pix3)             

    def OnClicked_ValidateSingleFaceDataset(self, strSel='single'):
        self.OnClicked_ValidateFaceDataset('single')

    def OnClicked_ValidateMultiFaceDataset(self, strSel='single'):
        self.OnClicked_ValidateFaceDataset('multi')        
              
    def GetAllowedTags(self):
        lstTags = []
        for item in self.chkTags:
            chk = item[0]
            if chk.isChecked():
                lstTags.append(chk.text())
        return lstTags
    def LoadDataset(self, dsFolder):
        # QMessageBox.information(None,'box',ui.cmbSubSet.currentText())
        # ui.cmbSubSet.currentData
        provider = None
        dsType = ui.cmbDSType.currentText()
        self.ui.statusBar.showMessage('数据集读取中...')
        QApplication.processEvents()
        try:
            setSel = self.ui.cmbSubSet.currentText()
            if dsType == 'wider_face':
                provider = wf_utils.WFUtils(dsFolder, setSel)
            elif dsType == 'crowd_human':
                provider = ch_utils.CrowdHumanUtils(dsFolder, setSel)
        except Exception as e:
            print(e)
            self.ui.statusBar.showMessage('代码错误：\n' + str(e))

        if provider is None or provider.dctFiles is None or provider.dctFiles == {}:
            self.ui.statusBar.showMessage('%s中的数据集无法按%s解析！' % (dsFolder, dsType) )            
        else:
            self.dsFolder = dsFolder
            self.provider = provider
            _translate = QtCore.QCoreApplication.translate
            MainWindow.setWindowTitle(_translate("MainWindow", "数据集化简分割工具 - 源数据集路径：%s" % (self.dsFolder)))            
            self.dataObj = patcher.Patcher(provider)
            self.ui.statusBar.showMessage('数据集读取成功!')
            for item in self.chkTags:
                chk = item[0]
                chk.hide()
                chk.deleteLater()
            self.chkTags = []
            topFiller = QWidget()
            for (i, tag) in enumerate(list(self.provider.GetTagSet())):
                chk = QCheckBox(topFiller)
                chk.setText(tag)
                chk.setChecked(True)
                chk.isChecked()                
                self.chkTags.append([chk, chk.text()])
                self.chkTags.sort(key=lambda x:x[1], reverse=False)
            for (i, item) in enumerate(self.chkTags):
                item[0].move(4, 5 + i * 20)
            topFiller.setMinimumSize(100, len(self.chkTags) * 20)
            mainUI.scrollTags.setWidget(topFiller)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = widertools.Ui_MainWindow()
    ui.setupUi(MainWindow)
    mainUI = ui
    MainWindow.show()
    mainLogic = MainAppLogic(ui, MainWindow)
    app.exec()
    # sys.exit(app.exec_())
