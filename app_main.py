import sys
from typing import AbstractSet
from PyQt5 import QtGui, QtCore
import PyQt5
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QCheckBox, QWidget, QApplication, QMainWindow, QMessageBox, QStatusBar, QFileDialog, QInputDialog, QComboBox
from PyQt5.QtCore import Qt
try:
    import qdarkstyle
except:
    pass
from numpy.lib.type_check import isreal
import widertools
import json
import os.path as path
import os
import numpy as np
import time
import shutil
import importlib
from patcher import DelTree
import patcher
import glob
import plugins_dsread.abstract_utils as abstract_utils
import langs
class MainAppLogic():
    def ScanPlugIns(self, sGlobPattern='./plugins_dsread/*_utils.py', cmb:QComboBox=None,
    dctPlugins:dict=None):
        lstTypes = [x[2:-3] for x in glob.glob(sGlobPattern)]
        lstTypes = [x.replace('\\', '.') for x in lstTypes]
        lstTypes = [x.replace('/', '.') for x in lstTypes]        
        pluginCnt = 0
        for plugin in lstTypes:
            a = importlib.import_module(plugin)
            dsType = a.GetDSTypeName()
            dsCls = a.GetUtilClass()
            if dsType is None:
                # None类型，用于表示抽象类
                continue
            if not isinstance(dsType, list):
                dsType = [dsType]
                dsCls = [dsCls]
            for (i,ds) in enumerate(dsType):
                cmb.addItem(ds)
                dctPlugins[ds] = dsCls[i]
                pluginCnt += 1
        # cmb.setCurrentIndex(pluginCnt - 1)

    def SwitchLang(self, newLang:str):
        ui = self.ui
        self.lang = newLang
        self.curMenuLang.setChecked(False)
        if newLang == 'ENG':
            self.curMenuLang = ui.menuLang_ENG
            self.dctLang = langs.g_dctENG
        elif newLang == 'CHS':
            self.curMenuLang = ui.menuLang_CHS
            self.dctLang = langs.g_dctCHS
        self.curMenuLang.setChecked(True)
        ui.MyretranslateUi(self.dctLang, MainWindow)
        
    
    def __init__(self, ui:widertools.Ui_MainWindow, mainWindow):
        self.ui:MyMainUI = ui
        self.mainWindow = mainWindow
        self.oriImage = '' # 当前子块的原始图片对象
        self.dsFolder = '' # 当前已经读取了有效数据集的目录
        self.nextDSFolder = '' # 下次从哪个目录读取数据集
        self.isToAbort = False  # 在生成子块数据集期间是否要中止
        self.patchNdx = 0  # 当前已生成的patch的数量
        self.lstPatches = [] # 记录当前已经生成的patch，每个元素是一个字典
        self.strOutFolder = '' # 子块数据集的输出目录
        self.chkTags = []  # 记录动态生成的表示类别名称和实例数量的复选框
        self.nextDSFolder = 'q:/datasets/wider_face'
        if not path.exists('./uicfgs'):
            os.makedirs('./uicfgs')
        self.rndNdx = 2305
        self.dctDsReadPlugins = dict() # 读取各种数据集的插件字典，键为数据集类型名，值为读取数据集的对象
        self.dctDsExportPlugins = dict()
        self.ScanPlugIns('./plugins_dsread/*_utils.py', ui.cmbDSType, self.dctDsReadPlugins)
        self.ScanPlugIns('./plugins_export/*_export.py', ui.cmbExportDSType, self.dctDsExportPlugins)
        self.lang = 'CHS'
        # 获取系统语言
        self.curMenuLang = ui.menuLang_CHS
        import locale
        lang = 'ENG'
        lang,enc = locale.getdefaultlocale()
        
        if lang.upper().find('CN') >= 0:
            lang = 'CHS'
        else:
            lang = 'ENG'
        # if self.lang == 'ENG':
        self.SwitchLang(lang)
            

        ui.cmbSubSet.addItems(['train','val', 'any'])
        ui.cmbMaxObjsPerCluster.addItems(['10', '9', '8', '7', '6','5', '4', '3', '2'])
        ui.cmbMaxObjsPerCluster.setCurrentIndex(8)
        #ui.cmbMinCloseRate.addItems(['0.5', '0.4', '0.32', '0.25', '0.2', '0.16', '0.125', '0.1', '0.08'])
        #ui.cmbMinCloseRate.setCurrentIndex(3)
        #else:
        #    self.SaveCfgDict()

        def CalcCmbValues(minVal, maxVal, step, isSquare=False):
            curVal = maxVal
            lstRet = []
            while curVal >= minVal:
                if isSquare:
                    outVal = curVal**0.5 * 100
                    sUnit = '% ^2'
                else:
                    outVal = curVal * 100
                    sUnit = '%'
                curVal /= step
                strVal = '%0.5f' % (outVal)
                lstRet.append(strVal[:4] + sUnit)
            return lstRet

        ui.cmbMinAreaRate.addItems(CalcCmbValues(1/2048, 0.5, 2**0.5, True))
        ui.cmbMaxAreaRate.addItems(CalcCmbValues(16/2048, 1.0, 2**0.5, True))
        ui.cmbMinCloseRate.addItems(CalcCmbValues(0.01, 0.6, 1.25, False))
        ui.cmbMinAreaRate.setCurrentIndex(12)
        ui.cmbMaxAreaRate.setCurrentIndex(0)
        ui.cmbMinCloseRate.setCurrentIndex(6)
        #ui.cmbSubSet.currentIndexChanged.connect(lambda: LoadDataset(ui))
        self.SetEnableStateBaseedOnDatasetAvailibiblity(False)
        def _SetAbort(self):
            self.isToAbort = True
        ui.btnAbort.clicked.connect(lambda: _SetAbort(self))
        ui.btnAbort.setEnabled(False)
        ui.btnForceLoadDS.clicked.connect(lambda: self.LoadDataset(self.nextDSFolder, True))
        ui.cmbSubSet.textActivated.connect(lambda: self.LoadDataset(self.nextDSFolder))
        #ui.cmbSubSet.highlighted.connect(lambda: LoadDataset(ui)) 
        ui.btnRandom.clicked.connect(self.OnClicked_Random)
        ui.btnGenSingleFaceDataSet.clicked.connect(lambda: self.OnClicked_GenPatchDataset(singleStr='single'))
        ui.btnValidateSingleFaceDataSet.clicked.connect(self.OnClicked_ValidateSingleFaceDataset)
        ui.btnValidateMultiFaceDataSet.clicked.connect(self.OnClicked_ValidateMultiFaceDataset) 
        ui.btnOriImg.clicked.connect(self.OnClicked_OriImage)       
        ui.btnGenMultiFaceDataSet.clicked.connect(lambda: self.OnClicked_GenPatchDataset())
        ui.btnTagSelAll.clicked.connect(self.OnClicked_TagSelAll)
        ui.btnTagSelInv.clicked.connect(self.OnClicked_TagSelInv)        
        ui.btnDSFolder.clicked.connect(self.OnClicked_DSFolder)

        ui.btnExport.clicked.connect(lambda :self.OnClicked_ScanAndMayExport())
        ui.btnRefreshLabels.clicked.connect(lambda :self.OnClicked_ScanAndMayExport(isToExport=False))
        
        ui.menuLang_CHS.triggered.connect(lambda: self.SwitchLang('CHS'))
        ui.menuLang_ENG.triggered.connect(lambda: self.SwitchLang('ENG'))

        ui.menuLoadConfig.triggered.connect(lambda: self.OnTriggered_MenuLoadUiCfg())
        ui.menuSaveConfigAs.triggered.connect(lambda: self.OnTriggered_MenuSaveUiCfgAs())        
        ui.menuSaveConfig.triggered.connect(lambda: self.OnTriggered_MenuSaveUiCfgAs(strPath=self.uiCfgFile))
        
        ui.menuAboxTool.triggered.connect(lambda: self.LaunchABoxToolInNewProcess())
        ui.menuDbgGenMultiForCurrent.triggered.connect(lambda: self.OnClicked_GenPatchDataset(ndcIn=[self.rndNdx]))
        ui.menuDelNonCheckedTags.setVisible(False)
        ui.btnDelNonCheckedTags.setEnabled(False)
        ui.menuDelNonCheckedTags.triggered.connect(lambda: self.DelTags())
        ui.btnDelNonCheckedTags.clicked.connect(lambda: self.DelTags())
        ui.menuSpecifyImageNdx.triggered.connect(lambda: self.OnMenuTriggered_SpecifyImageNdx())
        ui.pgsBar.setVisible(False)
        ui.tmrToHidePgsBar = QtCore.QTimer(self.mainWindow)
        ui.tmrToHidePgsBar.setInterval(300)
        ui.tmrToHidePgsBar.setSingleShot(True)
        ui.tmrToHidePgsBar.timeout.connect(self.OnTimeout_tmrToHidePgsBar)
        ui.statusBar = QStatusBar()
        MainWindow.setStatusBar(ui.statusBar)
        ui.statusBar.addPermanentWidget(ui.pgsBar)
        #ui.btnValidateMultiFaceDataSet.clicked.connect(self.OnClicked_ValidateSingleFaceDataset)

        # ui.btnGenMultiFaceDataSet.setEnabled(False)
        # ui.btnValidateMultiFaceDataSet.setEnabled(False)
        # ui.cmbMaxObjsPerCluster.setEnabled(False)
        ui.btnSaveOriBBoxes.clicked.connect(self.OnClicked_SaveOriBBoxes)
        if path.exists('./uicfgs/_ui_cfg_auto.uicfg'):
            cfgDct = self.LoadCfgDict()
            self.uiCfgFile = './uicfgs/_ui_cfg_auto.uicfg'
            mainUI.menuSaveConfig.setText('Save Config (%s)' % self.uiCfgFile)
        else:
            self.uiCfgFile = ''        
            if ui.chkAutoload.isChecked():
                self.LoadDataset(self.nextDSFolder)
    
    def OnTriggered_MenuLoadUiCfg(self):
        lstPaths = QFileDialog.getOpenFileName(MainWindow, '读取UI配置', './uicfgs', 'UICFG(*.uicfg)')
        if len(lstPaths) > 1:
            if path.exists(lstPaths[0]):
                self.LoadCfgDict(lstPaths[0])
                self.uiCfgFile = lstPaths[0]
                mainUI.menuSaveConfig.setText('Save Config (%s)' % self.uiCfgFile)

    def OnTriggered_MenuSaveUiCfgAs(self, strPath = ''):
        if len(strPath) < 1:
            lstPaths = QFileDialog.getSaveFileName(MainWindow, '保存UI配置', './uicfgs', 'UICFG(*.uicfg)')
        else:
            lstPaths = [strPath, 'JSON(*.json)']
        if len(lstPaths) == 2 and len(lstPaths[0]) > 0:
            self.SaveCfgDict(lstPaths[0])
            self.uiCfgFile = lstPaths[0]
            mainUI.menuSaveConfig.setText('Save Config (%s)' % self.uiCfgFile)

    def SaveCfgDict(self, savePath = './uicfgs/_ui_cfg_auto.uicfg'):
        mainName = path.split(savePath)[1]
        dctRet = {
            'lang' : self.lang,
            'version': '2022.02.07',
            'savePath': savePath,
            'DSType' : mainUI.cmbDSType.currentText(),
            'DSTypeNdx' : mainUI.cmbDSType.currentIndex(),
            'DSSubset': mainUI.cmbSubSet.currentText(),
            'DSSubsetNdx' : mainUI.cmbSubSet.currentIndex(),
            'IsDSAutoLoad' : mainUI.chkAutoload.isChecked(),
            'MinHvsW' : mainUI.txtMinHvsW.text(),
            'MaxHvsW' : mainUI.txtMaxHvsW.text(),
            'MinGTPerImg' : mainUI.txtMinGTPerImg.text(),
            'MaxGTPerImg' : mainUI.txtMaxGTPerImg.text(),
            'IsSkipDirtyImage' : mainUI.chkSkipDirtyImage.isChecked(),

            'MinAreaRate': mainUI.cmbMinAreaRate.currentText(),
            'MaxAreaRate' : mainUI.cmbMaxAreaRate.currentText(),
            'MinAreaRateNdx': mainUI.cmbMinAreaRate.currentIndex(),
            'MaxAreaRateNdx' : mainUI.cmbMaxAreaRate.currentIndex(), 
            'MinCloseRate': mainUI.cmbMinCloseRate.currentText(),
            'MinCloseRateNdx' : mainUI.cmbMinCloseRate.currentIndex(),
            'MaxObjsPerCluster' : mainUI.cmbMaxObjsPerCluster.currentText(),
            'MaxObjsPerClusterNdx' : mainUI.cmbMaxObjsPerCluster.currentIndex(),
            'IsAllowMore' : mainUI.chkAllowMoreObj.isChecked(),
            'txtOutY' : mainUI.txtOutY.text(),
            'txtOutX' : mainUI.txtOutX.text(),
            'txtOutN' : mainUI.txtOutN.text(),
            'IsSkipDirtyPatch' : mainUI.chkSkipDirtyPatch.isChecked(),

            'IsOutHasTmStmp' : mainUI.chkOutHasTmStmp.isChecked(),

            'nextDSFolder' : self.nextDSFolder,
            'rndNdx' : self.rndNdx
        }

        lstTags = []
        for item in self.chkTags:
            chk = item[0]
            if chk.isChecked():
                oriText = chk.text()
                lstTags.append(oriText.split(':')[0].strip())
        dctRet['allowedTags'] = lstTags

        if savePath != '':
            with open(savePath, 'w') as fd:
                json.dump(dctRet, fd, indent=4)
        return dctRet

    def LoadCfgDict(self, loadPath = './uicfgs/_ui_cfg_auto.uicfg', isApply=True):
        if not path.exists(loadPath):
            return -1
        with open(loadPath) as fd:
            cfgDct = json.load(fd)
        if isApply:
            mainUI.cmbDSType.setCurrentText(cfgDct['DSType'])
            if mainUI.cmbDSType.currentIndex() != (cfgDct['DSTypeNdx']): print('index changed!')
            mainUI.cmbSubSet.setCurrentText(cfgDct['DSSubset'])
            if mainUI.cmbSubSet.currentIndex() != (cfgDct['DSSubsetNdx']): print('index changed!')
            mainUI.chkAutoload.setChecked(cfgDct['IsDSAutoLoad'])
            mainUI.txtMinHvsW.setText(cfgDct['MinHvsW'])
            mainUI.txtMaxHvsW.setText(cfgDct['MaxHvsW'])
            mainUI.txtMinGTPerImg.setText(cfgDct['MinGTPerImg'])
            mainUI.txtMaxGTPerImg.setText(cfgDct['MaxGTPerImg'])
            mainUI.chkSkipDirtyImage.setChecked(cfgDct['IsSkipDirtyImage'])

            mainUI.cmbMinAreaRate.setCurrentText(cfgDct['MinAreaRate'])
            mainUI.cmbMaxAreaRate.setCurrentText(cfgDct['MaxAreaRate'])
            mainUI.cmbMinCloseRate.setCurrentText(cfgDct['MinCloseRate'])
            if mainUI.cmbMinAreaRate.currentIndex() != (cfgDct['MinAreaRateNdx']): print('index changed!')
            if mainUI.cmbMaxAreaRate.currentIndex() != (cfgDct['MaxAreaRateNdx']): print('index changed!') 
            if mainUI.cmbMinCloseRate.currentIndex() != (cfgDct['MinCloseRateNdx']): print('index changed!')
            mainUI.cmbMaxObjsPerCluster.setCurrentText(cfgDct['MaxObjsPerCluster'])
            if mainUI.cmbMaxObjsPerCluster.currentIndex() != (cfgDct['MaxObjsPerClusterNdx']): print('index changed!')
            mainUI.chkAllowMoreObj.setChecked(cfgDct['IsAllowMore'])
            mainUI.txtOutY.setText(cfgDct['txtOutY'])
            mainUI.txtOutX.setText(cfgDct['txtOutX'])
            mainUI.txtOutN.setText(cfgDct['txtOutN'])
            mainUI.chkSkipDirtyPatch.setChecked(cfgDct['IsSkipDirtyPatch'])

            mainUI.chkOutHasTmStmp.setChecked(cfgDct['IsOutHasTmStmp'])
            self.nextDSFolder = cfgDct['nextDSFolder']
            
            if 'lang' in cfgDct.keys():
                self.SwitchLang(cfgDct['lang'])

            # Following sentences can depend on self.cfgDct
            self.cfgDct = cfgDct
            
            if mainUI.chkAutoload.isChecked():
                # if self.nextDSFolder != self.dsFolder:
                self.LoadDataset(self.nextDSFolder)
        return cfgDct

    def DelTags(self):
        self.dataObj.FilterTags(self.GetDisallowedTags())
        self.UpdateDataset(self.provider, self.dsFolder, dsType = mainUI.cmbDSType.currentText())
    
    def LaunchABoxToolInNewProcess(self):
        import subprocess as sp
        sp.Popen('python ./abox_tools/abox_main.py', shell=True)
        #sp.call(["python", "./abox_tools/abox_main.py", ' '.join(sys.argv)], shell = True)

    def ShowAt(self, ndx):
        try:
            [table, strKey, _,  ndx] = self.dataObj.ShowAt(ndx, False, allowedTags=self.GetAllowedTags())
            self.ui.statusBar.showMessage('显示指定编号%d, 图片%s' % (ndx, strKey))
            self.ShowImage(table, strKey)        
            self.rndNdx = ndx
        except:
            self.ui.statusBar.showMessage('无效的图片号%d' % (ndx))
    
    def OnMenuTriggered_SpecifyImageNdx(self):
        ndx, isOK = QInputDialog.getInt(MainWindow, '设置当前图片索引', '请输入索引：', min = 0) 
        if isOK:
            self.ShowAt(ndx)

    def SetEnableStateBaseedOnDatasetAvailibiblity(self,isEn):
            self.ui.btnRandom.setEnabled(isEn)
            self.ui.btnSaveOriBBoxes.setEnabled(isEn)
            self.ui.btnGenSingleFaceDataSet.setEnabled(isEn)
            self.ui.btnGenMultiFaceDataSet.setEnabled(isEn) 
            self.ui.menuDbgGenMultiForCurrent.setEnabled(isEn)
            self.ui.menuSpecifyImageNdx.setEnabled(isEn)
            self.ui.menuDelNonCheckedTags.setEnabled(isEn)
            self.ui.btnTagSelAll.setEnabled(isEn)
            self.ui.btnTagSelInv.setEnabled(isEn)
            self.ui.btnValidateSingleFaceDataSet.setEnabled(isEn)
            self.ui.btnValidateMultiFaceDataSet.setEnabled(isEn)
            self.ui.btnOriImg.setEnabled(isEn)

    def OnTimeout_tmrToHidePgsBar(self):
        self.ui.pgsBar.setVisible(False)

    def ShowImage(self, table, strKey):
        item = self.dataObj.dctFiles[strKey]
        c = table.shape
        qImg = QtGui.QImage(bytearray(table), c[1], c[0], c[1]*3, QtGui.QImage.Format_BGR888)
        pix = QPixmap(QPixmap.fromImage(qImg))
        
        #pix2 = pix.scaled(32,32, Qt.KeepAspectRatio)#, Qt.SmoothTransformation)
        rect = ui.lblImg.rect()
        pix3 = pix.scaled(rect.width(),rect.height(), Qt.KeepAspectRatio)     #, Qt.SmoothTransformation)   

        
        # ui.imgWnd.setPixmap(pix2)
        ui.lblImg.setPixmap(pix3)

    def OnClicked_ScanAndMayExport(self, isToExport = True):
        def callback(pgs, msg=''):
            self.ui.pgsBar.setValue(pgs)
            if len(msg) > 0:
                self.ui.statusBar.showMessage(msg)
            QApplication.processEvents()

        def _doScanAndDelInvalidBBoxEntries(strInPath, maxCnt=1E7, callback=None):
            if path.exists(strInPath):
                with open(strInPath + '/bboxes.json') as fd:
                    lstBBoxes = json.load(fd)
            else:
                return -1

            cnt = 0
            total = len(lstBBoxes)

            lstNewBBoxes = list()
            for item in lstBBoxes:
                imgFile = './outs/' + item['filename']
                if path.exists(imgFile):
                    lstNewBBoxes.append(item)
            if len(lstNewBBoxes) < len(lstBBoxes):
                with open(strInPath + '/bboxes.json', 'w') as fd:
                    lstBBoxes = lstNewBBoxes
                    json.dump(lstNewBBoxes, fd)
            return cnt
            
        def ScanAndDelInvalidBBoxEntries(subsetSel='train', cntSel='multi', maxCnt=1E7, callback=None):
            lst = glob.glob('./outs/out_%s_%s*' % (subsetSel, cntSel))
            lst = [x.replace('\\', '/') for x in lst]
            lst = list(filter(lambda x: path.isdir(x) == True, lst))
            lstInPaths = lst            
            for strPath in lstInPaths:
                _doScanAndDelInvalidBBoxEntries(strPath, maxCnt, callback)

        self.ui.pgsBar.setVisible(True)
        for subsetSel in ['train', 'val', 'test', 'any']:
            for cntSel in ['single', 'multi']:
                self.ui.pgsBar.setValue(1)
                self.ui.statusBar.showMessage('正在清洗%s %s' % (subsetSel, cntSel), 60000)
                QApplication.processEvents()
                ScanAndDelInvalidBBoxEntries(subsetSel, cntSel, callback=callback)
                if isToExport:
                    self.ui.statusBar.showMessage('导出中...')
                    dsType = ui.cmbExportDSType.currentText()
                    exporter = self.dctDsExportPlugins[dsType](subsetSel, cntSel)
                    exporter.Export(callback=callback)
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
        [table, strKey, _, ndx] = self.dataObj.ShowRandom(False, allowedTags=self.GetAllowedTags())
        self.rndNdx = ndx
        self.ui.statusBar.showMessage('随机显示编号%d, 图片%s' % (ndx, strKey))
        self.ShowImage(table, strKey)

    def OnClicked_DSFolder(self):
        dir_choose = QFileDialog.getExistingDirectory(MainWindow,  
                                    "选取文件夹",  
                                    './') # 起始路径

        if dir_choose == "":
            print("\n取消选择")
            return

        print("\n你选择的文件夹为:")
        print(dir_choose)
        self.nextDSFolder = dir_choose        
        self.LoadDataset(dir_choose)

    def GetNextFreeFolder(self, strPrimary, isReplace=True):
        sTry = strPrimary
        if isReplace == True:
            if path.exists(sTry):
                shutil.rmtree(sTry)
        else:
            ndx = 1
            while True:
                now = int(time.time())
                #转换为其他日期格式,如:"%Y-%m-%d %H:%M:%S"
                timeArray = time.localtime(now)
                otherStyleTime = time.strftime("%Y-%m-%d-%H-%M-%S", timeArray)            
                sTry = '%s_%s' % (strPrimary, otherStyleTime)
                ndx += 1
                if not path.exists(sTry):
                    break
        return sTry

    def _StatUsage(self, dbgSkips):
        totalCnt = dbgSkips[0]
        lstReasons = [
            'total candidates ',            
            'total usage rate ',
            'bad aspect ratio ', 
            'not close enough ',
            'object out bound ',
            'object bad size  ',
            'objects too dense',
            'too many patches ',
        ]
        
        useRate = 1
        lstRets = [0,0,0,0,0,0,0,0]
        
        lstRets[0] = totalCnt
        useRate = 1
        if totalCnt != 0:
            for i in range(1, 7):
                lstRets[i+1] = dbgSkips[i] / totalCnt
                useRate -= lstRets[i+1]
            lstRets[1] = useRate
            strOut = lstReasons[0] + ' : ' + str(lstRets[0])
            for i in range(1, 8):
                strOut += '\n' + lstReasons[i] + ' : ' + '%02.1f%%' % (lstRets[i]*100)
            QMessageBox.information(MainWindow, '生成结果统计', strOut)
        else:
            QMessageBox.information(MainWindow, '生成结果统计', '没有生成任何数据')            

    def OnClicked_GenPatchDataset(self, ndcIn=[], singleStr='multi'):
        self.dsFolder = self.dsFolder
        if len(ndcIn) == 0:
            cnt = len(self.dataObj.dctFiles.keys())
            ndc = np.arange(cnt)
            maxPatchPerImg = 10
        else:
            # 为指定图片生成子块，多是调试目的
            cnt = len(ndcIn)
            ndc = np.array(ndcIn)
            # 调试目的下基本不限制每个图片生成的子块数
            maxPatchPerImg = 50
        np.random.shuffle(ndc)

        strOutFolder = './outs/out_%s_%s' % (self.ui.cmbSubSet.currentText(), singleStr)
        isReplace = True
        if path.abspath(strOutFolder) == path.abspath(self.dsFolder):
            isReplace = False
        elif mainUI.chkOutHasTmStmp.isChecked() == True:
            isReplace = False
        strOutFolder = self.GetNextFreeFolder(strOutFolder, isReplace=isReplace)
        if path.abspath(strOutFolder) == path.abspath(self.dsFolder):
            self.ui.statusBar.showMessage('不能制作 - 源数据集路径不能与输出路径相同！')
            return
        self.strOutFolder = strOutFolder
        outW = int(self.ui.txtOutX.text())
        outH = int(self.ui.txtOutY.text())        
        self.patchNdx = 0
        self.lstPatches = []
        self.ui.pgsBar.setValue(1)
        self.ui.pgsBar.setVisible(True)        
        dsSize = int(self.ui.txtOutN.text())
        if not path.exists(strOutFolder):
            os.makedirs(strOutFolder)
        maxObjPerCluster = int(self.ui.cmbMaxObjsPerCluster.currentText())
        minAreaRate = (float(self.ui.cmbMinAreaRate.currentText()[:4]) / 100.0) ** 2
        maxAreaRate = (float(self.ui.cmbMaxAreaRate.currentText()[:4]) / 100.0) ** 2  

        minClose = float(self.ui.cmbMinCloseRate.currentText()[:4]) / 100.0
        self.ui.pgsBar.setValue(1)
        self.ui.pgsBar.setVisible(True)
        lstAllowed = self.GetAllowedTags()
        self.ui.tmrToHidePgsBar.stop()
        dbgSkips = [0,0,0,0,0,0,0]
        mainUI.btnAbort.setEnabled(True)
        for ndx in ndc:
            #[table,ndx, strKey] = self.dataObj.ShowImage(ndx, False)
            #self.ShowImage(table, ndx, strKey)
            self.ui.statusBar.showMessage('制作中, 图片%d' % ndx, 3600000)
            if singleStr == 'multi':
                self.patchNdx, lstPatches = self.dataObj.CutClusterPatches(
                    strOutFolder, self.patchNdx, ndx=ndx, minCloseRate=minClose, 
                    maxObjPerCluster=maxObjPerCluster, 
                    isAllowMorePerPatch=mainUI.chkAllowMoreObj.isChecked(), 
                    isSkipDirtyPatch=mainUI.chkSkipDirtyPatch.isChecked(),
                    maxPatchPerImg=maxPatchPerImg,
                    areaRateRange=[minAreaRate, maxAreaRate], outWH=[outW, outH], allowedTags=lstAllowed,
                    dbgSkips=dbgSkips)
            else:
                self.patchNdx, lstPatches = self.dataObj.CutPatches(
                    strOutFolder, self.patchNdx, ndx=ndx, outWH=[outW, outH], 
                    isSkipDirtyPatch=mainUI.chkSkipDirtyPatch.isChecked(),
                    areaRateRange=[minAreaRate, maxAreaRate], allowedTags=lstAllowed,
                    dbgSkips=dbgSkips)

            self.lstPatches += lstPatches
            if self.patchNdx >= dsSize:
                break
            if self.isToAbort:
                self.isToAbort = False
                break            
            pgs = 100 * self.patchNdx / dsSize
            self.ui.pgsBar.setValue(pgs)
            QApplication.processEvents()
            print('%d/%d completed' % (self.patchNdx, dsSize))
        if hasattr(self.provider, 'TranslateTag'):
            for item in self.lstPatches:
                for xyxy in item['xyxys']:
                    for i in range(4, len(xyxy)):
                        xyxy[i] = self.provider.TranslateTag(xyxy[i])
        with open('%s/bboxes.json' % strOutFolder, 'w', encoding='utf-8') as fd:
            json.dump(self.lstPatches, fd, indent=4)
        self.ui.pgsBar.setValue(100)
        mainUI.btnAbort.setEnabled(False)
        self.ui.statusBar.showMessage('制作了%d/%d张图片于%s' % (self.patchNdx, dsSize, strOutFolder), 5000)
        self._StatUsage(dbgSkips)
        self.ui.tmrToHidePgsBar.start()
        #self.ui.pgsBar.setVisible(False)

    def OnClicked_ValidateDataset(self, strSel='single'):
        if self.strOutFolder == '':
            strOutFolder = './outs/out_%s_%s' % (self.ui.cmbSubSet.currentText(), strSel)
        else:
            strOutFolder = self.strOutFolder
        table, ndx, item = self.dataObj.ShowRandomValidate(strOutFolder)
        if table is None:
            self.ui.statusBar.showMessage('未找到制作的数据集')
            return
        c = table.shape
        qImg = QtGui.QImage(bytearray(table), c[1], c[0], c[1]*3, QtGui.QImage.Format_BGR888)
        pix = QPixmap(QPixmap.fromImage(qImg))

        rect = self.ui.lblImg.rect()
        pix3 = pix.scaled(rect.width(),rect.height(), Qt.KeepAspectRatio)
        # ui.imgWnd.setPixmap(pix2)
        self.ui.lblImg.setPixmap(pix3)
        self.ui.statusBar.showMessage('图片 %s' % item['filename']) 
        
        fileNameNoPath = item['filename'].split('/')[-1]
        [mainName, ext] = path.splitext(fileNameNoPath)
        mainName = '_'.join(mainName.split('_')[:-2])
        fileNameKey = mainName
        self.oriImage = self.dataObj.provider.MapFileKey(fileNameKey)
        bkpt = 0

    def OnClicked_ValidateSingleFaceDataset(self, strSel='single'):
        self.OnClicked_ValidateDataset('single')

    def OnClicked_ValidateMultiFaceDataset(self, strSel='single'):
        self.OnClicked_ValidateDataset('multi')        
    
    def OnClicked_OriImage(self):
        if len(self.oriImage) > 0:
            try:
                [table, strKey, _] = self.dataObj.ShowImageFile(self.oriImage, False, allowedTags=self.GetAllowedTags())
                self.ShowImage(table, strKey)        
            except:
                self.ui.statusBar.showMessage('未找到图片') 

    def OnClicked_TagSelAll(self):
        for item in self.chkTags:
            chk = item[0]
            chk.setChecked(True)

    def OnClicked_TagSelInv(self):
        for item in self.chkTags:
            chk = item[0]
            chk.setChecked(not chk.isChecked())

    def GetAllowedTags(self):
        lstTags = []
        for item in self.chkTags:
            chk = item[0]
            if chk.isChecked():
                oriText = chk.text()
                outText = ' :'.join(oriText.split(':')[:-1])[:-1]
                lstTags.append(outText)
        return lstTags

    def GetDisallowedTags(self):
        lstTags = []
        for item in self.chkTags:
            chk = item[0]
            if not chk.isChecked():
                oriText = chk.text()
                outText = ' :'.join(oriText.split(':')[:-1])[:-1]
                lstTags.append(outText)
        return lstTags        
    
    def UpdateDataset(self, provider:abstract_utils.AbstractUtils, dsFolder, dsType):

        if provider is None or provider.dctFiles is None or provider.dctFiles == {}:
            self.ui.statusBar.showMessage('%s中的数据集无法按%s解析！' % (dsFolder, dsType) )            
        else:
            mainUI.menuDelNonCheckedTags.setVisible(provider.CanDelTags())
            mainUI.btnDelNonCheckedTags.setEnabled(provider.CanDelTags())
            self.SetEnableStateBaseedOnDatasetAvailibiblity(True)
            self.dsFolder = dsFolder
            self.provider = provider
            _translate = QtCore.QCoreApplication.translate
            MainWindow.setWindowTitle(_translate("MainWindow", "数据集化简分割工具 - 已加载%s类型的数据集于：%s" % (mainUI.cmbDSType.currentText(), self.dsFolder)))            
            self.dataObj = patcher.Patcher(provider)
            self.ui.statusBar.showMessage('数据集%s (%s)读取成功!' % (dsFolder, dsType))
            for item in self.chkTags:
                chk = item[0]
                chk.hide()
                chk.deleteLater()
            self.chkTags = []
            topFiller = QWidget()
            dctTag = self.provider.GetTagDict()
            maxLineLen = 0
            
            for (i, tag) in enumerate(list(dctTag.keys())):
                chk = QCheckBox(topFiller)
                chk.setText(tag + ' : %d' % (dctTag[tag]))
                lineLen = len(chk.text())
                if lineLen > maxLineLen:
                    maxLineLen = lineLen
                chk.setChecked(True)
                chk.isChecked() 
                self.chkTags.append([chk, chk.text()])
                self.chkTags.sort(key=lambda x:x[1], reverse=False)
            if len(self.chkTags) == 1:
                chk.setEnabled(False)
            for (i, item) in enumerate(self.chkTags):
                item[0].move(4, 5 + i * 20)
            topFiller.setMinimumSize(8*maxLineLen, len(self.chkTags) * 20)
            mainUI.scrollTags.setWidget(topFiller)

            # 根据上次保存的配置来设置各checkbox的选中状态
            
            if hasattr(self, 'cfgDct'):
                cfgDct = self.cfgDct
                if cfgDct['nextDSFolder'] == dsFolder:
                    self.ShowAt(cfgDct['rndNdx'])
                    if 'allowedTags' in cfgDct.keys():
                        chkCnt = 0
                        for chk in self.chkTags:
                            oriText = chk[1]
                            tag = oriText.split(':')[0].strip()
                            isChk = tag in cfgDct['allowedTags']
                            if isChk:
                                chkCnt += 1
                            chk[0].setChecked(isChk)
                        if chkCnt == 0:
                            self.chkTags[0][0].setChecked(True)

            
    
    def LoadDataset(self, dsFolder, isForced=False):
        # QMessageBox.information(None,'box',ui.cmbSubSet.currentText())
        # ui.cmbSubSet.currentData
        if isForced == False and self.ui.chkAutoload.isChecked() == False:
            return
        def callback(pgs, msg=''):
            self.ui.pgsBar.setValue(pgs)
            if len(msg) > 0:
                self.ui.statusBar.showMessage(msg)
            QApplication.processEvents()
        
        self.ui.pgsBar.setValue(1)
        self.ui.pgsBar.setVisible(True)

        lstHvsW_config = [0.1, 10.0] # min, max
        lstNew = []
        for (i, txt) in enumerate([mainUI.txtMinHvsW.text(), mainUI.txtMaxHvsW.text()]):
            divSym = ''
            if ':' in txt:
                divSym = ':'
            elif '/' in txt:
                divSym = '/'
            
            try:
                if divSym != '':
                    lstVals = [float(x.strip()) for x in txt.split(divSym)]
                    valOut = lstVals[0] / lstVals[1]
                else:
                    valOut = float(txt.strip())
                lstNew.append(valOut)
            except:
                pass
        if len(lstNew) == 2 and lstNew[0] < lstNew[1]:
            lstHvsW_config = lstNew

        minGTPerImg = 1
        maxGTPerImg = 50
        lstGTPerImgs = [1, 50]
        lstNewCfg = [0,0]
        for (i, txt) in enumerate([mainUI.txtMinGTPerImg.text(), mainUI.txtMaxGTPerImg.text()]):
            try:
                lstNewCfg[i] = int(txt)
            except:
                lstNewCfg[i] = lstGTPerImgs[i]
        if lstNewCfg[0] == 0:
            lstNewCfg[0] = 1
        if lstNewCfg[0] <= lstNewCfg[1]:
            lstGTPerImgs = lstNewCfg
        dctCfg = {
            'minHvsW' : lstHvsW_config[0],
            'maxHvsW' : lstHvsW_config[1],
            'minGTPerImg': lstGTPerImgs[0],
            'maxGTPerImg': lstGTPerImgs[1],
            'isSkipDirtyImg': mainUI.chkSkipDirtyImage.isChecked()
        }

        provider: abstract_utils.AbstractUtils = None
        dsType = ui.cmbDSType.currentText()
        self.ui.statusBar.showMessage('数据集读取中...')
        QApplication.processEvents()
        try:
            setSel = self.ui.cmbSubSet.currentText()
            provider = self.dctDsReadPlugins[dsType](dsFolder, setSel, dctCfg, callback)

        except Exception as e:
            print(e)
            self.ui.statusBar.showMessage('代码错误：\n' + str(e))

        self.UpdateDataset(provider, dsFolder, dsType)
        self.ui.pgsBar.setValue(100)
        self.ui.tmrToHidePgsBar.start()

class MyMainUI(widertools.Ui_MainWindow):
    def __init__(self):
        super(MyMainUI, self).__init__()
    def MyretranslateUi(self, langDict, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", langDict["数据集化简分割工具"]))
        self.lblImg.setText(_translate("MainWindow", "TextLabel"))
        self.groupBox.setTitle(_translate("MainWindow", langDict["子块数据集配置"]))
        self.label_2.setText(_translate("MainWindow", langDict["合并至少几个邻近物体"]))
        self.label_5.setText(_translate("MainWindow", langDict["生成数量"]))
        self.txtOutN.setText(_translate("MainWindow", "300"))
        self.label_4.setText(_translate("MainWindow", langDict["子块 H"]))
        self.txtOutX.setText(_translate("MainWindow", "128"))
        self.txtOutY.setText(_translate("MainWindow", "128"))
        self.label_10.setText(_translate("MainWindow", langDict["单物体面积占比范围"]))
        self.label_6.setText(_translate("MainWindow", langDict["物体区域最小占比"]))
        self.label_12.setText(_translate("MainWindow", "W"))
        self.chkAllowMoreObj.setText(_translate("MainWindow", langDict["允许更密集"]))
        self.chkSkipDirtyPatch.setText(_translate("MainWindow", langDict["只使用长宽比符合约束的物体"]))
        self.groupBox_2.setTitle(_translate("MainWindow", langDict["源数据集选择"]))
        self.label.setText(_translate("MainWindow", langDict["子集"]))
        self.btnDSFolder.setText(_translate("MainWindow", langDict["选择目录..."]))
        self.label_13.setText(_translate("MainWindow", langDict["类型"]))
        self.btnForceLoadDS.setText(_translate("MainWindow", langDict["装载"]))
        self.chkAutoload.setText(_translate("MainWindow", langDict["自动装载"]))
        self.btnRandom.setText(_translate("MainWindow", langDict["随机显示"]))
        self.btnSaveOriBBoxes.setText(_translate("MainWindow", langDict["导出标注"]))
        self.txtMinHvsW.setText(_translate("MainWindow", "1/6"))
        self.label_7.setText(_translate("MainWindow", langDict["边框高/宽的范围"]))
        self.txtMaxHvsW.setText(_translate("MainWindow", "6/1"))
        self.label_8.setText(_translate("MainWindow", langDict["每张图中边框个数的范围"]))
        self.txtMinGTPerImg.setText(_translate("MainWindow", "1"))
        self.txtMaxGTPerImg.setText(_translate("MainWindow", "50"))
        self.chkSkipDirtyImage.setText(_translate("MainWindow", langDict["只读取符合约束的图"]))
        self.groupBox_3.setTitle(_translate("MainWindow", langDict["生成子块数据集"]))
        self.btnGenSingleFaceDataSet.setText(_translate("MainWindow", langDict["单框"]))
        self.btnGenMultiFaceDataSet.setText(_translate("MainWindow", langDict["多框"]))
        self.btnAbort.setText(_translate("MainWindow", langDict["中止"]))
        self.chkOutHasTmStmp.setText(_translate("MainWindow", langDict["输出目录有时间戳"]))
        self.groupBox_4.setTitle(_translate("MainWindow", langDict["验证子块数据集与工具"]))
        self.btnValidateMultiFaceDataSet.setText(_translate("MainWindow", langDict["多框"]))
        self.btnValidateSingleFaceDataSet.setText(_translate("MainWindow", langDict["单框"]))
        self.btnOriImg.setText(_translate("MainWindow", langDict["原图"]))
        self.btnRefreshLabels.setText(_translate("MainWindow", langDict["刷新标注"]))
        self.btnTagSelInv.setText(_translate("MainWindow", langDict["反选"]))
        self.label_9.setText(_translate("MainWindow", langDict["类别筛选"]))
        self.btnTagSelAll.setText(_translate("MainWindow", langDict["全选"]))
        self.btnDelNonCheckedTags.setText(_translate("MainWindow", langDict["删除未选中的类别"]))
        self.label_11.setText(_translate("MainWindow", langDict["导出数据集格式"]))
        self.btnExport.setText(_translate("MainWindow", langDict["导出"]))
        self.menu.setTitle(_translate("MainWindow", langDict["文件"]))
        self.menu_2.setTitle(_translate("MainWindow", langDict["调试"]))
        self.menu_3.setTitle(_translate("MainWindow", langDict["工具"]))
        self.actiontrain.setText(_translate("MainWindow", "train"))
        self.menuDbgGenMultiForCurrent.setText(_translate("MainWindow", langDict["只在当前图像上生成多框数据集"]))
        self.menuDbgGenMultiForCurrent.setShortcut(_translate("MainWindow", "Ctrl+D"))
        self.menuSpecifyImageNdx.setText(_translate("MainWindow", langDict["显示指定的图像..."]))
        self.menuSpecifyImageNdx.setShortcut(_translate("MainWindow", "Ctrl+N"))
        self.menuDelNonCheckedTags.setText(_translate("MainWindow", langDict["删除未选中的标签"]))
        self.menuAboxTool.setText(_translate("MainWindow", langDict["锚框工具"]))
        self.menuAboxTool.setShortcut(_translate("MainWindow", "Ctrl+H"))
        self.menuSaveConfig.setText(_translate("MainWindow", langDict["保存配置"]))
        self.menuSaveConfig.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.menuSaveConfigAs.setText(_translate("MainWindow", langDict["配置另存为..."]))
        self.menuSaveConfigAs.setShortcut(_translate("MainWindow", "Ctrl+A"))
        self.menuLoadConfig.setText(_translate("MainWindow", langDict["打开配置"]))
        self.menuLoadConfig.setShortcut(_translate("MainWindow", "Ctrl+L"))
        self.actionAuto_Load_last_Config.setText(_translate("MainWindow", langDict["自动打开上次配置"]))


class MyQMainWindow(QMainWindow):
    def __init__(self):
        super(MyQMainWindow, self).__init__()
    def closeEvent(self, closeEvent: QtGui.QCloseEvent):
        print('saving UI configuration')
        # 先使用默认自动保存文件保存配置
        mainLogic.SaveCfgDict()
        if mainLogic.uiCfgFile != '':
            mainLogic.SaveCfgDict(mainLogic.uiCfgFile)
        else:
            mainLogic.uiCfgFile = './uicfgs/_ui_cfg_auto.uicfg'

if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    MainWindow = MyQMainWindow()
    try:
        pass
        #app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    except:
        pass
    ui = MyMainUI()
    ui.setupUi(MainWindow)
    mainUI = ui
    MainWindow.show()
    mainLogic = MainAppLogic(ui, MainWindow)
    app.exec()
    # sys.exit(app.exec_())
