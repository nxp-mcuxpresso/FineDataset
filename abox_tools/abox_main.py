
from abc import abstractmethod
from typing import Dict
from PyQt5 import QtCore, QtGui, QtWidgets

from PyQt5.QtGui import QPixmap

from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QMessageBox, QStatusBar, QFileDialog, QInputDialog
from PyQt5.QtCore import QFile, Qt
import pyqt5_ab_main
import json
import os.path as path
import os, sys
import numpy as np
import cv2
from PIL import Image
import importlib
import glob

from box_utils import SSDBoxSizes
import gtb_clustering
import pyqt5_ab_dlg_plot
_translate = QtCore.QCoreApplication.translate
try:
    newPath = path.abspath('./')
    sys.path.append(newPath)
    import plugins_dsread.abstract_utils as abstract_utils
except:
    newPath = path.abspath('../')
    sys.path.append(newPath)
    import plugins_dsread.abstract_utils as abstract_utils
class DSUtils():
    def TestClassMethod():
        print('Class method!')
    
    def FromSubdataset(dctFiles):
        k2 = sorted(dctFiles.keys())
        lstRet = []
        for k in k2:
            xywhs = dctFiles[k]['xywhs']
            xyxys = [[x['x1'], x['y1'], x['x1']+x['w'], x['y1']+x['h']] for x in xywhs]
            lstRet.append({
                'filename' : k,
                'xyxys' : xyxys
            })
        return lstRet

    def ParseWiderFace(dsFolder, setSel='train', stBar:QStatusBar = None):
        pathBBox = '%s/wider_face_split/wider_face_%s_bbx_gt.txt' % (dsFolder, setSel)
        fd = open(pathBBox)
        lstLines = fd.readlines()
        fd.close()
        STATE_WANT_FILENAME = 0
        STATE_WANT_BBOX_CNT = 1
        STATE_WANT_BBOX_ITEM = 2
        st = STATE_WANT_FILENAME
        NDX_BLUR = 4
        NDX_INVALID = 7
        NDX_ATYPICALPOSE = 9
        NDX_OCCLUSION = 8
        bboxRem = 0
        lstBBxyxys = []
        dctFiles = {}
        print('scaning')
        if stBar is not None:
            stBar.showMessage('扫描数据集...')
        lnCnt = len(lstLines)
        for (ndx, strLine) in enumerate(lstLines):
            strLine = strLine.strip()
            if ndx % 100 == 0:
                stBar.showMessage('扫描数据集 %d/%d' % (ndx, lnCnt))
                QApplication.processEvents()
            ndx += 1
            if st == STATE_WANT_FILENAME:
                strFileName = './WIDER_%s/images/%s' % (setSel, strLine)
                st = STATE_WANT_BBOX_CNT
            elif st == STATE_WANT_BBOX_CNT:
                nBBoxCnt = int(strLine)
                st = STATE_WANT_BBOX_ITEM
                bboxRem = nBBoxCnt
                lstBBxyxys = []
            elif st == STATE_WANT_BBOX_ITEM:
                if nBBoxCnt > 0:
                    lstVals = [int(x) for x in strLine.split(' ')]
                    if lstVals[NDX_INVALID] == 0 and lstVals[NDX_BLUR] < 2 and lstVals[NDX_ATYPICALPOSE] == 0 and lstVals[NDX_OCCLUSION] < 1:
                        if lstVals[2] * lstVals[3] >= 36*36 and lstVals[2] != 0 and lstVals[3] / lstVals[2] < 2.0:
                            # 转换成xyxy形式并添加
                            lstBBxyxys.append( [lstVals[0], lstVals[1], lstVals[0] + lstVals[2], lstVals[1] + lstVals[3]])
                    bboxRem -= 1
                    if bboxRem == 0:
                        st = STATE_WANT_FILENAME
                        if len(lstBBxyxys) > 0:
                            dctFiles[strFileName] = {
                                'cnt0' : nBBoxCnt,
                                'cnt' : len(lstBBxyxys),
                                'xyxys' : lstBBxyxys
                            }
                else:
                    st = STATE_WANT_FILENAME
        k2 = sorted(dctFiles.keys())
        lstRet = []
        for k in k2:
            lstRet.append({
                'filename' : k,
                'xyxys' : dctFiles[k]['xyxys']
            })
        stBar.showMessage('数据集分析完毕')
        return lstRet
class MyMainUI(pyqt5_ab_main.Ui_MainWindow):
    def __init__(self, hostWnd):
        super(pyqt5_ab_main.Ui_MainWindow, self).__init__()
        self.dctPlugins = dict()
        self.hostWnd = hostWnd
        self.img = None
        self.setupUi(MainWindow)
        # 为了和qt designer无缝工作，先删除qt designer上的控件，再换成自己子类化的
        if self.lblMain is not None:
            self.lblMain.deleteLater()
            self.lblMain.hide()
            del self.lblMain
        self.lblMain = MyQLabel(self.centralwidget)
        self.lblMain.setGeometry(QtCore.QRect(10, 10, 960, 768))
        self.lblMain.setFrameShape(QtWidgets.QFrame.Box)
        self.lblMain.setObjectName("lblMain")
        self.lblMain.setText("lblMain")

        self.cmbSubSet.addItems(['train','val', 'any'])
        
        # 生成各个特征图上的锚框配置组框
        self.lstFraABs = [(FraAB(self.fraConsole, i)) for i in range(0, 8)]

        self.lstFraABs[2].fraAB.setChecked(True)
        self.lstABCfgs = []
        self.ccwhABs = []

        # 搜索 xxx_utils.py
        lstTypes = [x[2:-3] for x in glob.glob('./plugins_dsread/*_utils.py')]
        if len(lstTypes) == 0:
            lstTypes = [x[3:-3] for x in glob.glob('../plugins_dsread/*_utils.py')]
            #sys.path.append(os.path.abspath('../'))
        else:
            pass
            #sys.path.append(os.path.abspath('./'))
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
                self.cmbDSType.addItem(ds)
                self.dctPlugins[ds] = dsCls[i]
                pluginCnt += 1

        self.cmbDSType.setCurrentIndex(2)
        self.cmbIoUThd.addItems(['%d%%' % (x) for x in range(25, 85, 5)])
        self.cmbIoUThd.setCurrentIndex(5)
        self.iouThd = float(self.cmbIoUThd.currentText()[:-1]) / 100
        self.inWH = (640, 640)
        self.isFixedImgSize = True
        # 按下绘制部分锚框按钮时，绘制的锚框数
        self.absToDraw = 9

    
        self.pgsBar.setVisible(False)
        self.tmrToHidePgsBar = QtCore.QTimer(MainWindow)
        self.tmrToHidePgsBar.setInterval(300)
        self.tmrToHidePgsBar.setSingleShot(True)
        self.statusbar = QStatusBar()
        MainWindow.setStatusBar(self.statusbar)
        self.statusbar.addPermanentWidget(self.pgsBar)        
        self.tmrToHidePgsBar.timeout.connect(self.OnTimeout_tmrToHidePgsBar)

    def OnTimeout_tmrToHidePgsBar(self):
        self.pgsBar.setVisible(False)

    def CalcABMatching(self, inWH=(640,640), gts=[]):
        def CalcIoU_XYXY(r1, r2):
            x1O, y1O, x2O, y2O = min(r1[0], r2[0]), min(r1[1],r2[1]), max(r1[2], r2[2]), max([r1[3],r2[3]])
            wO = x2O - x1O
            hO = y2O - y1O
            x1I, y1I, x2I, y2I = max(r1[0], r2[0]), max(r1[1],r2[1]), min(r1[2], r2[2]), min([r1[3],r2[3]])
            wI = x2I - x1I
            hI = y2I - y1I
            if wI < 0 or hI < 0:
                return 0
            iou = (wI*hI) / (wO*hO)
            return iou        
        
        if self.inWH is None or self.inWH[0] != inWH[0] or self.inWH[1] != inWH[1]:
            self.inWH = inWH
            self.GetABConfig(True, False)
        
        xywhGTs = [[x[0], x[1], x[2] - x[0], x[3] - x[1]] for x in gts]
        
        dctMatched = {}
        for x in range(len(gts)):
            dctMatched[x] = []

        # 为每一个锚框检测匹配的gt
        ptNdx = 0
        for dct in self.ccwhABs:
            if dct['enabled'] == False:
                continue
            for lstABsAtOnePix in dct['aboxes']:
                if ptNdx == 40:
                    ptNdx = ptNdx                
                for (abNdx, ccwh) in enumerate(lstABsAtOnePix):
                    if len(ccwh) > 4:
                        ccwh = ccwh[:4]  # 删除上一次计算匹配关系时可能残留的真实框匹配关系
                        lstABsAtOnePix[abNdx] = ccwh
                    ab_xyxy = [ccwh[0] - ccwh[2] // 2, ccwh[1] - ccwh[3] // 2, ccwh[0] + ccwh[2] // 2, ccwh[1] + ccwh[3] // 2]
                    iouMax, maxAt = 0, -1
                    for (i, gt) in enumerate(gts):
                        iou = CalcIoU_XYXY(ab_xyxy, gt)
                        if iou > iouMax:
                            iouMax, maxAt = iou, i
                    if iouMax >= self.iouThd:
                        ccwh = ccwh[:4] + [gt, ab_xyxy, dct]
                        lstABsAtOnePix[abNdx] = ccwh
                        dctMatched[maxAt].append(ccwh)
                ptNdx += 1
        #'dctMatched的key就是GT的编号, value是[abx1, aby1, abw, abh, [gt, ab_xyxy, 当前尺度的锚框配置字典]'
        return dctMatched

    def GetABConfig(self, isMakeABs=True, isVisualize=True) -> list:
        lstRet = []
        lstOriCfg = []
        try:
            for (i,item) in enumerate(self.lstFraABs):
                dct = {
                    'ndx': i,
                    'enabled' : item.fraAB.isChecked(),
                    'aspects' : item.edtWVsH.text().split(','), # 长宽比
                    'sizes' : item.edtSize.text().split(','),
                    'dnsmp' : 2**(i + 2),
                    'flavor' : item.cmbFlavor.currentText()
                }
                lstOriCfg.append(dct)
                # 生成aspect ratios的倒数，并计算分数形式给出的结果
                def _GetABConfig_Xlate(dct, sKey, isPair=False):
                    lstTmp = []
                    for sItem in dct[sKey]:
                        sItem = sItem.strip()
                        if '/' in sItem:
                            if '/' in sItem:
                                lstVals = [float(x.strip()) for x in sItem.split('/')]
                            vOut = lstVals[0] / lstVals[1]
                        else:
                            if sItem.lower() == 'fib':
                                vOut = 1.618
                            else:
                                vOut = float(sItem)
                        if vOut < 0.1:
                            vOut = 0.1
                        lstTmp.append(vOut)
                        if isPair == True and vOut != 1:
                            lstTmp.append(1 / vOut)
                    dct[sKey] = lstTmp 
                # 计算sizes
                _GetABConfig_Xlate(dct, 'aspects', isPair=False)
                _GetABConfig_Xlate(dct, 'sizes')
                lstRet.append(dct)
                self.lstABCfgs = lstRet
        except:
            self.statusbar.showMessage('锚框配置不合法！')
            return -1
        if isMakeABs:
            self.ccwhABs = self.MakeABs(lstRet, self.inWH)
            if mainLogic.rndNdx >= 0:
                self.CalcABMatching(self.inWH, mainLogic.lstGTs[mainLogic.rndNdx]['xyxys'])
        if isVisualize:
            self.VisualizeABs(self.ccwhABs, self.inWH)
        uiMain.statusbar.showMessage('已贯彻新的锚框配置！', 5000)
        return 0

    @classmethod
    def MakeABs_OneFMap(cls, dctCfg={}, lstAcc=[], inWH=(300,300), isClamp = True):
        lstNew = []
        dnsmp = dctCfg['dnsmp']
        # 计算特征图的大小。遇到奇数大小时丢弃余数
        lstFmapSize=[]
        for tmp in inWH:
            div=1
            while(div < dnsmp):
                tmp >>= 1
                div *= 2
            lstFmapSize.append(tmp)

        lstABSizes = dctCfg['sizes']
        lstABAspects = dctCfg['aspects']
        # 生成eIQ风格的锚框：使用2种锚框尺寸，且第2种尺寸只生成正方形的框
        priors = []                
        # 使用2级循环遍历特征图上的所有位置
        for row in range(lstFmapSize[1]):
            for col in range(lstFmapSize[0]):
                ccwhsAtPix = []
                # 求出当前特征图像素上锚框的中心坐标，相对于特征图
                cy = (row + 0.5)
                cx = (col + 0.5)
                # 先为每个长宽比生成小尺寸的一组锚框
                for r in lstABAspects:
                    w = lstABSizes[0] * r**0.5
                    h = lstABSizes[0] / r**0.5
                    ccwhsAtPix.append([cx, cy, w, h])

                if len(lstABSizes) > 1:
                    if dctCfg['flavor'].lower() == 'eIQ'.lower():
                        # 再生成大尺寸的正方形框
                        if 1.0 in lstABAspects:
                            ccwhsAtPix.append([cx, cy, lstABSizes[1], lstABSizes[1]])
                    elif dctCfg['flavor'].lower() == 'd2l book'.lower():
                        # 为每个尺寸生成第1种长宽比的锚框
                        r = lstABAspects[0]
                        for sz in lstABSizes[1:]:
                            w = sz * r**0.5
                            h = sz / r**0.5
                            ccwhsAtPix.append([cx, cy, w, h])
                # 转换成真实坐标
                for (i, ccwh) in enumerate(ccwhsAtPix):
                    # 0=cx, 1=cy, 2=w, 3=h
                    ccwh[0] *= inWH[0] / lstFmapSize[0]                         
                    ccwh[2] *= inWH[0] / lstFmapSize[0]
                    ccwh[1] *= inWH[1] / lstFmapSize[1]
                    ccwh[3] *= inWH[1] / lstFmapSize[1]
                    # 把超出图像边界的锚框调整成图像边界以内
                    if isClamp:
                        if ccwh[0] + ccwh[2] / 2 >= inWH[0]:
                            ccwh[2] = (inWH[0] - ccwh[0]) * 2
                        if ccwh[1] + ccwh[3] / 2 >= inWH[1]:
                            ccwh[3] = (inWH[1] - ccwh[1]) * 2
                        if ccwh[0] - ccwh[2] / 2 < 0:
                            ccwh[2] = ccwh[0] * 2
                        if ccwh[1] - ccwh[3] / 2 < 0:
                            ccwh[3] = ccwh[1] * 2
                    ccwhsAtPix[i] = [int(x) for x in ccwh]

                # 添加结果
                lstNew.append(ccwhsAtPix)
        return lstAcc + lstNew

    @classmethod
    def MakeABs(cls, lstCfg, inWH=(640,640)):
        dctABs = {}
        lstRet = []
        for cfg in lstCfg:
            dctABs = cfg.copy()
            if cfg['enabled'] == False:
                lstRet.append(dctABs)
                continue
            ccwhABs = cls.MakeABs_OneFMap(cfg,inWH=inWH)
            
            dctABs['aboxes'] = ccwhABs
            lstRet.append(dctABs)
        
        return lstRet
    def VisualizeABs(self, ccwhABs, inWH=(256,192)):
        if self.img is None:
            vis = np.zeros((inWH[0], inWH[1],3), np.uint8)
            img = cv2.cvtColor(vis,cv2.COLOR_BGR2RGB)
        else:
            img = self.img.copy()

        ptNdx = 0
        cols = [(0,0,255), (0,255,255), (0,255,0), (255,255,0), (255,127,127), (255,127,0), (255,127,127), (255,127,255)]
        drks = [(127,127,191), (127,191,191), (127,191,127), (191,191,127), (191,159,159), (191,159,127), (191,159,159), (191,159,191)]
        # 统计锚框数
        cnt = 0
        absNdx = self.absToDraw
        for (i, dct) in enumerate(ccwhABs):
            if 'aboxes' in dct.keys():
                cnt += len(dct['aboxes'])

        for (i, dct) in enumerate(ccwhABs):
            if 'aboxes' not in dct.keys():
                continue

            for pts in dct['aboxes']:
                if ptNdx == absNdx: #% modulo == (modulo // 2):
                    for (colNdx, abox) in enumerate(pts):
                        pt1 = (int(abox[0] - abox[2] / 2), int(abox[1] - abox[3] / 2))
                        pt2 = (int(pt1[0] + abox[2]), int(pt1[1] + abox[3]))
                        width = 1 if inWH[0] < 480 else 2
                        if len(abox) > 4:
                            # 有匹配的真实框
                            cv2.rectangle(img, pt1, pt2, cols[colNdx], width, cv2.LINE_8)
                        else:
                            cv2.rectangle(img, pt1, pt2, drks[colNdx], width, cv2.LINE_8)
                ptNdx += 1
        table = np.array(img)
        qImg = QtGui.QImage(bytearray(table), inWH[0], inWH[1], inWH[0]*3, QtGui.QImage.Format_BGR888)
        pix = QPixmap(QPixmap.fromImage(qImg))        
        #pix2 = pix.scaled(32,32, Qt.KeepAspectRatio)#, Qt.SmoothTransformation)     
        # ui.imgWnd.setPixmap(pix2)
        uiMain.lblMain.geometry()
        rect = uiMain.lblMain.rect()
        pix3 = pix.scaled(rect.width(),rect.height(), Qt.KeepAspectRatio)#, Qt.SmoothTransformation)
        uiMain.lblMain.setPixmap(pix3)
        return img
    def OnChanged_SldABsToDraw(self):
        _translate = QtCore.QCoreApplication.translate
        cnt = self.sldABsToDraw.value()
        self.btnDrawABs.setText(_translate("MainWindow", "绘制第%d个锚框" % (cnt)))
        self.absToDraw = cnt
        self.VisualizeABs(self.ccwhABs, self.inWH)
class MainAppLogic():
    def __init__(self, ui:MyMainUI, mainWindow):
        self.ui = ui
        self.strDSFolder = ''
        self.strNextDsFolder = ''
        self.strABSettingsPath = ''
        self.mainWnd = mainWindow
        self.rndNdx = -1
        self.gtCenters = []
        self.gtSizes = []
        self.gtAspects = []
        self.appName = '锚框设计与验证工具'
        self.lstGTs = None
        self.provider:abstract_utils.AbstractUtils = None
        # 连接信号和槽

        ui.btnDSFolder.clicked.connect(self.OnClick_btnDSFolder)          
        ui.btnDrawABs.clicked.connect(self.ui.OnChanged_SldABsToDraw)
        ui.btnLoadDataset.clicked.connect(lambda: self._UpdateDatasetFolder(self.strNextDsFolder))
        ui.sldABsToDraw.valueChanged.connect(self.ui.OnChanged_SldABsToDraw)
        ui.menu_ClusterCfg.triggered.connect(lambda: self.ShowClusterDlg())
        ui.btnCluster.clicked.connect(lambda: self.ShowClusterDlg())
        ui.cmbIoUThd.currentTextChanged.connect(lambda: self.OnIoUThdChanged())
        ui.menuLoadABSettings.triggered.connect(lambda: self.OnTriggered_MenuLoadBSettings())
        ui.menuSaveABSettings.triggered.connect(lambda: self.OnTriggered_MenuSaveBSettings())
        ui.menuSaveABSettingsAs.triggered.connect(lambda: self.OnTriggered_MenuSaveABSettingsAs())
        ui.menuDSFolder.triggered.connect(self.OnClick_btnDSFolder)
        ui.menuStatAboxMatching.triggered.connect(lambda: self.StatAboxMatching())
        y = lambda: self.ShowRandomValidate(self.strDSFolder)
        ui.btnValidateRandom.clicked.connect(y)
        ui.menuShowRandom.triggered.connect(y)
        ui.menuShowAt.triggered.connect(self.OnMenuTriggered_SpecifyImageNdx)
        ui.btnValidate.clicked.connect(lambda: self.ShowValidate(self.rndNdx, self.strDSFolder))
        ui.btnApplyABCfg.clicked.connect(lambda: self.ui.GetABConfig(True, False))
        # self.ui.OnChanged_SldABsToDraw()
        
        # 为每个锚框设计组框添加变更的事件响应
        for fra in uiMain.lstFraABs:
            fra.edtSize.editingFinished.connect(self.OnAbCfgChanged)
            fra.edtWVsH.editingFinished.connect(self.OnAbCfgChanged)
            fra.cmbFlavor.currentTextChanged.connect(self.OnAbCfgChanged)
            fra.fraAB.clicked.connect(self.OnAbCfgChanged)

        self.tmrChkAbCfg = QtCore.QTimer(self.ui.hostWnd)
        self.tmrChkAbCfg.setInterval(100)
        self.tmrChkAbCfg.setSingleShot(True)
        self.tmrChkAbCfg.timeout.connect(self.ChkAbCfgUpdt)

        self.ui.btnDrawABs.setText(_translate("MainWindow", "绘制第%d个锚框" % (ui.sldABsToDraw.value())))

        self.strDSType = ''
        self._UpdateDatasetFolder( 'q:/gitrepos/subdataset/outs/out_train_multi')
        self.ShowValidate(0, isShow=False)
    def OnMenuTriggered_SpecifyImageNdx(self):
        ndx, isOK = QInputDialog.getInt(MainWindow, '设置当前图片索引', '请输入索引：', min = 0) 
        if isOK:
            try:
                self.ShowValidate(ndx, self.strDSFolder)
                self.rndNdx = ndx
            except:
                self.ui.statusBar.showMessage('无效的图片号%d' % (ndx))
    def ShowClusterDlg(self):
        ui_dlgCfgCluster.ScanDataset_WfUtils()
        ui_dlgCfgCluster.OnValueChanged_sldMaxClusters()
        dlgCfgCluster.show()
    
    def _doSaveABSettings(self):
        pass

    def _UpdateDatasetFolder(self, dir_choose):
        def callback(pgs, msg=''):
            self.ui.pgsBar.setValue(pgs)
            if len(msg) > 0:
                self.ui.statusbar.showMessage(msg)
            QApplication.processEvents()
        
        self.ui.pgsBar.setValue(1)
        self.ui.pgsBar.setVisible(True)

        lstHvsW_config = [0.1, 10.0] # min, max
        lstGTPerImgs = [1, 50]
        dctCfg = {
            'minHvsW' : lstHvsW_config[0],
            'maxHvsW' : lstHvsW_config[1],
            'minGTPerImg': lstGTPerImgs[0],
            'maxGTPerImg': lstGTPerImgs[1]
        }
        provider:abstract_utils = None
        dsType = uiMain.cmbDSType.currentText()
        uiMain.statusbar.showMessage('数据集读取中...')
        QApplication.processEvents()
        try:
            setSel = 'train' #uiMain.cmb
            provider = uiMain.dctPlugins[dsType](dir_choose, setSel, dctCfg, callback)

        except Exception as e:
            print(e)
            self.ui.statusbar.showMessage('代码错误：\n' + str(e))
        self.strNextDsFolder = dir_choose
        if provider is None or provider.dctFiles is None or provider.dctFiles == {}:
            self.ui.statusbar.showMessage('%s中的数据集无法按%s解析！' % (dir_choose, dsType) )            
        else:
            self.dsFolder = dir_choose
            self.provider = provider
            MainWindow.setWindowTitle(_translate("MainWindow", "锚框辅助设计与验证工具 - 已加载%s类型的数据集于：%s" % (uiMain.cmbDSType.currentText(), self.dsFolder)))            
            self.ui.statusbar.showMessage('数据集%s (%s)读取成功!' % (dir_choose, dsType))
            strDsType = uiMain.cmbDSType.currentText()
            self.strDSFolder = dir_choose
            self.strDSType = strDsType
            self.rndNdx = 0
            uiMain.isFixedImgSize = provider.IsFixedSizeImg()
            self.lstGTs = DSUtils.FromSubdataset(provider.dctFiles)

        self.ui.pgsBar.setValue(100)
        self.ui.tmrToHidePgsBar.start()
    def OnIoUThdChanged(self):
        uiMain.iouThd = float(uiMain.cmbIoUThd.currentText()[:-1]) / 100
        self.ShowValidate(self.rndNdx, self.strDSFolder)

    def OnTriggered_MenuSaveABSettingsAs(self, strPath = ''):
        if len(strPath) < 1:
            lstPaths = QFileDialog.getSaveFileName(self.mainWnd, '保存锚框配置', './', 'JSON(*.json)')
        else:
            lstPaths = [strPath, 'JSON(*.json)']
        if len(lstPaths) == 2 and len(lstPaths[0]) > 0:
            lstOut = []
            for (i,item) in enumerate(self.ui.lstFraABs):
                dct = {
                    'ndx': i,
                    'enabled' : item.fraAB.isChecked(),
                    'aspects' : item.edtWVsH.text(), # 长宽比
                    'sizes' : item.edtSize.text(),
                    'dnsmp' : str(2**(i + 2)),
                    'flavor' : item.cmbFlavor.currentText()
                }
                lstOut.append(dct)

            with open(lstPaths[0], 'w', encoding='utf-8') as fd:
                json.dump(lstOut, fd, indent=4)

    def OnTriggered_MenuSaveBSettings(self):
        self.OnTriggered_MenuSaveABSettingsAs(self.strABSettingsPath)

    def OnTriggered_MenuLoadBSettings(self):
        lstPaths = QFileDialog.getOpenFileName(self.mainWnd, '读取锚框配置', './', 'JSON(*.json)')
        if len(lstPaths) > 1:
            if path.exists(lstPaths[0]):
                with open(lstPaths[0]) as fd:
                    lstCfg = json.load(fd)
                for (i, item) in enumerate(lstCfg):
                    self.ui.lstFraABs[i].ApplyCfg(item)
                self.strABSettingsPath = lstPaths[0]
    
    def OnClick_btnDSFolder(self):
        dir_choose = QFileDialog.getExistingDirectory(self.mainWnd,  
                                    "选取文件夹",  
                                    './') # 起始路径

        if dir_choose == "":
            print("\n取消选择")
            return

        print("\n你选择的文件夹为:")
        print(dir_choose)
        self._UpdateDatasetFolder(dir_choose)

    def GetGTGemometries(self, strDSFolder = ''):
        if self.lstGTs is None:
            uiMain.statusbar.showMessage('请先选择数据集文件夹')
            return -1
        lst = self.lstGTs
        with open("%s/bboxes.json" % (strDSFolder)) as fd:
            lst = json.load(fd)
        cnt = len(lst)
        
        # 获取图片输入尺寸
        strFile = self.provider.MapFile(lst[0]['filename'])
        # strFile = path.join(strDSFolder, path.split(lst[0]['filename'])[1])

        for (i, item) in enumerate(lst):
            for xyxy in item['bboxes']:
                cx, cy = (xyxy[0] + xyxy[2]) // 2 , (xyxy[1] + xyxy[3]) // 2
                w,h = (xyxy[2] - xyxy[0]) , (xyxy[3] - xyxy[1])
                area = w * h
                r = w / h
                self.gtAspects.append(r)
                self.gtSizes.append(area)
                self.gtCenters.append([cx, cy])
                    
    def ChkAbCfgUpdt(self):
        uiMain.GetABConfig(True, False)
    def OnAbCfgChanged(self):
        self.tmrChkAbCfg.start()
    def ShowValidate(self, ndx, strDSFolder = '', isShow = True):
        if ndx < 0:
            return None
        if strDSFolder == '':
            strDSFolder = self.strDSFolder
        img = None
        if self.lstGTs is None:
            uiMain.statusbar.showMessage('请先选择数据集文件夹')
            return -1
        lst = self.lstGTs
        item = lst[ndx]

        strFile = self.provider.MapFile(item['filename'])

        # strFile = strFile.replace('/', path.sep)
        image = Image.open(strFile)
        imgWH = (image.width, image.height)
        if uiMain.inWH[0] != imgWH[0] or uiMain.inWH[1] != imgWH[1]:
            uiMain.inWH = imgWH
            uiMain.GetABConfig(True, False)
        img = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
        image.close()
        width = 1 if imgWH[0] < 480 else 2
        for bbox in item['xyxys']:
            pt1 = (bbox[0], bbox[1])
            pt2 = (bbox[2], bbox[3])
            col = (255, 255, 255)
            cv2.rectangle(img, pt1, pt2, col, width, 4)
        dctMatched = self.ui.CalcABMatching(inWH=imgWH, gts = item['xyxys'])
        self.ui.img = img.copy()
        width = 1 if imgWH[0] < 480 else 2
        for i in range(len(item['xyxys'])):
            # 为每个真实框绘制匹配的锚框
            for (n, ccwh) in enumerate(dctMatched[i]):
                pt1 = (ccwh[0] - ccwh[2] // 2, ccwh[1] - ccwh[3] // 2)
                pt2 = (ccwh[0] + ccwh[2] // 2, ccwh[1] + ccwh[3] // 2)
                if ccwh[3] / ccwh[2] > 1.8:
                    r = 0
                r = 63 + (n % 3) * 64
                g = 63 + ((n//3) % 3) * 64
                b = 63 + (n//9) * 64
                cv2.rectangle(img, pt1, pt2, (b,g,r), width, 4)         
        if not img is None:
            if isShow:
                table = np.array(img)
                qImg = QtGui.QImage(bytearray(table), imgWH[0], imgWH[1], imgWH[0]*3, QtGui.QImage.Format_BGR888)
                pix = QPixmap(QPixmap.fromImage(qImg))
                
                #pix2 = pix.scaled(32,32, Qt.KeepAspectRatio)#, Qt.SmoothTransformation)
                rect = self.ui.lblMain.rect()
                pix3 = pix.scaled(rect.width(),rect.height(), Qt.KeepAspectRatio)#, Qt.SmoothTransformation)
                # pix3 = pix.scaled(128,128, Qt.KeepAspectRatio)#, Qt.SmoothTransformation)
                
                # ui.imgWnd.setPixmap(pix2)
                self.ui.lblMain.setPixmap(pix3)
        self.ui.statusbar.showMessage('显示图片%d' % (ndx))
        return np.array(img)

    def ShowRandomValidate(self, strDSFolder = '', isShow = True):
        if self.lstGTs is None:
            uiMain.statusbar.showMessage('请先选择数据集文件夹')
            return -1
        lst = self.lstGTs
        cnt = len(lst)
        ndx = np.random.randint(cnt)
        self.rndNdx = ndx
        return self.ShowValidate(ndx, strDSFolder, isShow)


    def StatAboxMatching(self, strDSFolder = ''):
        # 统计锚框匹配情况
        strDSFolder = self.strDSFolder if strDSFolder == '' else strDSFolder
        if self.lstGTs is None:
            uiMain.statusbar.showMessage('请先选择数据集文件夹')
            return -1
        lstLabeledImages = self.lstGTs

        missCnt = 0
        # dctMatchSmry: key是匹配锚框的数量, value是字典
        dctMatchSmry = {}

        lstMatchesPerFmap = []
        for (i, fraABs) in enumerate(uiMain.ccwhABs):
            if fraABs['enabled'] == False:
                lstMatchesPerFmap.append([0,0])
            else:
                lstMatchesPerFmap.append([0, len(fraABs['aboxes']) * len(fraABs['aboxes'][0])])

        for i in range(16):
            dctMatchSmry[i] = []
        imgCnt = len(lstLabeledImages)
        lstMatchSizes = []
        lstMatchAspects = []
        for (i, labeledImg) in enumerate(lstLabeledImages):
            if uiMain.isFixedImgSize == False:
                if 'imgWH' not in labeledImg.keys():
                    if self.strDSType == 'subdataset':
                        filePart =  path.split(labeledImg['filename'])[1]
                    else:
                        filePart = labeledImg['filename']                    
                    strFile = path.join(strDSFolder, filePart)
                    # strFile = strFile.replace('/', path.sep)
                    image = Image.open(strFile)
                    imgWH = (image.width, image.height)
                    labeledImg['imgWH'] = imgWH
                else:
                    imgWH = labeledImg['imgWH']
            else:
                imgWH = uiMain.inWH       
            if i % 10 == 0:
                uiMain.statusbar.showMessage('已完成%d/%d' % (i+1, imgCnt), 3000)
                QApplication.processEvents()
            dctMatched = self.ui.CalcABMatching(inWH= imgWH, gts = labeledImg['xyxys'])
            gtCnt = len(labeledImg['xyxys'])            
            lstCurMissed = [labeledImg]
            
            for gtNdx in range(gtCnt):
                # 为当前图片的每个gt判断匹配情况
                gtMatABCnt = len(dctMatched[gtNdx])
                lstABOut = []
                if gtMatABCnt == 0:
                    missCnt += 1
                    lstCurMissed.append(gtNdx)
                else:                    
                    for ccwh in dctMatched[gtNdx]:
                        lstABOut.append([ccwh[:6], ccwh[6]['ndx']])
                        dctAB = ccwh[6] # cx,cy,w,h,[gt], [xyxy], dctABsAtOneFmap
                        lstTmp = lstMatchesPerFmap[dctAB['ndx']]
                        lstTmp[0] += 1
                    matchInfo = dctMatched[gtNdx]
                xyxy = labeledImg['xyxys'][gtNdx]
                ret = [gtMatABCnt,labeledImg['filename'], xyxy, lstABOut]#, dctMatched[gtNdx]]
                tmp = gtMatABCnt if (gtMatABCnt < len(dctMatchSmry) - 1) else len(dctMatchSmry) - 1
                area = ((xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])) ** 0.5
                aspect = (xyxy[2] - xyxy[0]) / (xyxy[3] - xyxy[1])
                #if aspect < 1:
                #    aspect = 1 / aspect
                lstMatchAspects.append((tmp, aspect))
                lstMatchSizes.append((tmp, area))
                dctMatchSmry[tmp].append(ret)


        # 统计每个尺度上锚框的匹配率
        lstFMapMatchRates = []
        for tmp in lstMatchesPerFmap:
            if tmp[1] == 0:
                lstFMapMatchRates.append(0)
            else:
                lstFMapMatchRates.append(tmp[0] / tmp[1])

        self.dctMatchResult = {
            'missed' : missCnt,
            'match_cnt' : [len(x) for x in dctMatchSmry.values()],
            'fmap_match_rates' : lstFMapMatchRates,
            'match_dist_size' : lstMatchSizes,
            'match_dist_aspect' : lstMatchAspects,
            'details': dctMatchSmry
        }

        with open('match_stat.json', 'w') as fd:
            json.dump(self.dctMatchResult, fd, indent=4)

        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        
        zhfont1 = fm.FontProperties(fname='C:\Windows\Fonts\Dengb.ttf', size=14)
        _, axes = plt.subplots(2,2, figsize=(12, 12))
        #ax = plt.subplot(2,2,1)
        ax = axes[0][0]
        ax.bar(range(len(self.dctMatchResult['match_cnt'])), self.dctMatchResult['match_cnt'])
        ax.set_title('真实框匹配到的锚框数据分布图',fontproperties=zhfont1)
        ax.set_xlabel('匹配到的锚框数',fontproperties=zhfont1)
        #ax.set_xticklabels(['%d' % (x) for x in range(len(self.dctMatchResult['match_cnt']))])
        ax.set_ylabel('真实框数',fontproperties=zhfont1)
        
        ax = axes[0][1]
        ax.bar(range(len(lstFMapMatchRates)),lstFMapMatchRates)
        ax.set_title('各特征图的锚框匹配率',fontproperties=zhfont1)
        ax.set_xticklabels(['0', '8','16','32','64','128','256','512','1024'])
        ax.set_xlabel('特征图下采样率',fontproperties=zhfont1)
        ax.set_ylabel('锚框匹配率',fontproperties=zhfont1)

        # 分析真实框面积平方根的匹配情况
        ax = axes[1][0]
        X = np.array(lstMatchSizes)
        ax.scatter(X[:, 0], X[:, 1], alpha=0.2, marker='.')
        ax.set_title('不同面积真实框的匹配情况',fontproperties=zhfont1)
        # ax.set_xticklabels(['8,16,32,64,128,256,512,1024'])
        #ax.set_xticklabels(['%d' % (x) for x in range(len(self.dctMatchResult['match_cnt']))])
        ax.set_xlabel('匹配到的锚框数',fontproperties=zhfont1)
        ax.set_ylabel('真实框面积平方根',fontproperties=zhfont1)        

        # 分析真实框长宽比平方根的匹配情况
        ax = axes[1][1]
        X = np.array(lstMatchAspects)
        ax.scatter(X[:, 0], X[:, 1], alpha=0.2, marker='.')
        ax.set_title('不同长宽比真实框的匹配情况',fontproperties=zhfont1)
        # ax.set_xticklabels(['8,16,32,64,128,256,512,1024'])
        #ax.set_xticklabels(['%d' % (x) for x in range(len(self.dctMatchResult['match_cnt']))])
        ax.set_xlabel('匹配到的锚框数',fontproperties=zhfont1)
        ax.set_ylabel('真实框长宽比',fontproperties=zhfont1)        

        plt.show()
        plt.savefig('match_stat.png')
        return dctMatchSmry

class FraAB():
    def __init__(self, inWhich:QtWidgets.QGroupBox, ndx=0, w=180, h=135):
        '''
            生成一个配置锚框的frame
        '''
        self.fraAB = QtWidgets.QGroupBox(inWhich)
        self.fraAB.setGeometry(QtCore.QRect(20 + (ndx % 2) * (w+12), 30 + (ndx // 2) * (h+12), w, h))
        self.fraAB.setFlat(False)
        self.fraAB.setCheckable(True)
        self.fraAB.setChecked(False)
        self.fraAB.setObjectName("fraAB%d" % (ndx))
        self.label_9 = QtWidgets.QLabel(self.fraAB)
        self.label_9.setGeometry(QtCore.QRect(20, 20, w-30, 16))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.fraAB)
        self.label_10.setGeometry(QtCore.QRect(20, 40, 21, 16))
        self.label_10.setText("")
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.fraAB)
        self.label_11.setGeometry(QtCore.QRect(20, 65, w-30, 16))
        self.label_11.setObjectName("label_11")
        self.edtWVsH = QtWidgets.QLineEdit(self.fraAB)
        self.edtWVsH.setGeometry(QtCore.QRect(20, 40, w-30, 20))
        self.edtWVsH.setObjectName("edtWVsH")
        self.edtSize = QtWidgets.QLineEdit(self.fraAB)
        self.edtSize.setGeometry(QtCore.QRect(20, 85, w-30, 20))
        self.edtSize.setObjectName("edtSize")
        

        self.label_12 = QtWidgets.QLabel(self.fraAB)
        self.label_12.setGeometry(QtCore.QRect(20, 110, 70, 16))
        self.label_12.setObjectName("label_12")        
        self.cmbFlavor = QtWidgets.QComboBox(self.fraAB)
        self.cmbFlavor.setGeometry(QtCore.QRect(80, 110, w-90, 18))
        self.cmbFlavor.addItems(['eIQ', 'D2L book'])
        _translate = QtCore.QCoreApplication.translate
        self.fraAB.setTitle(_translate("MainWindow", "下采样/%d" % (2**(ndx + 2))))
        self.label_9.setText(_translate("MainWindow", "宽/高, 逗号隔开"))
        self.label_11.setText(_translate("MainWindow", "面积平方根(特征图像素)"))
        self.label_12.setText(_translate("MainWindow", "生成方式"))
        self.edtWVsH.setText(_translate("MainWindow", "1, fib"))
        lstCfg = ['5, 7', '4, 6', '3, 5', '2, 3', '1.5,2', '1, 1.3', '1, 1.3', '1, 1.3']
        self.edtSize.setText(_translate("MainWindow", lstCfg[ndx]))

    def ApplyCfg(self, item):
        self.fraAB.setChecked(item['enabled'])
        self.edtWVsH.setText(item['aspects'])
        self.edtSize.setText(item['sizes'])
        self.cmbFlavor.setCurrentText(item['flavor'])

class MyQLabel(QtWidgets.QLabel):
    myLabelClick = QtCore.pyqtSignal(str)
    def __init__(self, parent):
        super(MyQLabel, self).__init__(parent)
    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        ctx = self.objectName()
        self.myLabelClick.emit(ctx)

if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    # tests
    DSUtils.TestClassMethod()

    app = QApplication(sys.argv)
    # 创建窗口控件
    MainWindow = QMainWindow()
    dlgCfgCluster = QDialog()
    # 创建窗口控件中的对象
    uiMain = MyMainUI(MainWindow)    
    MainWindow.show()
    mainLogic = MainAppLogic(uiMain, MainWindow)
    ui_dlgCfgCluster = pyqt5_ab_dlg_plot.CDlgClusterCfg(dlgCfgCluster, mainLogic)
    
    
    # app.exec()
    sys.exit(app.exec_())
