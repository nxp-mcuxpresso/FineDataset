#-*-coding:utf-8-*-
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
from cv2 import dct
import numpy as np

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from PyQt5 import QtCore, QtGui, QtWidgets
from pyqt5_ab_dlg_cfg_cluster import Ui_Dialog_Cfg_Cluster

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(718, 515)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(370, 470, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setGeometry(QtCore.QRect(10, 10, 691, 451))
        self.widget.setObjectName("widget")
        self.groupBox = QtWidgets.QGroupBox(self.widget)
        self.groupBox.setGeometry(QtCore.QRect(0, 0, 691, 451))
        self.groupBox.setObjectName("groupBox")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        # self.groupBox.setTitle(_translate("Dialog", "GroupBox_Matplotlib的图形显示："))


import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
# import app_main
#创建一个matplotlib图形绘制类
class MyFigure(FigureCanvas):
    def __init__(self,width=5, height=4, dpi=100):
        #第一步：创建一个Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        #第二步：在父类中激活Figure窗口
        super(MyFigure,self).__init__(self.fig) #此句必不可少，否则不能显示图形
        #第三步：创建一个子图，用于绘制图形用，111表示子图编号，如matlab的subplot(1,1,1)
        # self.axes = self.fig.add_subplot(111)
    #第四步：就是画图，【可以在此类中画，也可以在其它类中画】
    def plotsin(self):
        self.axes0 = self.fig.add_subplot(331)
        t = np.arange(0.0, 3.0, 0.01)
        s = np.sin(2 * np.pi * t)
        self.axes0.set_title('sin')
        self.axes0.plot(t, s)
    def plotcos(self):
        self.axes1 = self.fig.add_subplot(332)
        self.axes1.set_title('cos')
        t = np.arange(0.0, 3.0, 0.01)
        s = np.cos(2 * np.pi * t)
        self.axes1.plot(t, s)



class CDlgClusterCfg(Ui_Dialog_Cfg_Cluster):
    def __init__(self, hostWnd:QDialog, appObj):
        super(Ui_Dialog_Cfg_Cluster).__init__()
        self.hostWnd = hostWnd
        self.appObj = appObj
        self.setupUi(hostWnd)
        self.buttonBox.accepted.connect(lambda: print('hello accept'))
        self.fraGraph.setTitle('聚类图') 
        self.gridlayout = QGridLayout(self.fraGraph)
        self.sldMaxClusters.valueChanged.connect(lambda: self.OnValueChanged_sldMaxClusters())

        self.dctGeos = {}
        self.dctClsts = {}
        self.cache = {}
        self.prevFig = None

        #self.ScanDataset_WfUtils()
        #self.Clusterize()
        #self.Visualize()
        #self.OnValueChanged_sldMaxClusters()

    def ScanDataset_WfUtils(self):
        import json
        import os.path as path
        strDSFolder = self.appObj.strDSFolder 
    
        gtAspects, gtSizes, gtCenters = [], [], []
        with open("%s/bboxes.json" % (strDSFolder)) as fd:
            lst = json.load(fd)
        cnt = len(lst)
        
        # 获取图片输入尺寸
        strFile = path.join(strDSFolder, path.split(lst[0]['filename'])[1])

        for (i, item) in enumerate(lst):
            for xyxy in item['xyxys']:
                cx, cy = (xyxy[0] + xyxy[2]) // 2 , (xyxy[1] + xyxy[3]) // 2
                w,h = (xyxy[2] - xyxy[0]) , (xyxy[3] - xyxy[1])
                area = w * h
                r = h / w
                gtAspects.append([r, 1]) # 1只是占位，为了生成2维数据
                gtSizes.append([area, 1])
                gtCenters.append([cx, cy])
        
        self.dctGeos = {
            'centers' : np.array(gtCenters),
            'sizes' : np.array(gtSizes),
            'aspects' : np.array(gtAspects)
        }
        self.cache = {}

    def Clusterize(self):
        clusterCnt = self.sldMaxClusters.value()
        dctClsts = {}
        for sKey in self.dctGeos.keys():
            

            X = self.dctGeos[sKey]
        
            

            kmObj = KMeans(n_clusters=clusterCnt, random_state=9)
            y_pred = kmObj.fit_predict(X)

            # plt.scatter(X[:, 0], X[:, 1], c=y_pred)
            # plt.show()
            # 求取各个聚类的中心
            a = np.zeros([clusterCnt])
            b = a.copy()
            for i in range(len(X)):
                a[y_pred[i]] += X[i][0]
                b[y_pred[i]] += 1
            a /= b
            dctClsts[sKey] = {'clusters': clusterCnt, 'clusterIDs' : y_pred, 'centerLocs' : a}
        self.cache[clusterCnt] = dctClsts
        self.dctClsts = dctClsts

    def Visualize(self):
        fig = MyFigure(width=3, height=2, dpi=100)
        keys = ['centers', 'aspects']
        if self.prevFig is not None:
            self.gridlayout.removeWidget(self.prevFig)

        for (i, subPlotLoc) in enumerate([211, 212]):
            ax = fig.fig.add_subplot(subPlotLoc)
            y_pred = self.dctClsts[keys[i]]['clusterIDs']
            X = self.dctGeos[keys[i]]
            ax.set_title(keys[i])
            ax.scatter(X[:, 0], X[:, 1], c=y_pred, alpha=0.1, marker='.')
        self.gridlayout.addWidget(fig,0,1)
        self.prevFig = fig
        
    def OnValueChanged_sldMaxClusters(self):
        _translate = QtCore.QCoreApplication.translate
        self.lblClusters.setText(_translate('Dialog_Cfg_Cluster', '聚类数:%d' % (self.sldMaxClusters.value())))
        n = self.sldMaxClusters.value()
        if n not in self.cache.keys():
            self.Clusterize()

        self.dctClsts = self.cache[n]
        self.Visualize()

class MainDialogImgBW(QDialog,Ui_Dialog):
    def __init__(self):
        super(MainDialogImgBW,self).__init__()
        self.setupUi(self)
        self.setWindowTitle("显示matplotlib绘制图形")
        self.setMinimumSize(0,0)

        #第五步：定义MyFigure类的一个实例
        self.F = MyFigure(width=3, height=2, dpi=100)

        self.F.plotsin()
        self.F.plotcos()
        #第六步：在GUI的groupBox中创建一个布局，用于添加MyFigure类的实例（即图形）后其他部件。
        self.gridlayout = QGridLayout(self.groupBox)  # 继承容器groupBox
        self.gridlayout.addWidget(self.F,0,1)

        #补充：另创建一个实例绘图并显示
        #self.plotother()

    def plotother(self):
        F1 = MyFigure(width=5, height=4, dpi=100)
        F1.fig.suptitle("Figuer_4")
        F1.axes1 = F1.fig.add_subplot(221)
        x = np.arange(0, 50)
        y = np.random.rand(50)
        F1.axes1.hist(y, bins=50)
        F1.axes1.plot(x, y)
        F1.axes1.bar(x, y)
        F1.axes1.set_title("hist")
        F1.axes2 = F1.fig.add_subplot(222)

        ## 调用figure下面的add_subplot方法，类似于matplotlib.pyplot下面的subplot方法
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        y = [23, 21, 32, 13, 3, 132, 13, 3, 1]
        F1.axes2.plot(x, y)
        F1.axes2.set_title("line")
        # 散点图
        F1.axes3 = F1.fig.add_subplot(223)
        F1.axes3.scatter(np.random.rand(20), np.random.rand(20))
        F1.axes3.set_title("scatter")
        # 折线图
        F1.axes4 = F1.fig.add_subplot(224)
        x = np.arange(0, 5, 0.1)
        F1.axes4.plot(x, np.sin(x), x, np.cos(x))
        F1.axes4.set_title("sincos")
        self.gridlayout.addWidget(F1, 0, 2)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainDialogImgBW()
    main.show()
    #app.installEventFilter(main)
    sys.exit(app.exec_())