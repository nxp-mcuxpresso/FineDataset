# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'widertools.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1376, 891)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.lblImg = QtWidgets.QLabel(self.centralwidget)
        self.lblImg.setGeometry(QtCore.QRect(10, 20, 1171, 821))
        self.lblImg.setFrameShape(QtWidgets.QFrame.Box)
        self.lblImg.setObjectName("lblImg")
        self.btnRandom = QtWidgets.QPushButton(self.centralwidget)
        self.btnRandom.setGeometry(QtCore.QRect(1190, 730, 75, 23))
        self.btnRandom.setObjectName("btnRandom")
        self.cmbMain = QtWidgets.QComboBox(self.centralwidget)
        self.cmbMain.setGeometry(QtCore.QRect(1200, 40, 111, 21))
        self.cmbMain.setObjectName("cmbMain")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(1200, 20, 71, 16))
        self.label.setObjectName("label")
        self.btnSplitSingle = QtWidgets.QPushButton(self.centralwidget)
        self.btnSplitSingle.setGeometry(QtCore.QRect(1190, 760, 111, 23))
        self.btnSplitSingle.setObjectName("btnSplitSingle")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(1200, 80, 61, 16))
        self.label_3.setObjectName("label_3")
        self.txtOutX = QtWidgets.QLineEdit(self.centralwidget)
        self.txtOutX.setGeometry(QtCore.QRect(1280, 80, 51, 20))
        self.txtOutX.setObjectName("txtOutX")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(1200, 110, 61, 16))
        self.label_4.setObjectName("label_4")
        self.txtOutY = QtWidgets.QLineEdit(self.centralwidget)
        self.txtOutY.setGeometry(QtCore.QRect(1280, 110, 51, 20))
        self.txtOutY.setObjectName("txtOutY")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(1200, 140, 71, 16))
        self.label_5.setObjectName("label_5")
        self.txtDatasetSize = QtWidgets.QLineEdit(self.centralwidget)
        self.txtDatasetSize.setGeometry(QtCore.QRect(1280, 140, 51, 20))
        self.txtDatasetSize.setObjectName("txtDatasetSize")
        self.btnGenSingleFaceDataSet = QtWidgets.QPushButton(self.centralwidget)
        self.btnGenSingleFaceDataSet.setGeometry(QtCore.QRect(1190, 170, 131, 31))
        self.btnGenSingleFaceDataSet.setObjectName("btnGenSingleFaceDataSet")
        self.btnValidateSingleFaceDataSet = QtWidgets.QPushButton(self.centralwidget)
        self.btnValidateSingleFaceDataSet.setGeometry(QtCore.QRect(1190, 210, 131, 31))
        self.btnValidateSingleFaceDataSet.setObjectName("btnValidateSingleFaceDataSet")
        self.btnGenMultiFaceDataSet = QtWidgets.QPushButton(self.centralwidget)
        self.btnGenMultiFaceDataSet.setGeometry(QtCore.QRect(1190, 500, 131, 31))
        self.btnGenMultiFaceDataSet.setObjectName("btnGenMultiFaceDataSet")
        self.btnValidateMultiFaceDataSet = QtWidgets.QPushButton(self.centralwidget)
        self.btnValidateMultiFaceDataSet.setGeometry(QtCore.QRect(1190, 540, 131, 31))
        self.btnValidateMultiFaceDataSet.setObjectName("btnValidateMultiFaceDataSet")
        self.btnSaveOriBBoxes = QtWidgets.QPushButton(self.centralwidget)
        self.btnSaveOriBBoxes.setGeometry(QtCore.QRect(1190, 790, 91, 31))
        self.btnSaveOriBBoxes.setObjectName("btnSaveOriBBoxes")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(1190, 350, 171, 141))
        self.groupBox.setObjectName("groupBox")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(10, 80, 121, 16))
        self.label_2.setObjectName("label_2")
        self.cmbCloseRatio = QtWidgets.QComboBox(self.groupBox)
        self.cmbCloseRatio.setGeometry(QtCore.QRect(10, 50, 81, 21))
        self.cmbCloseRatio.setObjectName("cmbCloseRatio")
        self.label_6 = QtWidgets.QLabel(self.groupBox)
        self.label_6.setGeometry(QtCore.QRect(10, 20, 141, 16))
        self.label_6.setObjectName("label_6")
        self.cmbMaxFacesPerCluster = QtWidgets.QComboBox(self.groupBox)
        self.cmbMaxFacesPerCluster.setGeometry(QtCore.QRect(10, 100, 81, 21))
        self.cmbMaxFacesPerCluster.setObjectName("cmbMaxFacesPerCluster")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1376, 23))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actiontrain = QtWidgets.QAction(MainWindow)
        self.actiontrain.setObjectName("actiontrain")
        self.menuFile.addAction(self.actiontrain)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "WiderFace数据集化简分割工具"))
        self.lblImg.setText(_translate("MainWindow", "TextLabel"))
        self.btnRandom.setText(_translate("MainWindow", "Random"))
        self.label.setText(_translate("MainWindow", "数据集选择"))
        self.btnSplitSingle.setText(_translate("MainWindow", "Test_SplitSingle"))
        self.label_3.setText(_translate("MainWindow", "输出宽:"))
        self.txtOutX.setText(_translate("MainWindow", "96"))
        self.label_4.setText(_translate("MainWindow", "输出高"))
        self.txtOutY.setText(_translate("MainWindow", "128"))
        self.label_5.setText(_translate("MainWindow", "数据集大小"))
        self.txtDatasetSize.setText(_translate("MainWindow", "300"))
        self.btnGenSingleFaceDataSet.setText(_translate("MainWindow", "生成单脸检测数据集"))
        self.btnValidateSingleFaceDataSet.setText(_translate("MainWindow", "验证单脸检测数据集"))
        self.btnGenMultiFaceDataSet.setText(_translate("MainWindow", "生成多脸检测数据集"))
        self.btnValidateMultiFaceDataSet.setText(_translate("MainWindow", "验证多脸检测数据集"))
        self.btnSaveOriBBoxes.setText(_translate("MainWindow", "保存原始标签"))
        self.groupBox.setTitle(_translate("MainWindow", "多脸检测数据集配置"))
        self.label_2.setText(_translate("MainWindow", "单张图最多物体数"))
        self.label_6.setText(_translate("MainWindow", "物体面积之和的最小比例"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actiontrain.setText(_translate("MainWindow", "train"))
