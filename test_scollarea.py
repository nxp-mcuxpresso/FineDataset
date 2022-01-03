import sys
from PyQt5.QtWidgets import *
class test(QWidget):
    def __init__(self):
        super().__init__()
        self.initui()
    def initui(self):
        la=QHBoxLayout()
        lb=QVBoxLayout()
        lc=QHBoxLayout()
        scroll=QScrollArea()
        a=QWidget()
        a.setLayout(lb)
        lb.addLayout(lc)
        for x in range(50):
            lb.addWidget(QPushButton(str(x)))
        for x in range(50):
            lc.addWidget(QPushButton(str(x)))
        scroll.setMinimumSize(400,400)
        #scrollarea 作为一个组件，可以设置窗口
        scroll.setWidget(a)
        la.addWidget(scroll)
        self.setLayout(la)
        self.show()

if __name__=='__main__':
    app=QApplication(sys.argv)
    win=test()
    sys.exit(app.exec_())
