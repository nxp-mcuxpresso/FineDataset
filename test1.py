import sys

from PyQt5 import QtCore, QtGui, QtWidgets


class Delegate(QtWidgets.QStyledItemDelegate):
    def editorEvent(self, event, model, option, index):
        checked = index.data(QtCore.Qt.CheckStateRole)
        ret = QtWidgets.QStyledItemDelegate.editorEvent(self, event, model, option, index)
        if checked != index.data(QtCore.Qt.CheckStateRole):
            self.parent().checked.emit(index)
        return ret


class ListView(QtWidgets.QListView):
    checked = QtCore.pyqtSignal(QtCore.QModelIndex)
    def __init__(self, *args, **kwargs):
        super(ListView, self).__init__(*args, **kwargs)
        self.setItemDelegate(Delegate(self))


class AppRemovalPage(QtWidgets.QWizardPage):
    def __init__( self, parent=None):
        super(AppRemovalPage, self).__init__(parent)
        self.setTitle('Apps to Remove')
        self.setSubTitle('Listview')
        self.list_view = ListView(self)
        self.list_view.setMinimumSize(465, 200)

        self.model = QtGui.QStandardItemModel(self)
        for line in ('a', 'b', 'c', 'd', 'e'):
            self.item = QtGui.QStandardItem(line)
            self.item.setCheckable(True)
            self.item.setCheckState(QtCore.Qt.Unchecked)
            self.model.appendRow(self.item)

        self.list_view.setModel(self.model)
        self.list_view.checked.connect(self.onChecked)

    #@QtCore.pyqtSlot(QtCore.QModelIndex)
    def onChecked(self, index):
        item = self.model.itemFromIndex(index)
        if item.checkState() == QtCore.Qt.Checked:
            print(item.text(), "was checked")
        else:
            print(item.text(), "was unchecked")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    listview = AppRemovalPage()
    listview.show()
    sys.exit(app.exec_())