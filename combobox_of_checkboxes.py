# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 20:43:19 2018

@author: PascPeli
"""

from PyQt5 import QtGui, QtCore, QtWidgets
import sys, os

class CheckableComboBox(QtWidgets.QComboBox):
    def __init__(self):
        super(CheckableComboBox, self).__init__()
        self.view().pressed.connect(self.handleItemPressed)
        self.setModel(QtGui.QStandardItemModel(self))

    def handleItemPressed(self, index):
        item = self.model().itemFromIndex(index)
        if item.checkState() == QtCore.Qt.Checked:
            item.setCheckState(QtCore.Qt.Unchecked)
        else:
            item.setCheckState(QtCore.Qt.Checked)

class Dialog_01(QtWidgets.QMainWindow):
    def __init__(self):
        super(QtWidgets.QMainWindow,self).__init__()
        myQWidget = QtWidgets.QWidget()
        myBoxLayout = QtWidgets.QVBoxLayout()
        myQWidget.setLayout(myBoxLayout)
        self.setCentralWidget(myQWidget)
        self.ComboBox = CheckableComboBox()
        for i in range(3):
            self.ComboBox.addItem("Combobox Item " + str(i))
            item = self.ComboBox.model().item(i, 0)
            item.setCheckState(QtCore.Qt.Unchecked)
        self.toolbutton = QtWidgets.QToolButton(self)
        self.toolbutton.setText('Select Categories ')
        self.toolmenu = QtWidgets.QMenu(self)
        for i in range(3):
            action = self.toolmenu.addAction("Category " + str(i))
            action.setCheckable(True)
        self.toolbutton.setMenu(self.toolmenu)
        self.toolbutton.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        myBoxLayout.addWidget(self.toolbutton)
        myBoxLayout.addWidget(self.ComboBox)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    dialog_1 = Dialog_01()
    dialog_1.show()
    dialog_1.resize(480,320)
    sys.exit(app.exec_())