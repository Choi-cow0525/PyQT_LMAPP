import typing
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget

import sys

# Qt.Dialog | Qt.WindowCloseButtonHint

class LMAPP(QWidget):
    def __init__(self, parent: QWidget = None, flags: Qt.WindowFlags | Qt.WindowType = Qt.Dialog | Qt.WindowCloseButtonHint | Qt.WindowStaysOnBottomHint) -> None:
        super().__init__(parent, flags)
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('Layout Maker Activated')
        self.move(300, 300)
        self.resize(400, 200)
        self.show()

if __name__ == '__main__':
    # https://stackoverflow.com/questions/5770017/qt-what-is-qapplication-simply
    app = QApplication(sys.argv)
    ex = LMAPP()
    app.exec_()