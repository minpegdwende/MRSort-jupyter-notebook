from __future__ import division
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../../")
from PyQt5 import QtCore
from PyQt5 import QtGui

class QGraphPiecewiseLinear(QtGui.QGraphicsScene):

    def __init__(self, piecewise_linear, size, parent = None):
        super(QGraphPiecewiseLinear, self).__init__(parent)

        self.pl = piecewise_linear
        self.update(size)

    def update(self, size):
        self.size = size
        axis_height = self.size.height() - 60
        axis_width = self.size.width() - 70
        self.ymin, self.ymax = - 25 / 2, - axis_height + 25 / 2
        self.xmin, self.xmax = 25 / 2, axis_width + 25 / 2
        self.xmax_excess = 25
        self.ymax_excess = 25

        self.xcoord = []
        self.ycoord = []

        self.clear()
        self.__plot_axis()
        self.__plot_function()
        self.setSceneRect(self.itemsBoundingRect())

    def __plot_axis(self):
        item = QtGui.QGraphicsPathItem()

        path = item.path()
        path.moveTo(self.xmin, self.ymin)
        path.lineTo(self.xmax + self.xmax_excess, self.ymin)
        path.moveTo(self.xmax + self.xmax_excess, self.ymin)
        path.lineTo(self.xmax + self.xmax_excess - 6, self.ymin - 3)
        path.lineTo(self.xmax + self.xmax_excess - 6, self.ymin + 3)
        path.closeSubpath()

        path.moveTo(self.xmin, self.ymin)
        path.lineTo(self.xmin, self.ymax - self.ymax_excess)
        path.moveTo(self.xmin, self.ymax - self.ymax_excess)
        path.lineTo(self.xmin - 3, self.ymax - self.ymax_excess + 6)
        path.lineTo(self.xmin + 3, self.ymax - self.ymax_excess + 6)
        path.closeSubpath()

        item.setPath(path)

        pen = QtGui.QPen()
        pen.setWidth(2)
        item.setPen(pen)

        brush = QtGui.QBrush(QtGui.QColor("black"))
        item.setBrush(brush)

        self.addItem(item)

    def __get_coordinates(self, point):
        xmin, xmax = self.pl.xmin, self.pl.xmax
        ymin, ymax = self.pl.ymin, self.pl.ymax

        if xmax - xmin != 0:
            x = (point.x - xmin) / (xmax - xmin)
        else:
            x = 0

        if ymax - ymin != 0:
            y = (point.y - ymin) / (ymax - ymin)
        else:
            y = 0

        x = self.xmin + x * (self.xmax - self.xmin)
        y = self.ymin + y * (self.ymax - self.ymin)

        return x, y

    def __create_text_coordinate(self, x):
        item = QtGui.QGraphicsTextItem()
        item.setPlainText("%3.3g" % x)
        item.setZValue(1)

        font = QtGui.QFont()
        font.setBold(True)
        font.setPointSize(6)
        item.setFont(font)

        return item

    def __plot_coordinates(self, point, x, y):
        item = QtGui.QGraphicsPathItem()

        pen = QtGui.QPen(QtCore.Qt.DashLine)
        pen.setWidth(1)
        pen.setBrush(QtGui.QColor("black"))
        item.setPen(pen)

        path = item.path()

        path.moveTo(x, self.ymin)
        if x not in self.xcoord:
            path.lineTo(x, y)

            txtitem = self.__create_text_coordinate(point.x)
            txtitem.setPos(x - txtitem.boundingRect().width() / 2,
                           self.ymin)
            self.addItem(txtitem)

            self.xcoord.append(x)
        else:
            path.moveTo(x, y)

        if y not in self.ycoord:
            path.lineTo(self.xmin, y)

            txtitem = self.__create_text_coordinate(point.y)
            txtitem.setPos(self.xmin,
                           y - txtitem.boundingRect().height() / 2 - 3)
            self.addItem(txtitem)

            self.ycoord.append(y)

        item.setPath(path)
        self.addItem(item)

    def __plot_function(self):
        item = QtGui.QGraphicsPathItem()

        pen = QtGui.QPen()
        pen.setWidth(2)
        pen.setBrush(QtGui.QColor("red"))
        item.setPen(pen)

        path = item.path()
        for segment in self.pl.get_ordered():
            if segment.p1.x < segment.p2.x:
                x1, y1 = self.__get_coordinates(segment.p1)
                x2, y2 = self.__get_coordinates(segment.p2)
            else:
                x2, y2 = self.__get_coordinates(segment.p1)
                x1, y1 = self.__get_coordinates(segment.p2)

            path.moveTo(x1, y1)
            path.lineTo(x2, y2)

            self.__plot_coordinates(segment.p1, x1, y1)
            self.__plot_coordinates(segment.p2, x2, y2)

        item.setPath(path)
        self.addItem(item)

class QGraphCriterionFunction(QGraphPiecewiseLinear):

    def __init__(self, cfunction, size, parent = None):
        self.cfunction = cfunction
        super(QGraphCriterionFunction, self).__init__(cfunction.function,
                                                      size, parent)

    def __create_text_axis(self, txt):
        item = QtGui.QGraphicsTextItem(txt)

        font = QtGui.QFont()
        font.setBold(True)
        font.setPointSize(6)
        item.setFont(font)

        return item

    def update(self, size):
        super(QGraphCriterionFunction, self).update(size)

        xitem = self.__create_text_axis(self.cfunction.id)
        yitem = self.__create_text_axis("u()")

        xitem.setPos(self.xmax + self.xmax_excess \
                     - xitem.boundingRect().width(), self.ymin)
        yitem.setPos(self.xmin - yitem.boundingRect().width(),
                     self.ymax - self.ymax_excess)

        self.addItem(xitem)
        self.addItem(yitem)

class _MyGraphicsview(QtGui.QGraphicsView):

    def __init__(self, parent = None):
        super(QtGui.QGraphicsView, self).__init__(parent)

    def resizeEvent(self, event):
        scene = self.scene()
        scene.update(self.size())
        self.resetCachedContent()

def display_utadis_model(cfs):
    app = QtGui.QApplication(sys.argv)

    dialog = QtGui.QDialog()
    layout = QtGui.QGridLayout(dialog)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
    sizePolicy.setHorizontalStretch(1)
    sizePolicy.setVerticalStretch(0)
    sizePolicy.setHeightForWidth(sizePolicy.hasHeightForWidth())

    n_per_row = len(cfs) / 2
    i = 0
    for cf in cfs:
        view = _MyGraphicsview()
        view.setRenderHint(QtGui.QPainter.Antialiasing)
        view.setSizePolicy(sizePolicy)
        graph = QGraphCriterionFunction(cf, view.size(), parent = view)

        layout.addWidget(view, i / n_per_row, i % n_per_row)
        i = i + 1

        view.setScene(graph)

    dialog.setLayout(layout)
    dialog.resize(640, 480)
    dialog.show()

    app.exec_()

if __name__ == "__main__":
    from pymcda.types import Criterion, Criteria
    from pymcda.types import CriterionValue, CriteriaValues
    from pymcda.types import PiecewiseLinear, Segment, Point
    from pymcda.types import CriterionFunction, CriteriaFunctions

    c1 = Criterion("c1")
    c2 = Criterion("c2")
    c3 = Criterion("c3")
    c = Criteria([c1, c2, c3])

    cv1 = CriterionValue("c1", 0.5)
    cv2 = CriterionValue("c2", 0.25)
    cv3 = CriterionValue("c3", 0.25)
    cvs = CriteriaValues([cv1, cv2, cv3])

    f1 = PiecewiseLinear([Segment('s1', Point(0, 0), Point(2.5, 0.2)),
                           Segment('s2', Point(2.5, 0.2), Point(5, 1),
                                   True, True)])
    f2 = PiecewiseLinear([Segment('s1', Point(0, 0), Point(2.5, 0.8)),
                           Segment('s2', Point(2.5, 0.8), Point(5, 1),
                                   True, True)])
    f3 = PiecewiseLinear([Segment('s1', Point(0, 0), Point(2.5, 0.5)),
                           Segment('s2', Point(2.5, 0.5), Point(5, 1),
                                   True, True)])
    cf1 = CriterionFunction("c1", f1)
    cf2 = CriterionFunction("c2", f2)
    cf3 = CriterionFunction("c3", f3)
    cfs = CriteriaFunctions([cf1, cf2, cf3])

    app = QtGui.QApplication(sys.argv)

    sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding,
                                   QtGui.QSizePolicy.Expanding)
    sizePolicy.setHorizontalStretch(1)
    sizePolicy.setVerticalStretch(0)
    sizePolicy.setHeightForWidth(sizePolicy.hasHeightForWidth())

    view = _MyGraphicsview()
    view.setRenderHint(QtGui.QPainter.Antialiasing)
    view.setSizePolicy(sizePolicy)
    view2 = _MyGraphicsview()
    view2.setRenderHint(QtGui.QPainter.Antialiasing)
    view2.setSizePolicy(sizePolicy)
    view3 = _MyGraphicsview()
    view3.setRenderHint(QtGui.QPainter.Antialiasing)
    view3.setSizePolicy(sizePolicy)

    dialog = QtGui.QDialog()
    hbox = QtGui.QHBoxLayout(dialog)

    hbox.addWidget(view)
    hbox.addWidget(view2)
    hbox.addWidget(view3)

    graph = QGraphCriterionFunction(cf1, view.size(), parent = view)
    graph2 = QGraphCriterionFunction(cf2, view2.size(), parent = view2)
    graph3 = QGraphCriterionFunction(cf3, view3.size(), parent = view3)

    view.setScene(graph)
    view2.setScene(graph2)
    view3.setScene(graph3)

    dialog.setLayout(hbox)
    dialog.resize(600, 200)
    dialog.show()

    app.exec_()
