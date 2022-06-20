import sys
import os
from essential_generators import DocumentGenerator
from PyQt5.QtWidgets import (
    QPushButton,
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QMenu,
    QAction,
    QVBoxLayout,
    QHBoxLayout
)
from PyQt5.QtGui import (
    QPixmap,
    QPainter
)
from PyQt5 import QtCore
import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

#import tensorflow as tf

from Encoder import Encoder
from Visualizer import Visualizer

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class InputWidget(QWidget):
    """ Input widget. """
    def __init__(self):
        """ Initializer. """
        super().__init__()
        self.label = QLabel()
        self.label.setText("Input field")
        self.label.setFixedSize(100, 20)
        self.label.setAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter
        )

        self.widget = QLabel()
        self.widget.setAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter
        )
        self.setCursor(QtCore.Qt.CrossCursor)

        canvas = QPixmap(1280, 340)
        canvas.fill() # Fill canvas with color (default is white).
        self.widget.setPixmap(canvas)

        layout = QVBoxLayout()
        layout.addWidget(self.label, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.widget, alignment=QtCore.Qt.AlignCenter)
        self.setLayout(layout)

        self.setFixedSize(1280, 300)

        self.initialize_offsets()

        self.last_x, self.last_y = None, None
        self.current_stroke = []
        self.history = []

    def initialize_offsets(self):
        """
        Keep track of the offset of the drawing canvas in the widget and the
        cursor offset while drawing.
        """
        self.canvas_offset_left = (self.widget.width() -
                self.widget.pixmap().width())/2
        self.canvas_offset_top = (self.widget.height() -
                self.widget.pixmap().height())/2
        self.cursor_offset_top = 12.5
        self.cursor_offset_left = 12.5

    def getHistory(self):
        return self.history

    def getCurrentStroke(self):
        return self.current_stroke

    def getLastX(self):
        return self.last_x

    def getLastY(self):
        return self.last_y

    def clearCanvas(self):
        """ Clears drawing canvas. """
        self.widget.pixmap().fill()
        self.history = []
        self.update()

    def tabletEvent(self, event):
        """ Handles tablet events. """
        current_x = int(event.x() - self.canvas_offset_left -
                self.cursor_offset_left)
        current_y = int(event.y() - self.canvas_offset_top -
                self.cursor_offset_top)

        if self.last_x is None: # First event.
            self.last_x = current_x
            self.last_y = current_y
            return # Ignore the first time.

        painter = QPainter(self.widget.pixmap())
        painter.drawLine(self.last_x, self.last_y, current_x, current_y)
        painter.end()
        self.update()

        # Update the origin for next time.
        self.last_x = current_x
        self.last_y = current_y
        self.current_stroke.append(
                [
                    self.last_x,
                    self.last_y,
                    QtCore.QDateTime.currentMSecsSinceEpoch()
                ]
        )

    def mouseReleaseEvent(self, event):
        """
        If the mouse is released that means the end of the current stroke.
        The stroke is added to the history.
        """
        if len(self.current_stroke) > 0:
            self.history.append(self.current_stroke)

        self.current_stroke = []
        self.last_x = None
        self.last_y = None

    def resizeEvent(self, event):
        """ Handles resizing of the canvas. """
        self.canvas_offset_left =(self.widget.width() -
                self.widget.pixmap().width())/2
        self.canvas_offset_top =(self.widget.height() -
                self.widget.pixmap().height())/2


class DisplayWidget(QWidget):
    """ Drawing widget. """
    def __init__(self):
        """ Initializer. """
        super().__init__()
        self.widget = QLabel()
        self.widget.setAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter
        )

        canvas = QPixmap(640, 640)
        canvas.fill() # Fill canvas with color (default is white).
        self.widget.setPixmap(canvas)

        self.text_widget = DisplayTextWidget("")

        layout = QVBoxLayout()
        layout.addWidget(self.text_widget)
        layout.addWidget(self.widget)
        self.setLayout(layout)

    def showModelOutput(self, text, output):
        """ Displays the output of the model. """
        self.text_widget.updateText(text)
#         output = np.array(output)
        # output[:,:3] *= 640

        # painter = QPainter(self.widget.pixmap())

        # last_x = output[0, 0]
        # last_y = output[0, 1]

        # for (current_x, current_y, _, _, _) in output[1:]:
            # painter.drawLine(last_x, last_y, current_x, current_y)
            # last_x = current_x
            # last_y = current_y

        # painter.end()
        # self.update()

    def drawStrokes(self, strokes):
        """ Draws results on the display widget. """
        painter = QPainter(self.widget.pixmap())

        for stroke in strokes:
            if len(stroke) > 0:
                last_x = stroke[0][0]
                last_y = stroke[0][1]

            for (current_x, current_y) in stroke:
                painter.drawLine(last_x, last_y, current_x, current_y)
                last_x = current_x
                last_y = current_y

        painter.end()
        self.update()

    def clearCanvas(self):
        """ Clears display canvas. """
        self.widget.pixmap().fill()
        self.update()


class DisplayTextWidget(QWidget):
    """ Drawing widget. """
    def __init__(self, text):
        """ Initializer. """
        super().__init__()
        self.widget = QLabel(text)
        self.widget.setAlignment(
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter
        )

        layout = QHBoxLayout()
        layout.addWidget(self.widget)
        self.setLayout(layout)

    def updateText(self, new_text):
        """ Changes displayed text. """
        self.widget.setText(new_text)


class RecordingWindow(QMainWindow):
    """" Window for recording a personal dataset. """
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = dataset_path

        self.setWindowTitle("AutoWrite Recording Window")
        self.resize(640, 640)

        self.number_of_samples = self.getNumberOfSamples()

        # Creates menu.
        self._createActions()
        self._connectActions()
        self._createMenuBar()

        self.generator = DocumentGenerator()
        self.current_text = self.generator.sentence()

        self.drawingWidget = InputWidget()
        self.displayProgressWidget = DisplayTextWidget(
                f"progress: {self.number_of_samples}/100"
                )
        self.displayTextWidget = DisplayTextWidget(self.current_text)

        self.centralWidget = QWidget()
        self.centralWidgetLayout = QVBoxLayout()
        self.centralWidgetLayout.addWidget(self.displayProgressWidget)
        self.centralWidgetLayout.addWidget(self.displayTextWidget)
        self.centralWidgetLayout.addWidget(self.drawingWidget)
        self.centralWidget.setLayout(self.centralWidgetLayout)

        self.setCentralWidget(self.centralWidget)

    def _createActions(self):
        """ Creates actions. """
        self.goToNextSampleAction = QAction("&Input next sample (Enter)", self)
        self.clearCanvasAction = QAction("&Clear Canvas (c)", self)
        self.exitAction = QAction("&Exit", self)

    def _connectActions(self):
        """" Connects actions to functions. """
        self.goToNextSampleAction.triggered.connect(self.goToNextSample)
        self.clearCanvasAction.triggered.connect(self.clearCanvas)
        self.exitAction.triggered.connect(self.close) # Inbuilt function.

    def _createMenuBar(self):
        """ Creates the menu bar. """
        menuBar = self.menuBar()

        # Add file menu options.
        menu = menuBar.addMenu("&Menu")
        menu.addAction(self.goToNextSampleAction)
        menu.addAction(self.clearCanvasAction)
        menu.addAction(self.exitAction)

    def goToNextSample(self):
        """ Stores current input with label and generates new sample. """
        data = self.drawingWidget.getHistory()

        if len(data) == 0:
            return

        np.save(f"{self.dataset_path}/{self.number_of_samples}", padData(data))

        with open(f"{self.dataset_path}/{self.number_of_samples}.txt", "w") as f:
            f.write(self.current_text)

        self.current_text = self.generator.sentence()
        self.displayTextWidget.updateText(self.current_text)
        self.number_of_samples += 1
        self.displayProgressWidget.updateText(
                f"progress: {self.number_of_samples}/100"
                )
        self.clearCanvas()

    def clearCanvas(self):
        """ Clears drawing canvas. """
        self.drawingWidget.clearCanvas()

    def keyPressEvent(self, event):
        """ Handels shortcuts. """
        if event.key() == QtCore.Qt.Key_Return:
            self.goToNextSample()
        if event.key() == QtCore.Qt.Key_C:
            self.clearCanvas()

    def getNumberOfSamples(self):
        """ Returns the current number of stored samples. """
        return int(len([name for name in os.listdir(self.dataset_path)])/2)

class MainWindow(QMainWindow):
    """ Main Window. """
    def __init__(self, last_input_path, dataset_path):
        """ Initializer. """
        super().__init__()
        self.last_input_path = last_input_path
        self.dataset_path = dataset_path

        self.setWindowTitle("AutoWrite")

        self.processInputButton = QPushButton()
        self.processInputButton.setText("Process input")
        self.processInputButton.setFixedSize(100, 20)
        self.processInputButton.clicked.connect(self.processInput)

        # Creates menu.
        self._createActions()
        self._connectActions()
        self._createMenuBar()

        self.inputWidget = InputWidget()

        self.bezierCurveWidget = MplCanvas(self, width=5, height=4, dpi=100)

        self.displayBezierWidget = DisplayWidget()
        self.displayOutputWidget = DisplayWidget()

        self.centralWidget = QWidget()
        self.centralWidgetLayout = QVBoxLayout()
        self.centralWidgetLayout.addWidget(
            self.inputWidget,
            alignment=QtCore.Qt.AlignCenter
        )
        self.centralWidgetLayout.addWidget(
            self.processInputButton,
            alignment=QtCore.Qt.AlignCenter
        )
        self.centralWidgetLayout.addWidget(
            self.bezierCurveWidget,
            alignment=QtCore.Qt.AlignCenter
        )
##        self.centralWidgetLayout.addWidget(self.displayBezierWidget)
#        self.centralWidgetLayout.addWidget(self.displayOutputWidget)
        self.centralWidget.setLayout(self.centralWidgetLayout)

        self.setCentralWidget(self.centralWidget)

        self.recordingWindow = RecordingWindow(dataset_path)

        self.encoder = Encoder(
                "./model_data/weights/cp.ckpt",
                "./model_data/alphabet"
        )

        self.bezier_visualizer = Visualizer(self.bezierCurveWidget.axes)

    def _createActions(self):
        """ Creates actions. """
        self.saveAction = QAction("&Save current input", self)
        self.clearCanvassesAction = QAction("&Clear Canvasses", self)
        self.recordAction = QAction("&Record personal data set", self)
        self.exitAction = QAction("&Exit", self)

    def _connectActions(self):
        """" Connects actions to functions. """
        self.saveAction.triggered.connect(self.saveData)
        self.exitAction.triggered.connect(self.close) # Inbuilt function.
        self.clearCanvassesAction.triggered.connect(self.clearCanvasses)
        self.recordAction.triggered.connect(self.record)

    def _createMenuBar(self):
        """ Creates the menu bar. """
        menuBar = self.menuBar()

        # Add file menu options.
        menu = menuBar.addMenu("&Menu")
        menu.addAction(self.saveAction)
        menu.addAction(self.clearCanvassesAction)
        menu.addAction(self.recordAction)
        menu.addAction(self.exitAction)

    def processInput(self):
        """ Calls the model on the current input. """
        data = self.inputWidget.getHistory()

        if not data:
            return

        data = padData(data)
        bezier_curves = self.encoder.preprocess(data)

        self.bezierCurveWidget.axes.cla()
        self.bezier_visualizer.plot_bezier_curves(bezier_curves)
#       self.bezierCurveWidget.axes.plot([0,1,2,3,4], [10,1,20,3,40])
        self.bezierCurveWidget.draw()
#         output = self.encoder.call(
            # tf.convert_to_tensor(np.expand_dims(bezier_curves, 0))
        # )

        # text = self.encoder.decode_output(output)

        # self.displayWidget.showModelOutput(text, [])

    def saveData(self):
        """ Saves the stroke currently on canvas. """
        data = self.inputWidget.getHistory()

        if len(data) == 0:
            return

        np.save(self.last_input_path + '/strokes', padData(data))

    def clearCanvasses(self):
        """ Clears all canvasses. """
        self.drawingWidget.clearCanvas()
        self.displayWidget.clearCanvas()

    def record(self):
        """
        Opens a new window in which the user can record a personal dataset.
        """
        self.recordingWindow.show()

def padData(data):
    """ Stores a ragged list in a numpy array by padding it. """
    max_stroke_len = max(len(r) for r in data)

    padded_data = np.zeros((len(data), max_stroke_len, 3))
    padded_data[:, :, 2] -= 1

    for i, row in enumerate(data):
        padded_data[i, :len(row)] = row

    return padded_data

def makeDirs(dirs):
    """ Makes directories to store data if they do not yet exist. """
    for path in dirs:
        if not os.path.exists(path):
            os.makedirs(path)

def run():
    """ Entry point for the application. """
    autowrite_last_input_path = "./data/last_input"
    autowrite_personal_dataset_path = "./data/personal_data_set"
    makeDirs([autowrite_last_input_path, autowrite_personal_dataset_path])

    app = QApplication(sys.argv)

    window = MainWindow(autowrite_last_input_path, autowrite_personal_dataset_path)
    window.show()

    app.exec_()

if __name__ == "__main__":
    run()
