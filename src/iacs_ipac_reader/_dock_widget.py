import napari
from napari_plugin_engine import napari_hook_implementation
from PyQt5 import QtCore, QtGui, QtWidgets

import cv2, h5py, os, time, shutil, pickle, csv
import numpy as np
import pandas as pd

from .aid_cv2_dnn import*
from .image_processing import get_masks_ipac,get_masks_iacs,tiled_2_list,get_boundingbox_features,get_brightness,get_contourfeatures,add_colormap,get_color,uint16_2_unit8,vstripes_removal,hstripes_removal
from .ApplyRF import*
from .background_program import*



#Turn off error warning
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)  

dir_root = os.path.dirname(__file__)#ask the module for its origin

class MyTable(QtWidgets.QTableWidget):
    dropped = QtCore.pyqtSignal(list) 

    def __init__(self,  rows, columns, parent):
        super(MyTable, self).__init__(rows, columns, parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        #self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.drag_item = None
        self.drag_row = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        self.drag_item = None
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
            links = []
            
            for url in event.mimeData().urls():
                links.append(str(url.toLocalFile()))  
            self.dropped.emit(links) #发射信号
            
        else:
            event.ignore()       

    def startartDrag(self, supportedActions):
        super(MyTable, self).startDrag(supportedActions)
        self.drag_item = self.currentItem()
        self.drag_row = self.row(self.drag_item)

        
class LineEdit(QtWidgets.QLineEdit):
        def __init__( self, parent ):
            super(LineEdit, self).__init__(parent)
            self.setDragEnabled(True)

        def dragEnterEvent( self, event ):
            data = event.mimeData()
            urls = data.urls()
            if ( urls and urls[0].scheme() == 'file' ):
                event.acceptProposedAction()

        def dragMoveEvent( self, event ):
            data = event.mimeData()
            urls = data.urls()
            if ( urls and urls[0].scheme() == 'file' ):
                event.acceptProposedAction()

        def dropEvent( self, event ):
            data = event.mimeData()
            urls = data.urls()
            if ( urls and urls[0].scheme() == 'file' ):
                # for some reason, this doubles up the intro slash
                filepath = str(urls[0].path())[1:]
                self.setText(filepath)


class iacs_ipac_reader(QtWidgets.QWidget):
    
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        Form = QtWidgets.QWidget()
        self.setLayout(QtWidgets.QGridLayout())
        self.layout().addWidget(Form)
        
        Form.setObjectName("Form")
        Form.setMaximumSize(QtCore.QSize(600, 16777215))
        
        self.gridLayout_4 = QtWidgets.QGridLayout(Form)
        self.gridLayout_4.setContentsMargins(1, 1, 1, 1)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.tabWidget = QtWidgets.QTabWidget(Form)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_ipac = QtWidgets.QWidget()
        self.tab_ipac.setObjectName("tab_ipac")
        self.gridLayout = QtWidgets.QGridLayout(self.tab_ipac)
        self.gridLayout.setContentsMargins(6, 6, 6, 6)
        self.gridLayout.setObjectName("gridLayout")
        self.groupBox_ipac = QtWidgets.QGroupBox(self.tab_ipac)
        self.groupBox_ipac.setObjectName("groupBox_ipac")
        self.gridLayout_21 = QtWidgets.QGridLayout(self.groupBox_ipac)
        self.gridLayout_21.setObjectName("gridLayout_21")
        
        ##### Manually change ####
        self.table_ipac = MyTable(0,5,self.groupBox_ipac)
        self.table_ipac.setObjectName("table_ipac")
        
        self.gridLayout_21.addWidget(self.table_ipac, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.groupBox_ipac, 0, 0, 1, 1)
        self.line_ipac = QtWidgets.QFrame(self.tab_ipac)
        self.line_ipac.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_ipac.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_ipac.setObjectName("line_ipac")
        self.gridLayout.addWidget(self.line_ipac, 1, 0, 1, 1)
        self.splitter_ipac_pixel = QtWidgets.QSplitter(self.tab_ipac)
        self.splitter_ipac_pixel.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_ipac_pixel.setObjectName("splitter_ipac_pixel")
        self.label_ipac_pixel = QtWidgets.QLabel(self.splitter_ipac_pixel)
        self.label_ipac_pixel.setObjectName("label_ipac_pixel")
        self.ipac_SpinBox_pixel = QtWidgets.QDoubleSpinBox(self.splitter_ipac_pixel)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ipac_SpinBox_pixel.sizePolicy().hasHeightForWidth())
        self.ipac_SpinBox_pixel.setSizePolicy(sizePolicy)
        self.ipac_SpinBox_pixel.setDecimals(5)
        self.ipac_SpinBox_pixel.setProperty("value", 0.8)
        self.ipac_SpinBox_pixel.setObjectName("ipac_SpinBox_pixel")
        self.gridLayout.addWidget(self.splitter_ipac_pixel, 2, 0, 1, 1)
        self.groupBox_ipac_thresh = QtWidgets.QGroupBox(self.tab_ipac)
        self.groupBox_ipac_thresh.setEnabled(True)
        self.groupBox_ipac_thresh.setObjectName("groupBox_ipac_thresh")
        self.gridLayout_22 = QtWidgets.QGridLayout(self.groupBox_ipac_thresh)
        self.gridLayout_22.setObjectName("gridLayout_22")
        self.label_ipac_noise = QtWidgets.QLabel(self.groupBox_ipac_thresh)
        self.label_ipac_noise.setEnabled(True)
        self.label_ipac_noise.setObjectName("label_ipac_noise")
        self.gridLayout_22.addWidget(self.label_ipac_noise, 0, 0, 1, 1)
        self.Slider_ipac_noise = QtWidgets.QSlider(self.groupBox_ipac_thresh)
        self.Slider_ipac_noise.setEnabled(True)
        self.Slider_ipac_noise.setMaximum(100)
        self.Slider_ipac_noise.setProperty("value", 10)
        self.Slider_ipac_noise.setOrientation(QtCore.Qt.Horizontal)
        self.Slider_ipac_noise.setObjectName("Slider_ipac_noise")
        self.gridLayout_22.addWidget(self.Slider_ipac_noise, 0, 2, 1, 1)
        self.doubleSpinBox_ipac_noise = QtWidgets.QDoubleSpinBox(self.groupBox_ipac_thresh)
        self.doubleSpinBox_ipac_noise.setEnabled(True)
        self.doubleSpinBox_ipac_noise.setDecimals(1)
        self.doubleSpinBox_ipac_noise.setProperty("value", 10.0)
        self.doubleSpinBox_ipac_noise.setObjectName("doubleSpinBox_ipac_noise")
        self.gridLayout_22.addWidget(self.doubleSpinBox_ipac_noise, 0, 1, 1, 1)
        self.gridLayout.addWidget(self.groupBox_ipac_thresh, 3, 0, 1, 1)
        self.groupBox_ipac_contour = QtWidgets.QGroupBox(self.tab_ipac)
        self.groupBox_ipac_contour.setObjectName("groupBox_ipac_contour")
        self.gridLayout_23 = QtWidgets.QGridLayout(self.groupBox_ipac_contour)
        self.gridLayout_23.setObjectName("gridLayout_23")
        self.checkBox_ipac_ca = QtWidgets.QCheckBox(self.groupBox_ipac_contour)
        self.checkBox_ipac_ca.setObjectName("checkBox_ipac_ca")
        self.gridLayout_23.addWidget(self.checkBox_ipac_ca, 2, 0, 1, 1)
        self.spinBox_ipac_cl_max = QtWidgets.QSpinBox(self.groupBox_ipac_contour)
        self.spinBox_ipac_cl_max.setMaximum(999999)
        self.spinBox_ipac_cl_max.setProperty("value", 50)
        self.spinBox_ipac_cl_max.setObjectName("spinBox_ipac_cl_max")
        self.gridLayout_23.addWidget(self.spinBox_ipac_cl_max, 1, 2, 1, 1)
        self.label_ipac_min = QtWidgets.QLabel(self.groupBox_ipac_contour)
        self.label_ipac_min.setAlignment(QtCore.Qt.AlignCenter)
        self.label_ipac_min.setObjectName("label_ipac_min")
        self.gridLayout_23.addWidget(self.label_ipac_min, 0, 1, 1, 1)
        self.checkBox_ipac_cn = QtWidgets.QCheckBox(self.groupBox_ipac_contour)
        self.checkBox_ipac_cn.setChecked(True)
        self.checkBox_ipac_cn.setObjectName("checkBox_ipac_cn")
        self.gridLayout_23.addWidget(self.checkBox_ipac_cn, 3, 0, 1, 1)
        self.spinBox_ipac_ca_min = QtWidgets.QSpinBox(self.groupBox_ipac_contour)
        self.spinBox_ipac_ca_min.setMaximum(999999)
        self.spinBox_ipac_ca_min.setProperty("value", 10)
        self.spinBox_ipac_ca_min.setObjectName("spinBox_ipac_ca_min")
        self.gridLayout_23.addWidget(self.spinBox_ipac_ca_min, 2, 1, 1, 1)
        self.spinBox_ipac_cn = QtWidgets.QSpinBox(self.groupBox_ipac_contour)
        self.spinBox_ipac_cn.setMaximum(100)
        self.spinBox_ipac_cn.setProperty("value", 10)
        self.spinBox_ipac_cn.setObjectName("spinBox_ipac_cn")
        self.gridLayout_23.addWidget(self.spinBox_ipac_cn, 3, 1, 1, 2)
        self.spinBox_ipac_ca_max = QtWidgets.QSpinBox(self.groupBox_ipac_contour)
        self.spinBox_ipac_ca_max.setMaximum(999999)
        self.spinBox_ipac_ca_max.setProperty("value", 50)
        self.spinBox_ipac_ca_max.setObjectName("spinBox_ipac_ca_max")
        self.gridLayout_23.addWidget(self.spinBox_ipac_ca_max, 2, 2, 1, 1)
        self.label_ipac_max = QtWidgets.QLabel(self.groupBox_ipac_contour)
        self.label_ipac_max.setAlignment(QtCore.Qt.AlignCenter)
        self.label_ipac_max.setObjectName("label_ipac_max")
        self.gridLayout_23.addWidget(self.label_ipac_max, 0, 2, 1, 1)
        self.checkBox_ipac_cl = QtWidgets.QCheckBox(self.groupBox_ipac_contour)
        self.checkBox_ipac_cl.setChecked(True)
        self.checkBox_ipac_cl.setObjectName("checkBox_ipac_cl")
        self.gridLayout_23.addWidget(self.checkBox_ipac_cl, 1, 0, 1, 1)
        self.spinBox_ipac_cl_min = QtWidgets.QSpinBox(self.groupBox_ipac_contour)
        self.spinBox_ipac_cl_min.setMaximum(999999)
        self.spinBox_ipac_cl_min.setProperty("value", 10)
        self.spinBox_ipac_cl_min.setObjectName("spinBox_ipac_cl_min")
        self.gridLayout_23.addWidget(self.spinBox_ipac_cl_min, 1, 1, 1, 1)
        self.gridLayout.addWidget(self.groupBox_ipac_contour, 4, 0, 1, 1)
        self.groupBox_ipac_preview = QtWidgets.QGroupBox(self.tab_ipac)
        self.groupBox_ipac_preview.setObjectName("groupBox_ipac_preview")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_ipac_preview)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.splitter_7 = QtWidgets.QSplitter(self.groupBox_ipac_preview)
        self.splitter_7.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_7.setObjectName("splitter_7")
        self.checkBox_ipac_contour = QtWidgets.QCheckBox(self.splitter_7)
        self.checkBox_ipac_contour.setMaximumSize(QtCore.QSize(130, 16777215))
        self.checkBox_ipac_contour.setChecked(True)
        self.checkBox_ipac_contour.setObjectName("checkBox_ipac_contour")
        self.comboBox_ipac_cnt_color = QtWidgets.QComboBox(self.splitter_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_ipac_cnt_color.sizePolicy().hasHeightForWidth())
        self.comboBox_ipac_cnt_color.setSizePolicy(sizePolicy)
        self.comboBox_ipac_cnt_color.setObjectName("comboBox_ipac_cnt_color")
        
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(os.path.join(dir_root,"art","green.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.comboBox_ipac_cnt_color.addItem(icon, "")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(os.path.join(dir_root,"art","aqua.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.comboBox_ipac_cnt_color.addItem(icon1, "")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(os.path.join(dir_root,"art","red.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.comboBox_ipac_cnt_color.addItem(icon2, "")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(os.path.join(dir_root,"art","black.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.comboBox_ipac_cnt_color.addItem(icon3, "")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(os.path.join(dir_root,"art","white.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        
        self.comboBox_ipac_cnt_color.addItem(icon4, "")
        self.gridLayout_5.addWidget(self.splitter_7, 0, 0, 1, 2)
        self.splitter_8 = QtWidgets.QSplitter(self.groupBox_ipac_preview)
        self.splitter_8.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_8.setObjectName("splitter_8")
        self.checkBox_ipac_index = QtWidgets.QCheckBox(self.splitter_8)
        self.checkBox_ipac_index.setMaximumSize(QtCore.QSize(130, 16777215))
        self.checkBox_ipac_index.setChecked(True)
        self.checkBox_ipac_index.setObjectName("checkBox_ipac_index")
        self.comboBox_ipac_ind_color = QtWidgets.QComboBox(self.splitter_8)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_ipac_ind_color.sizePolicy().hasHeightForWidth())
        self.comboBox_ipac_ind_color.setSizePolicy(sizePolicy)
        self.comboBox_ipac_ind_color.setObjectName("comboBox_ipac_ind_color")
        self.comboBox_ipac_ind_color.addItem(icon4, "")
        self.comboBox_ipac_ind_color.addItem(icon3, "")
        self.comboBox_ipac_ind_color.addItem(icon, "")
        self.comboBox_ipac_ind_color.addItem(icon2, "")
        self.comboBox_ipac_ind_color.addItem(icon1, "")
        self.gridLayout_5.addWidget(self.splitter_8, 1, 0, 1, 2)
        self.gridLayout.addWidget(self.groupBox_ipac_preview, 5, 0, 1, 1)
        self.splitter_ipac_btn = QtWidgets.QSplitter(self.tab_ipac)
        self.splitter_ipac_btn.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_ipac_btn.setObjectName("splitter_ipac_btn")
        self.btn_ipac_save = QtWidgets.QPushButton(self.splitter_ipac_btn)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_ipac_save.sizePolicy().hasHeightForWidth())
        self.btn_ipac_save.setSizePolicy(sizePolicy)
        self.btn_ipac_save.setMaximumSize(QtCore.QSize(89, 16777215))
        self.btn_ipac_save.setObjectName("btn_ipac_save")
        self.splitter_ipac_pre = QtWidgets.QSplitter(self.splitter_ipac_btn)
        self.splitter_ipac_pre.setLineWidth(1)
        self.splitter_ipac_pre.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_ipac_pre.setHandleWidth(0)
        self.splitter_ipac_pre.setObjectName("splitter_ipac_pre")
        self.btn_ipac_stack = QtWidgets.QPushButton(self.splitter_ipac_pre)
        self.btn_ipac_stack.setObjectName("btn_ipac_stack")
        self.gridLayout.addWidget(self.splitter_ipac_btn, 6, 0, 1, 1)
        self.tabWidget.addTab(self.tab_ipac, "")
        self.tab_iacs = QtWidgets.QWidget()
        self.tab_iacs.setObjectName("tab_iacs")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.tab_iacs)
        self.gridLayout_7.setContentsMargins(6, 6, 6, 6)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.groupBox_iacs = QtWidgets.QGroupBox(self.tab_iacs)
        self.groupBox_iacs.setObjectName("groupBox_iacs")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox_iacs)
        self.gridLayout_6.setObjectName("gridLayout_6")
        
        ##### Manually change ####
        self.table_iacs = MyTable(0,5,self.groupBox_iacs)
        self.table_iacs.setObjectName("table_iacs")
        
        
        self.gridLayout_6.addWidget(self.table_iacs, 0, 0, 1, 2)
        self.label = QtWidgets.QLabel(self.groupBox_iacs)
        self.label.setObjectName("label")
        self.gridLayout_6.addWidget(self.label, 1, 0, 1, 1)
        self.spinBox_n_channel = QtWidgets.QSpinBox(self.groupBox_iacs)
        self.spinBox_n_channel.setMinimum(1)
        self.spinBox_n_channel.setMaximum(2)
        self.spinBox_n_channel.setObjectName("spinBox_n_channel")
        self.gridLayout_6.addWidget(self.spinBox_n_channel, 1, 1, 1, 1)
        self.groupBox_colormap = QtWidgets.QGroupBox(self.groupBox_iacs)
        self.groupBox_colormap.setFlat(False)
        self.groupBox_colormap.setCheckable(True)
        self.groupBox_colormap.setChecked(False)
        self.groupBox_colormap.setObjectName("groupBox_colormap")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_colormap)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_channel0 = QtWidgets.QLabel(self.groupBox_colormap)
        self.label_channel0.setObjectName("label_channel0")
        self.gridLayout_2.addWidget(self.label_channel0, 0, 0, 1, 1)
        self.comboBox_iacs_ch0 = QtWidgets.QComboBox(self.groupBox_colormap)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_iacs_ch0.sizePolicy().hasHeightForWidth())
        self.comboBox_iacs_ch0.setSizePolicy(sizePolicy)
        self.comboBox_iacs_ch0.setObjectName("comboBox_iacs_ch0")
        self.comboBox_iacs_ch0.addItem(icon2, "")
        self.comboBox_iacs_ch0.addItem(icon, "")
        self.comboBox_iacs_ch0.addItem(icon1, "")
        self.gridLayout_2.addWidget(self.comboBox_iacs_ch0, 0, 1, 1, 1)
        self.label_channel1 = QtWidgets.QLabel(self.groupBox_colormap)
        self.label_channel1.setObjectName("label_channel1")
        self.gridLayout_2.addWidget(self.label_channel1, 1, 0, 1, 1)
        self.comboBox_iacs_ch1 = QtWidgets.QComboBox(self.groupBox_colormap)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_iacs_ch1.sizePolicy().hasHeightForWidth())
        self.comboBox_iacs_ch1.setSizePolicy(sizePolicy)
        self.comboBox_iacs_ch1.setObjectName("comboBox_iacs_ch1")
        self.comboBox_iacs_ch1.addItem(icon, "")
        self.comboBox_iacs_ch1.addItem(icon1, "")
        self.comboBox_iacs_ch1.addItem(icon2, "")
        self.gridLayout_2.addWidget(self.comboBox_iacs_ch1, 1, 1, 1, 1)
        self.gridLayout_6.addWidget(self.groupBox_colormap, 2, 0, 1, 2)
        self.gridLayout_7.addWidget(self.groupBox_iacs, 0, 0, 1, 1)
        self.line_iacs = QtWidgets.QFrame(self.tab_iacs)
        self.line_iacs.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_iacs.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_iacs.setObjectName("line_iacs")
        self.gridLayout_7.addWidget(self.line_iacs, 1, 0, 1, 1)
        self.splitter_iacs_pixel = QtWidgets.QSplitter(self.tab_iacs)
        self.splitter_iacs_pixel.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_iacs_pixel.setObjectName("splitter_iacs_pixel")
        self.label_iacs_pixel = QtWidgets.QLabel(self.splitter_iacs_pixel)
        self.label_iacs_pixel.setObjectName("label_iacs_pixel")
        self.SpinBox_iacs_pixel = QtWidgets.QDoubleSpinBox(self.splitter_iacs_pixel)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.SpinBox_iacs_pixel.sizePolicy().hasHeightForWidth())
        self.SpinBox_iacs_pixel.setSizePolicy(sizePolicy)
        self.SpinBox_iacs_pixel.setDecimals(5)
        self.SpinBox_iacs_pixel.setProperty("value", 0.36111)
        self.SpinBox_iacs_pixel.setObjectName("SpinBox_iacs_pixel")
        self.gridLayout_7.addWidget(self.splitter_iacs_pixel, 2, 0, 1, 1)
        self.groupBox_iacs_contour = QtWidgets.QGroupBox(self.tab_iacs)
        self.groupBox_iacs_contour.setObjectName("groupBox_iacs_contour")
        self.gridLayout_16 = QtWidgets.QGridLayout(self.groupBox_iacs_contour)
        self.gridLayout_16.setObjectName("gridLayout_16")
        self.label_iacs_min = QtWidgets.QLabel(self.groupBox_iacs_contour)
        self.label_iacs_min.setAlignment(QtCore.Qt.AlignCenter)
        self.label_iacs_min.setObjectName("label_iacs_min")
        self.gridLayout_16.addWidget(self.label_iacs_min, 0, 1, 1, 1)
        self.label_iacs_max = QtWidgets.QLabel(self.groupBox_iacs_contour)
        self.label_iacs_max.setAlignment(QtCore.Qt.AlignCenter)
        self.label_iacs_max.setObjectName("label_iacs_max")
        self.gridLayout_16.addWidget(self.label_iacs_max, 0, 2, 1, 1)
        self.checkBox_iacs_cl = QtWidgets.QCheckBox(self.groupBox_iacs_contour)
        self.checkBox_iacs_cl.setChecked(True)
        self.checkBox_iacs_cl.setObjectName("checkBox_iacs_cl")
        self.gridLayout_16.addWidget(self.checkBox_iacs_cl, 1, 0, 1, 1)
        self.spinBox_iacs_cl_min = QtWidgets.QSpinBox(self.groupBox_iacs_contour)
        self.spinBox_iacs_cl_min.setMaximum(999999)
        self.spinBox_iacs_cl_min.setProperty("value", 10)
        self.spinBox_iacs_cl_min.setObjectName("spinBox_iacs_cl_min")
        self.gridLayout_16.addWidget(self.spinBox_iacs_cl_min, 1, 1, 1, 1)
        self.spinBox_iacs_cl_max = QtWidgets.QSpinBox(self.groupBox_iacs_contour)
        self.spinBox_iacs_cl_max.setMaximum(999999)
        self.spinBox_iacs_cl_max.setProperty("value", 250)
        self.spinBox_iacs_cl_max.setObjectName("spinBox_iacs_cl_max")
        self.gridLayout_16.addWidget(self.spinBox_iacs_cl_max, 1, 2, 1, 1)
        self.checkBox_iacs_ca = QtWidgets.QCheckBox(self.groupBox_iacs_contour)
        self.checkBox_iacs_ca.setObjectName("checkBox_iacs_ca")
        self.gridLayout_16.addWidget(self.checkBox_iacs_ca, 2, 0, 1, 1)
        self.spinBox_iacs_ca_min = QtWidgets.QSpinBox(self.groupBox_iacs_contour)
        self.spinBox_iacs_ca_min.setMaximum(999999)
        self.spinBox_iacs_ca_min.setProperty("value", 10)
        self.spinBox_iacs_ca_min.setObjectName("spinBox_iacs_ca_min")
        self.gridLayout_16.addWidget(self.spinBox_iacs_ca_min, 2, 1, 1, 1)
        self.spinBox_iacs_ca_max = QtWidgets.QSpinBox(self.groupBox_iacs_contour)
        self.spinBox_iacs_ca_max.setMaximum(999999)
        self.spinBox_iacs_ca_max.setProperty("value", 250)
        self.spinBox_iacs_ca_max.setObjectName("spinBox_iacs_ca_max")
        self.gridLayout_16.addWidget(self.spinBox_iacs_ca_max, 2, 2, 1, 1)
        self.checkBox_iacs_cn = QtWidgets.QCheckBox(self.groupBox_iacs_contour)
        self.checkBox_iacs_cn.setChecked(True)
        self.checkBox_iacs_cn.setObjectName("checkBox_iacs_cn")
        self.gridLayout_16.addWidget(self.checkBox_iacs_cn, 3, 0, 1, 1)
        self.spinBox_iacs_cn = QtWidgets.QSpinBox(self.groupBox_iacs_contour)
        self.spinBox_iacs_cn.setMaximum(100)
        self.spinBox_iacs_cn.setProperty("value", 1)
        self.spinBox_iacs_cn.setObjectName("spinBox_iacs_cn")
        self.gridLayout_16.addWidget(self.spinBox_iacs_cn, 3, 1, 1, 2)
        self.gridLayout_7.addWidget(self.groupBox_iacs_contour, 3, 0, 1, 1)
        self.groupBox_iacs_preview_opt = QtWidgets.QGroupBox(self.tab_iacs)
        self.groupBox_iacs_preview_opt.setObjectName("groupBox_iacs_preview_opt")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_iacs_preview_opt)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.splitter_4 = QtWidgets.QSplitter(self.groupBox_iacs_preview_opt)
        self.splitter_4.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_4.setObjectName("splitter_4")
        self.checkBox_iacs_index = QtWidgets.QCheckBox(self.splitter_4)
        self.checkBox_iacs_index.setMaximumSize(QtCore.QSize(130, 16777215))
        self.checkBox_iacs_index.setChecked(True)
        self.checkBox_iacs_index.setObjectName("checkBox_iacs_index")
        self.comboBox_iacs_ind_color = QtWidgets.QComboBox(self.splitter_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_iacs_ind_color.sizePolicy().hasHeightForWidth())
        self.comboBox_iacs_ind_color.setSizePolicy(sizePolicy)
        self.comboBox_iacs_ind_color.setObjectName("comboBox_iacs_ind_color")
        self.comboBox_iacs_ind_color.addItem(icon4, "")
        self.comboBox_iacs_ind_color.addItem(icon3, "")
        self.comboBox_iacs_ind_color.addItem(icon, "")
        self.comboBox_iacs_ind_color.addItem(icon2, "")
        self.comboBox_iacs_ind_color.addItem(icon1, "")
        self.gridLayout_3.addWidget(self.splitter_4, 1, 0, 1, 2)
        self.splitter_3 = QtWidgets.QSplitter(self.groupBox_iacs_preview_opt)
        self.splitter_3.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_3.setObjectName("splitter_3")
        self.checkBox_iacs_contour = QtWidgets.QCheckBox(self.splitter_3)
        self.checkBox_iacs_contour.setMaximumSize(QtCore.QSize(130, 16777215))
        self.checkBox_iacs_contour.setChecked(True)
        self.checkBox_iacs_contour.setObjectName("checkBox_iacs_contour")
        self.comboBox_iacs_cnt_color = QtWidgets.QComboBox(self.splitter_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_iacs_cnt_color.sizePolicy().hasHeightForWidth())
        self.comboBox_iacs_cnt_color.setSizePolicy(sizePolicy)
        self.comboBox_iacs_cnt_color.setObjectName("comboBox_iacs_cnt_color")
        self.comboBox_iacs_cnt_color.addItem(icon, "")
        self.comboBox_iacs_cnt_color.addItem(icon1, "")
        self.comboBox_iacs_cnt_color.addItem(icon2, "")
        self.comboBox_iacs_cnt_color.addItem(icon3, "")
        self.comboBox_iacs_cnt_color.addItem(icon4, "")
        self.gridLayout_3.addWidget(self.splitter_3, 0, 0, 1, 2)
        self.gridLayout_7.addWidget(self.groupBox_iacs_preview_opt, 4, 0, 1, 1)
        self.splitter_iacs_btn = QtWidgets.QSplitter(self.tab_iacs)
        self.splitter_iacs_btn.setLineWidth(0)
        self.splitter_iacs_btn.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_iacs_btn.setObjectName("splitter_iacs_btn")
        self.btn_iacs_save = QtWidgets.QPushButton(self.splitter_iacs_btn)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_iacs_save.sizePolicy().hasHeightForWidth())
        self.btn_iacs_save.setSizePolicy(sizePolicy)
        self.btn_iacs_save.setMaximumSize(QtCore.QSize(89, 16777215))
        self.btn_iacs_save.setObjectName("btn_iacs_save")
        self.splitter_iacs_pre = QtWidgets.QSplitter(self.splitter_iacs_btn)
        self.splitter_iacs_pre.setLineWidth(0)
        self.splitter_iacs_pre.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_iacs_pre.setHandleWidth(0)
        self.splitter_iacs_pre.setObjectName("splitter_iacs_pre")
        self.btn_iacs_tiles = QtWidgets.QPushButton(self.splitter_iacs_pre)
        self.btn_iacs_tiles.setObjectName("btn_iacs_tiles")
        self.btn_iacs_stack = QtWidgets.QPushButton(self.splitter_iacs_pre)
        self.btn_iacs_stack.setObjectName("btn_iacs_stack")
        self.gridLayout_7.addWidget(self.splitter_iacs_btn, 5, 0, 1, 1)
        self.tabWidget.addTab(self.tab_iacs, "")
        #self.gridLayout_4.addWidget(self.tabWidget, 0, 0, 1, 1)

        self.tab_aid = QtWidgets.QWidget()
        self.tab_aid.setObjectName("tab_aid")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.tab_aid)
        self.gridLayout_9.setContentsMargins(6, 6, 6, 6)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.splitter_2 = QtWidgets.QSplitter(self.tab_aid)
        self.splitter_2.setOrientation(QtCore.Qt.Vertical)
        self.splitter_2.setChildrenCollapsible(False)
        self.splitter_2.setObjectName("splitter_2")
        self.groupBox_aid_files = QtWidgets.QGroupBox(self.splitter_2)
        self.groupBox_aid_files.setObjectName("groupBox_aid_files")
        self.gridLayout_8 = QtWidgets.QGridLayout(self.groupBox_aid_files)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.splitter = QtWidgets.QSplitter(self.groupBox_aid_files)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setObjectName("splitter")
        self.btn_aid_load_model = QtWidgets.QPushButton(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_aid_load_model.sizePolicy().hasHeightForWidth())
        self.btn_aid_load_model.setSizePolicy(sizePolicy)
        self.btn_aid_load_model.setMaximumSize(QtCore.QSize(100, 16777215))
        self.btn_aid_load_model.setObjectName("btn_aid_load_model")
        
        
        self.lineEdit_aid_model_path = LineEdit(self.splitter)
        self.lineEdit_aid_model_path.setReadOnly(True)
        #self.lineEdit_aid_model_path.setEnabled(False)
        self.lineEdit_aid_model_path.setDragEnabled(True)
        self.lineEdit_aid_model_path.setObjectName("lineEdit_aid_model_path")
        
        self.gridLayout_8.addWidget(self.splitter, 1, 0, 1, 2)
        self.btn_aid_classify = QtWidgets.QPushButton(self.groupBox_aid_files)
        self.btn_aid_classify.setObjectName("btn_aid_classify")
        self.gridLayout_8.addWidget(self.btn_aid_classify, 2, 0, 1, 2)
        ##### Manually change ####
        self.table_aid_files = MyTable(0,4,self.groupBox_aid_files)
        self.table_aid_files.setObjectName("table_aid_files")
        self.gridLayout_8.addWidget(self.table_aid_files, 0, 0, 1, 2)
        self.groupBox_aid_analy = QtWidgets.QGroupBox(self.splitter_2)
        self.groupBox_aid_analy.setObjectName("groupBox_aid_analy")
        self.gridLayout_25 = QtWidgets.QGridLayout(self.groupBox_aid_analy)
        self.gridLayout_25.setObjectName("gridLayout_25")
        
        ##### Manually change ####
        self.table_aid_analy = MyTable(0,5,self.groupBox_aid_analy)
        self.table_aid_analy.setEnabled(True)
        self.table_aid_analy.setObjectName("table_aid_analy")
        self.gridLayout_25.addWidget(self.table_aid_analy, 0, 0, 1, 1)
        
        self.groupBox_dise_class = QtWidgets.QGroupBox(self.splitter_2)
        self.groupBox_dise_class.setObjectName("groupBox_dise_class")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.groupBox_dise_class)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.table_dise_class = MyTable(0,2, self.groupBox_dise_class)
        self.table_dise_class.setObjectName("table_dise_class")
        self.gridLayout_10.addWidget(self.table_dise_class, 0, 0, 1, 1)
        
        self.gridLayout_9.addWidget(self.splitter_2, 0, 0, 1, 1)
        self.btn_aid_add_rtdc = QtWidgets.QPushButton(self.tab_aid)
        self.btn_aid_add_rtdc.setObjectName("btn_aid_add_rtdc")
        self.gridLayout_9.addWidget(self.btn_aid_add_rtdc, 1, 0, 1, 1)
        self.tabWidget.addTab(self.tab_aid, "")
        self.gridLayout_4.addWidget(self.tabWidget, 0, 0, 1, 1)


        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(Form)
        
        
        self.doubleSpinBox_ipac_noise.valueChanged['double'].connect(lambda item: self.Slider_ipac_noise.setValue(int(item)))
        self.Slider_ipac_noise.valueChanged['int'].connect(lambda item: self.doubleSpinBox_ipac_noise.setValue(float(item)))        
        self.checkBox_ipac_cl.stateChanged.connect(lambda:self.chk_statues(self.spinBox_ipac_cl_min))
        self.checkBox_ipac_cl.stateChanged.connect(lambda:self.chk_statues(self.spinBox_ipac_cl_max))
        self.checkBox_ipac_ca.stateChanged.connect(lambda:self.chk_statues(self.spinBox_ipac_ca_min))
        self.checkBox_ipac_ca.stateChanged.connect(lambda:self.chk_statues(self.spinBox_ipac_ca_max))
        self.checkBox_ipac_cn.stateChanged.connect(lambda:self.chk_statues(self.spinBox_ipac_cn))
        self.checkBox_ipac_contour.stateChanged.connect(lambda:self.chk_statues(self.comboBox_ipac_cnt_color))
        self.checkBox_ipac_index.stateChanged.connect(lambda:self.chk_statues(self.comboBox_ipac_ind_color))
        self.btn_ipac_stack.clicked.connect(self.export_stack_ipac)
        self.btn_ipac_save.clicked.connect(self.ipac_save_rtdc)
        self.checkBox_iacs_cl.stateChanged.connect(lambda:self.chk_statues(self.spinBox_iacs_cl_min))
        self.checkBox_iacs_cl.stateChanged.connect(lambda:self.chk_statues(self.spinBox_iacs_cl_max))
        self.checkBox_iacs_ca.stateChanged.connect(lambda:self.chk_statues(self.spinBox_iacs_ca_min))
        self.checkBox_iacs_ca.stateChanged.connect(lambda:self.chk_statues(self.spinBox_iacs_ca_max))
        self.checkBox_iacs_cn.stateChanged.connect(lambda:self.chk_statues(self.spinBox_iacs_cn))
        self.checkBox_iacs_contour.stateChanged.connect(lambda:self.chk_statues(self.comboBox_iacs_cnt_color))
        self.checkBox_iacs_index.stateChanged.connect(lambda:self.chk_statues(self.comboBox_iacs_ind_color))
        self.btn_iacs_tiles.clicked.connect(self.export_tiles)
        self.btn_iacs_stack.clicked.connect(self.export_stack)
        self.btn_iacs_save.clicked.connect(self.iacs_save_rtdc)
        self.btn_aid_classify.clicked.connect(self.run_classify)
        self.btn_aid_load_model.clicked.connect(self.aid_load_model)
        self.btn_aid_add_rtdc.clicked.connect(self.aid_add_rtdc)
        
# =============================================================================
#         table_ipac
# =============================================================================
        self.table_ipac.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        header_labels = ["TF", "Load", "File Path" , "Shape", "Del"]
        self.table_ipac.setHorizontalHeaderLabels(header_labels) 
        header = self.table_ipac.horizontalHeader()
        for i in [0,1,3,4]:#,2,3,4]:#
            header.setSectionResizeMode(i,QtWidgets.QHeaderView.ResizeToContents)
            #header.setSectionResizeMode(i, QtWidgets.QHeaderView.Interactive) 
            #header.setSectionResizeMode(i,QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        
        self.table_ipac.setAcceptDrops(True)
        self.table_ipac.setDragEnabled(True)
        
        self.table_ipac.dropped.connect(self.dataDropped_ipac)
        self.table_ipac.horizontalHeader().sectionClicked.connect(self.select_all)
        
# =============================================================================
#         table_iacs
# =============================================================================
        self.table_iacs.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        header_labels = ["TF", "Load", "File Path" , "Shape", "Del"]
        self.table_iacs.setHorizontalHeaderLabels(header_labels) 
        header = self.table_iacs.horizontalHeader()
        for i in [0,1,3,4]:#range(len(header_labels)):
            header.setSectionResizeMode(i,QtWidgets.QHeaderView.ResizeToContents)
            #header.setSectionResizeMode(i, QtWidgets.QHeaderView.Interactive) 
            #header.setSectionResizeMode(i,QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        
        self.table_iacs.setAcceptDrops(True)
        self.table_iacs.setDragEnabled(True)
        
        self.table_iacs.dropped.connect(self.dataDropped_iacs)
        self.table_iacs.horizontalHeader().sectionClicked.connect(self.select_all)
        
# =============================================================================
#         table_aid_files
# =============================================================================
        self.table_aid_files.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        header_labels = ["TF", "Load", "File Path" , "Del"]
        self.table_aid_files.setHorizontalHeaderLabels(header_labels) 
        header = self.table_aid_files.horizontalHeader()
        for i in [0,1,3]:
            header.setSectionResizeMode(i,QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)

        self.table_aid_files.setAcceptDrops(True)
        self.table_aid_files.setDragEnabled(True)
        self.table_aid_files.dropped.connect(self.dataDropped_aid)
        
        
        self.table_aid_files.horizontalHeader().sectionClicked.connect(self.select_all_aid)
        
# =============================================================================
#       table_aid_analy
# =============================================================================
        self.table_aid_analy.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        header_labels = ["Load","Class", "Name", "Nr.events" , "% of events"]
        self.table_aid_analy.setHorizontalHeaderLabels(header_labels) 
        header = self.table_aid_analy.horizontalHeader()
        for i in [0,1,3,4]:
            header.setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
        for i in [2]:
            header.setSectionResizeMode(i, QtWidgets.QHeaderView.Stretch)
        
        self.table_aid_analy.horizontalHeader().sectionClicked.connect(self.send_to_napari_all_class)

# =============================================================================
#     table_dise_class
# =============================================================================
        self.table_dise_class.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        header_labels = ["Disease","Probability"]
        self.table_dise_class.setHorizontalHeaderLabels(header_labels)
        header = self.table_dise_class.horizontalHeader()
        for i in [0,1]:
            header.setSectionResizeMode(i, QtWidgets.QHeaderView.Stretch)
        

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.groupBox_ipac.setTitle(_translate("Form", "Files table"))
        self.label_ipac_pixel.setText(_translate("Form", "Pixel size"))
        self.groupBox_ipac_thresh.setTitle(_translate("Form", "Maunal threshold"))
        self.label_ipac_noise.setText(_translate("Form", "Noisel level"))
        self.groupBox_ipac_contour.setTitle(_translate("Form", "Contour options"))
        self.checkBox_ipac_ca.setText(_translate("Form", "Contour area"))
        self.label_ipac_min.setText(_translate("Form", "Min"))
        self.checkBox_ipac_cn.setText(_translate("Form", "N of contours"))
        self.label_ipac_max.setText(_translate("Form", "Max"))
        self.checkBox_ipac_cl.setText(_translate("Form", "Contour length"))
        self.groupBox_ipac_preview.setTitle(_translate("Form", "Preview options"))
        self.checkBox_ipac_contour.setText(_translate("Form", "Show contours"))
        self.comboBox_ipac_cnt_color.setItemText(0, _translate("Form", "Green"))
        self.comboBox_ipac_cnt_color.setItemText(1, _translate("Form", "Aqua"))
        self.comboBox_ipac_cnt_color.setItemText(2, _translate("Form", "Red"))
        self.comboBox_ipac_cnt_color.setItemText(3, _translate("Form", "Black"))
        self.comboBox_ipac_cnt_color.setItemText(4, _translate("Form", "White"))
        self.checkBox_ipac_index.setText(_translate("Form", "Show index"))
        self.comboBox_ipac_ind_color.setItemText(0, _translate("Form", "White"))
        self.comboBox_ipac_ind_color.setItemText(1, _translate("Form", "Black"))
        self.comboBox_ipac_ind_color.setItemText(2, _translate("Form", "Green"))
        self.comboBox_ipac_ind_color.setItemText(3, _translate("Form", "Red"))
        self.comboBox_ipac_ind_color.setItemText(4, _translate("Form", "Aqua"))
        self.btn_ipac_save.setText(_translate("Form", "Save .rtdc"))
        self.btn_ipac_stack.setText(_translate("Form", "Preview stack"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_ipac), _translate("Form", "iPAC"))
        self.groupBox_iacs.setTitle(_translate("Form", "Files table"))
        self.label.setText(_translate("Form", "Number of channels"))
        self.groupBox_colormap.setTitle(_translate("Form", "Colormap"))
        self.label_channel0.setText(_translate("Form", "Channel 0"))
        self.comboBox_iacs_ch0.setItemText(0, _translate("Form", "Red"))
        self.comboBox_iacs_ch0.setItemText(1, _translate("Form", "Green"))
        self.comboBox_iacs_ch0.setItemText(2, _translate("Form", "Aqua"))
        self.label_channel1.setText(_translate("Form", "Channel 1"))
        self.comboBox_iacs_ch1.setItemText(0, _translate("Form", "Green"))
        self.comboBox_iacs_ch1.setItemText(1, _translate("Form", "Aqua"))
        self.comboBox_iacs_ch1.setItemText(2, _translate("Form", "Red"))
        self.label_iacs_pixel.setText(_translate("Form", "Pixel size"))
        self.groupBox_iacs_contour.setTitle(_translate("Form", "Contour options"))
        self.label_iacs_min.setText(_translate("Form", "Min"))
        self.label_iacs_max.setText(_translate("Form", "Max"))
        self.checkBox_iacs_cl.setText(_translate("Form", "Contour length"))
        self.checkBox_iacs_ca.setText(_translate("Form", "Contour area"))
        self.checkBox_iacs_cn.setText(_translate("Form", "N of contours"))
        self.groupBox_iacs_preview_opt.setTitle(_translate("Form", "Preview options"))
        self.checkBox_iacs_index.setText(_translate("Form", "Show grid index"))
        self.comboBox_iacs_ind_color.setItemText(0, _translate("Form", "White"))
        self.comboBox_iacs_ind_color.setItemText(1, _translate("Form", "Black"))
        self.comboBox_iacs_ind_color.setItemText(2, _translate("Form", "Green"))
        self.comboBox_iacs_ind_color.setItemText(3, _translate("Form", "Red"))
        self.comboBox_iacs_ind_color.setItemText(4, _translate("Form", "Aqua"))
        self.checkBox_iacs_contour.setText(_translate("Form", "Show contours"))
        self.comboBox_iacs_cnt_color.setItemText(0, _translate("Form", "Green"))
        self.comboBox_iacs_cnt_color.setItemText(1, _translate("Form", "Aqua"))
        self.comboBox_iacs_cnt_color.setItemText(2, _translate("Form", "Red"))
        self.comboBox_iacs_cnt_color.setItemText(3, _translate("Form", "Black"))
        self.comboBox_iacs_cnt_color.setItemText(4, _translate("Form", "White"))
        self.btn_iacs_save.setText(_translate("Form", "Save .rtdc"))
        self.btn_iacs_tiles.setText(_translate("Form", "Preview tiles"))
        self.btn_iacs_stack.setText(_translate("Form", "Preview stack"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_iacs), _translate("Form", "IACS"))
        
        self.table_aid_files.setToolTip(_translate("Form", "Drag and drop .rtdc or .bin files here."))
        self.lineEdit_aid_model_path.setToolTip(_translate("Form", "Drag and drop the model folder containing the .pb file and meta.xlsx file here."))
        self.groupBox_aid_files.setTitle(_translate("Form", "Files table"))
        self.btn_aid_load_model.setText(_translate("Form", "Choose model"))
        self.btn_aid_load_model.setToolTip(_translate("Form", "Choose the model folder containing the .pb file and meta.xlsx file."))
        self.btn_aid_classify.setText(_translate("Form", "Classify"))
        self.groupBox_aid_analy.setTitle(_translate("Form", "Phenotype classification"))
        self.groupBox_dise_class.setTitle(_translate("Form", "Disease classification"))
        self.btn_aid_add_rtdc.setText(_translate("Form", "Add classification to .rtdc file"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_aid), _translate("Form", "AID classif."))


    def chk_statues(self, spinbox):
        if self.sender().isChecked():
            spinbox.setEnabled(True)
        else:
            spinbox.setEnabled(False)


    def select_all(self,col):
        """
        Check/Uncheck items on table_dragdrop
        """
        parent = self.sender().parent()
        if col == 0:
            rows = range(parent.rowCount()) #Number of rows of the table
            
            tableitems = [parent.item(row, col) for row in rows]
            
            checkStates = [tableitem.checkState() for tableitem in tableitems]
            checked = [state==QtCore.Qt.Checked for state in checkStates]
            if set(checked)=={True}:#all are checked!
                #Uncheck all!
                for tableitem in tableitems:
                    tableitem.setCheckState(QtCore.Qt.Unchecked)
            else:#otherwise check all   
                for tableitem in tableitems:
                    tableitem.setCheckState(QtCore.Qt.Checked)
                    
        if col == 1: #load all
            children= parent.findChildren(QtWidgets.QPushButton,"btn_load")
            for button in children:
                button.click() #click load button

        if col == 4: #delete all
            for i in range(parent.rowCount()):
                parent.removeRow(0)
        else:
            return
        
  
    def delete_item(self,item):
        """
        delete table item and corresponding layers
        """
        buttonClicked = self.sender()
        table = buttonClicked.parent().parent()

        index = table.indexAt(buttonClicked.pos())
        rowPosition = index.row()
        table.removeRow(rowPosition) #remove table item


    def select_layers(self):
        "select all raw date layer"
        layers = []
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                name = layer.name
                if "stack" in name or "tiles" in name:  #find "stack" or "tiles" in filename
                    pass
                else:
                    layers.append(layer) # return raw data
        
        return layers
                    
                
        
# =============================================================================
# IACS
# =============================================================================

    def dataDropped_iacs(self, l):
        
        #Iterate over l and check if it is a folder or a file (directory)    
        isfile = [os.path.isfile(str(url)) for url in l]
        isfolder = [os.path.isdir(str(url)) for url in l]

        #####################For folders with images:##########################            
        #where are folders?
        ind_true = np.where(np.array(isfolder)==True)[0]
        foldernames = list(np.array(l)[ind_true]) #select the indices that are valid
        #On mac, there is a trailing / in case of folders; remove them
        foldernames = [os.path.normpath(url) for url in foldernames]

        basename = [os.path.basename(f) for f in foldernames]
        #Look quickly inside the folders and ask the user if he wants to convert
        #to .rtdc (might take a while!)
        if len(foldernames)>0: #User dropped (also) folders (which may contain images)
            url_converted = []
            for url in foldernames:
                #get a list of tiff files inside this directory:
                images = []
                for root, dirs, files in os.walk(url):
                    for file in files:
                        if file.endswith(".png"):
                            if not "mask" in file:
                                url_converted.append(os.path.join(root, file)) 
                                  
            self.dataDropped_iacs(url_converted)

        #####################For .png files:##################################            
        #where are files?
        ind_true = np.where(np.array(isfile)==True)[0]
        filenames = list(np.array(l)[ind_true]) #select the indices that are valid
        filenames = [x for x in filenames if x.endswith(".png")]
        fileinfo = []
        for png_path in filenames:
            png = cv2.imread(png_path, -1)
            shape = png.shape
            
            fileinfo.append({"png_path":png_path,"shape":shape})
            #print(fileinfo)
                
        for rowNumber in range(len(fileinfo)):#for url in l:
            url = fileinfo[rowNumber]["png_path"]
            shape = str(fileinfo[rowNumber]["shape"])

            #add to table
            rowPosition = self.table_iacs.rowCount()
            self.table_iacs.insertRow(rowPosition)

            columnPosition = 0
            #for each item, also create 2 checkboxes (train/valid)
            item = QtWidgets.QTableWidgetItem()#("item {0} {1}".format(rowNumber, columnNumber))
            item.setFlags( QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled  )
            item.setCheckState(QtCore.Qt.Unchecked)
            self.table_iacs.setItem(rowPosition, columnPosition, item)

            columnPosition = 1
            #Place a button which allows to send to napari for viewing
            btn = QtWidgets.QPushButton(self.table_iacs)
            btn.setObjectName("btn_load")
            btn.setMinimumSize(QtCore.QSize(30, 30))
            btn.setMaximumSize(QtCore.QSize(100, 100))
            btn.clicked.connect(self.send_to_napari_iacs)
            icon_path = os.path.join(dir_root,"art","eye.png")
            btn.setIcon(QtGui.QIcon(str(icon_path)))
            self.table_iacs.setCellWidget(rowPosition, columnPosition, btn) 
            self.table_iacs.resizeRowsToContents()

            columnPosition = 2
            line = QtWidgets.QTableWidgetItem(str(url)) 
            line.setFlags( QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.table_iacs.setItem(rowPosition,columnPosition, line) 
            
            
            columnPosition = 3
            line = QtWidgets.QLabel(self.table_iacs)
            line.setText(shape)
            line.setDisabled(True)
            line.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.table_iacs.setCellWidget(rowPosition, columnPosition, line) 
            
            columnPosition = 4
            #Place a button which allows to send to napari for viewing
            btn_delete = QtWidgets.QPushButton(self.table_iacs)
            
            btn_delete.setMinimumSize(QtCore.QSize(30, 30))
            btn_delete.setMaximumSize(QtCore.QSize(100, 100))
            icon_path = os.path.join(dir_root,"art","delete.png")
            #print("icon_path:", icon_path)
            btn_delete.setIcon(QtGui.QIcon("/Users/nana/iacs_ipac_reader/src/iacs_ipac_reader/art/delete.png"))
            self.table_iacs.setCellWidget(rowPosition, columnPosition, btn_delete) 
            self.table_iacs.resizeRowsToContents()
            btn_delete.clicked.connect(self.delete_item)
        
    
    def send_to_napari_iacs(self,item):
        buttonClicked = self.sender()
        index = self.table_iacs.indexAt(buttonClicked.pos())
        rowPosition = index.row()
    
        path_image = self.table_iacs.item(rowPosition, 2).text()
        image = cv2.imread(path_image,-1)
        image,_ = uint16_2_unit8(image)
        name = os.path.basename(path_image)
            
        new_layer = self.viewer.add_image(image,name=name)

    
    def read_image_iacs_1ch(self,layer):
# =============================================================================
#         input parameters
# =============================================================================
        filter_len = self.checkBox_iacs_cl.isChecked()
        len_min = self.spinBox_iacs_ca_min.value()
        len_max = self.spinBox_iacs_cl_max.value()
        filter_area = self.checkBox_iacs_ca.isChecked()
        area_min = self.spinBox_iacs_ca_min.value()
        area_max = self.spinBox_iacs_ca_max.value()
        filter_n = self.checkBox_iacs_cn.isChecked()
        nr_contours = self.spinBox_iacs_cn.value()
        cnt_color = get_color(self.comboBox_iacs_cnt_color.currentText())
        ind_color = get_color(self.comboBox_iacs_ind_color.currentText())
# ============================================================================
        image = layer.data
        img_list = tiled_2_list(image)
        contours_images_list = []
        for index in range(len(img_list)):
            img = img_list[index]
            img, factor = uint16_2_unit8(img)
            img = vstripes_removal(img)

            contours_,masks_ = get_masks_iacs(img, filter_len, len_min, len_max, 
                                                   filter_area, area_min, area_max, 
                                                   filter_n, nr_contours)
            
            ### draw contours and index ###
            mask = np.zeros((100,88,4),dtype=np.uint8) 
            if self.checkBox_iacs_contour.isChecked(): #draw contour
                cv2.drawContours(mask, contours_, -1, cnt_color, 1) 
                
            if self.checkBox_iacs_index.isChecked(): #draw index
                cv2.rectangle(mask,(0,0),(87,99),ind_color, 1)          
                cv2.putText(mask,str(index),(7,15),cv2.FONT_HERSHEY_DUPLEX, 0.4, ind_color,1)
            
            contours_images_list.append(mask)
            
        return contours_images_list


    def read_image_iacs_2ch(self,layer_ch0,layer_ch1):
        """
        Prepare a set of imges for napari.

        Parameters
        ----------
        layer_ch0 : napari.layer
            
        layer_ch1 : napari.layer
            

        Returns
        -------
        contours_images_list

        """
# =============================================================================
#         Input parameters
# =============================================================================
        filter_len = self.checkBox_iacs_cl.isChecked()
        len_min = self.spinBox_iacs_ca_min.value()
        len_max = self.spinBox_iacs_cl_max.value()
        filter_area = self.checkBox_iacs_ca.isChecked()
        area_min = self.spinBox_iacs_ca_min.value()
        area_max = self.spinBox_iacs_ca_max.value()
        filter_n = self.checkBox_iacs_cn.isChecked()
        nr_contours = self.spinBox_iacs_cn.value()
        cnt_color = get_color(self.comboBox_iacs_cnt_color.currentText())
        ind_color = get_color(self.comboBox_iacs_ind_color.currentText())
        
        ch0_color = self.comboBox_iacs_ch0.currentText()
        ch1_color = self.comboBox_iacs_ch1.currentText()
# =============================================================================

        tiled_img_ch0 = layer_ch0.data
        images_ch0 = tiled_2_list(tiled_img_ch0)
        tiled_img_ch1 = layer_ch1.data
        images_ch1 = tiled_2_list(tiled_img_ch1)
        
        contours_images_list = []
        colormap_images_list = []
        for i in range(len(images_ch0)):
            image_ch0 = images_ch0[i]
            image_ch1 = images_ch1[i]
            
            image_ch0,factor0 = uint16_2_unit8(image_ch0) 
            image_ch1,factor1 = uint16_2_unit8(image_ch1) 
            image_ch0 = vstripes_removal(image_ch0)
            image_ch1 = vstripes_removal(image_ch1)

            img_sup = cv2.add(image_ch0,image_ch1)
            contours_,masks_ = get_masks_iacs(img_sup, filter_len, len_min, len_max, 
                                                   filter_area, area_min, area_max, 
                                                   filter_n, nr_contours)
            
            #### draw contours and index ####
            trans_mask = np.zeros((100,88,4),dtype=np.uint8) 
            if self.checkBox_iacs_contour.isChecked(): #draw contour
                cv2.drawContours(trans_mask, contours_, -1, cnt_color, 1) 
                
            if self.checkBox_iacs_index.isChecked(): #draw index
                cv2.rectangle(trans_mask,(0,0),(87,99),ind_color, 1)          
                cv2.putText(trans_mask,str(i),(7,15),cv2.FONT_HERSHEY_DUPLEX, 0.4, ind_color,1)
            
            contours_images_list.append(trans_mask)
            
            if self.groupBox_colormap.isChecked():
                image_ch0_color = add_colormap(image_ch0,ch0_color)
                image_ch1_color = add_colormap(image_ch1,ch1_color)
                img_sup = cv2.add(image_ch0_color,image_ch1_color)
            
            colormap_images_list.append(img_sup)
            
            
        return contours_images_list, colormap_images_list
    
    
    def export_stack(self):   
        n_channel = self.spinBox_n_channel.value()
        if n_channel == 1:
            for layer in self.select_layers():
                index = layer.name.split(".")[0]
                cnt_img_list = self.read_image_iacs_1ch(layer)                
                img_stack = np.array(cnt_img_list)
                self.viewer.add_image(img_stack, name=index+"_cnt_stack")
        
        if n_channel == 2:
            ## split channel
            layers_ch0 = [layer for layer in self.select_layers() if layer.name.endswith("CH0.png")]
            layers_ch1 = [layer for layer in self.select_layers() if layer.name.endswith("CH1.png")]
            
            ### if find corresponding number of ch0 and ch1
            if len(layers_ch0) > len(layers_ch1):
                print("Could not find ch1 corresponding to ch0")
            if len(layers_ch0) < len(layers_ch1):
                print("Could not find ch0 corresponding to ch1")
            else:
                for i in range(len(layers_ch0)): #for every set oh ch0 and ch1
                    layer_ch0 = layers_ch0[i]
                    layer_ch1 = layers_ch1[i]
                    index = layer_ch0.name.split("_")[0]
                    
                    cnt_img_list,colormap_images_list = self.read_image_iacs_2ch(layer_ch0,layer_ch1)
                    img_stack_cnt = np.array(cnt_img_list)
                    img_stack_color = np.array(colormap_images_list)
                    
                    self.viewer.add_image(img_stack_color, name=index+"_color_stack")
                    self.viewer.add_image(img_stack_cnt, name=index+"_cnt_stack")
                    
    
    def export_tiles(self):
        n_channel = self.spinBox_n_channel.value()
        
        if n_channel == 1:
            for layer in self.select_layers():
                index = layer.name.split(".")[0]
                cnt_img_list = self.read_image_iacs_1ch(layer)                
               
                #tile images and show in napari
                rows = [np.column_stack(cnt_img_list[0+c*40:40+c*40]) for c in range(25)]
                column = np.row_stack(rows)
                self.viewer.add_image(column, name = index+"_cnt_tiles") 

        if n_channel == 2:
            ## split channel
            layers_ch0 = [layer for layer in self.select_layers() if layer.name.endswith("CH0.png")]
            layers_ch1 = [layer for layer in self.select_layers() if layer.name.endswith("CH1.png")]
            
            ### if find corresponding number of ch0 and ch1
            if len(layers_ch0) > len(layers_ch1):
                print("Could not find ch1 corresponding to ch0")
            if len(layers_ch0) < len(layers_ch1):
                print("Could not find ch0 corresponding to ch1")
            else:
                for i in range(len(layers_ch0)): #for every set oh ch0 and ch1
                    layer_ch0 = layers_ch0[i]
                    layer_ch1 = layers_ch1[i]
                    index = layer_ch0.name.split("_")[0]
                    
                    cnt_img_list,colormap_images_list = self.read_image_iacs_2ch(layer_ch0,layer_ch1)
        
                    #tile cnt images
                    rows = [np.column_stack(cnt_img_list[0+c*40:40+c*40]) for c in range(25)]
                    tiles_cnt = np.row_stack(rows)
                    #tile colormap images
                    rows = [np.column_stack(colormap_images_list[0+c*40:40+c*40]) for c in range(25)]
                    tiles_colormap = np.row_stack(rows)
                    
                    
                    self.viewer.add_image(tiles_colormap, name = index+"_color_tiles") 
                    self.viewer.add_image(tiles_cnt, name = index+"_cnt_tiles") 
                    
    def iacs_save_rtdc(self):
        n_channel = self.spinBox_n_channel.value()
        if n_channel == 1:
            self.iacs_save_rtdc_1ch()
        if n_channel == 2:
            self.iacs_save_rtdc_2ch()
    
    
    def iacs_save_rtdc_1ch(self):
        
        ###############Parameters##########################
        filter_len = self.checkBox_iacs_cl.isChecked()
        len_min = self.spinBox_iacs_ca_min.value()
        len_max = self.spinBox_iacs_cl_max.value()
    
        filter_area = self.checkBox_iacs_ca.isChecked()
        area_min = self.spinBox_iacs_ca_min.value()
        area_max = self.spinBox_iacs_ca_max.value()
        
        filter_n = self.checkBox_iacs_cn.isChecked()
        nr_contours = self.spinBox_iacs_cn.value()
        
        pixel_size = self.SpinBox_iacs_pixel.value()
        ###############Parameters##########################
        
        rowPosition = self.table_iacs.rowCount()
        ch0_paths = []
        save_paths = []
        
        for i in range(rowPosition):
            path = self.table_iacs.item(i, 2).text()
            path_target = os.path.dirname(path) + os.sep + os.path.basename(path).split(".png")[0] + ".rtdc" #filename of resulting file
            save_paths.append(path_target)
            ch0_paths.append(path)
            
            #check if the file that should be written may already exists (e.g. after runnig script twice)
            
            if os.path.isfile(path_target):
                print("Following file already exists and will be overwritten: "+path_target)
                #delete the file
                os.remove(path_target)
            
            #Initialize lists for all properties
            #images_ch1_save, factor_ch1,
            folder_names,index_orig,images_ch0_save,masks,contours,\
            pos_x,pos_y,size_x,size_y,\
            bright_avg,bright_sd,factor_ch0,\
            area_um,area_orig,area_hull,\
            area_ratio,circularity,inert_ratio_raw = \
            [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]#,[],[]
            
            t1 = time.time()
            #get all images from a large tiled image
            images_ch0 = []
            
            for index in range(len(ch0_paths)):
                ch0_path = ch0_paths[index]
            
                img_ch0 = cv2.imread(ch0_path,-1) #Load one big (tiled) image
                img_ch0 = tiled_2_list(img_ch0) #separate tiled img into individual images
                images_ch0.append(img_ch0)
            
            images_ch0 = np.concatenate(images_ch0)
            images_ch0 = list(images_ch0)
            
            ###############Parameters##########################
            filter_len = self.checkBox_iacs_cl.isChecked()
            len_min = self.spinBox_iacs_ca_min.value()
            len_max = self.spinBox_iacs_cl_max.value()
        
            filter_area = self.checkBox_iacs_ca.isChecked()
            area_min = self.spinBox_iacs_ca_min.value()
            area_max = self.spinBox_iacs_ca_max.value()
            
            filter_n = self.checkBox_iacs_cn.isChecked()
            nr_contours = self.spinBox_iacs_cn.value()
            
            pixel_size = self.SpinBox_iacs_pixel.value()
            ###############Parameters##########################
            
            
            for index in range(len(images_ch0)):
                img_ch0 = images_ch0[index]
                img_ch0,factor0 = uint16_2_unit8(img_ch0)
                img_ch0_bg_removed = vstripes_removal(img_ch0)
                contours_,masks_ = get_masks_iacs(img_ch0_bg_removed, filter_len, len_min, len_max, 
                                                       filter_area, area_min, area_max, 
                                                       filter_n, nr_contours)
                
                
                for contour,mask in zip(contours_,masks_):
                    output = get_boundingbox_features(img_ch0,contour,pixel_size)
            
                    if type(contour)!=np.ndarray or np.isnan(output[0]) or np.isnan(output[1]):
                        folder_names.append(ch0_path)
                        index_orig.append(index)
                        images_ch0_save.append(np.zeros(shape=img_ch0.shape))
                        masks.append(np.zeros(shape=img_ch0.shape))
                        contours.append(np.nan)
                        factor_ch0.append(np.nan)
                        pos_x.append(np.nan)
                        pos_y.append(np.nan)
                        size_x.append(np.nan)
                        size_y.append(np.nan)
                        bright_avg.append(np.nan)
                        bright_sd.append(np.nan)
                        area_orig.append(np.nan)
                        area_hull.append(np.nan)
                        area_um.append(np.nan)
                        area_ratio.append(np.nan)
                        circularity.append(np.nan)
                        inert_ratio_raw.append(np.nan)
            
                    else:
                        folder_names.append(ch0_path)
                        index_orig.append(index)
                        images_ch0_save.append(img_ch0)
                        masks.append(mask)
                        contours.append(contour)
                        factor_ch0.append(factor0)
                        pos_x.append(output[0])
                        pos_y.append(output[1])
                        size_x.append(output[2])
                        size_y.append(output[3])
                        output = get_brightness(img_ch0,mask)
                        bright_avg.append(output["bright_avg"]/factor0)
                        bright_sd.append(output["bright_sd"]/factor0)
                        
                        output = get_contourfeatures(contour, pixel_size)
                        area_orig.append(output["area_orig"])
                        area_hull.append(output["area_hull"])
                        area_um.append(output["area_um"])
                        area_ratio.append(output["area_ratio"])
                        circularity.append(output["circularity"])
                        inert_ratio_raw.append(output["inert_ratio_raw"])
            
            t2 = time.time()
            dt = t2-t1
            print("Required time to compute features: " +str(np.round(dt,2) )+"s ("+str(np.round(dt/len(images_ch0_save)*1000,2) )+"ms per cell)")
            
            
            #remove events where no contours were found (pos_x and pos_y is nan)
            ind_nan = np.isnan(pos_x)
            ind_nan = np.where(ind_nan==False)[0]
            if len(ind_nan)>0:
                index_orig = list(np.array(index_orig)[ind_nan])
                pos_x = list(np.array(pos_x)[ind_nan])
                pos_y = list(np.array(pos_y)[ind_nan])
                images_ch0_save = list(np.array(images_ch0_save)[ind_nan])
                masks = list(np.array(masks)[ind_nan])
                contours = list(np.array(contours)[ind_nan])
                factor_ch0 = list(np.array(factor_ch0)[ind_nan])
                bright_avg = list(np.array(bright_avg)[ind_nan])
                bright_sd = list(np.array(bright_sd)[ind_nan])
                area_orig = list(np.array(area_orig)[ind_nan])
                area_hull = list(np.array(area_hull)[ind_nan])
                area_um = list(np.array(area_um)[ind_nan])
                area_ratio = list(np.array(area_ratio)[ind_nan])
                circularity = list(np.array(circularity)[ind_nan])
                inert_ratio_raw = list(np.array(inert_ratio_raw)[ind_nan])
                
                #Save images and corresponding pos_x and pos_y to an hdf5 file for AIDeveloper
                images_ch0_save = np.r_[images_ch0_save].astype(np.uint8)
                #images_ch1_save = np.r_[images_ch1_save].astype(np.uint8)
                masks = (np.r_[masks]*255).astype(np.uint8)
            
            
                maxshape_1channel = (None, images_ch0_save.shape[1], images_ch0_save.shape[2])
                maxshape_mask = (None, masks.shape[1], masks.shape[2])
                
                #Create hdf5 dataset
                hdf = h5py.File(path_target,'a')
                #events = hdf.require_group("events")
            
                dset = hdf.create_dataset("events/image", data=images_ch0_save, dtype=np.uint8,maxshape=maxshape_1channel,fletcher32=True,chunks=True)
                dset.attrs.create('CLASS', np.string_('IMAGE'))
                dset.attrs.create('IMAGE_VERSION', np.string_('1.2'))
                dset.attrs.create('IMAGE_SUBCLASS', np.string_('IMAGE_GRAYSCALE'))
            
                dset.attrs.create('CLASS', np.string_('IMAGE'))
                dset.attrs.create('IMAGE_VERSION', np.string_('1.2'))
                dset.attrs.create('IMAGE_SUBCLASS', np.string_('IMAGE_GRAYSCALE'))
            
            
                dset = hdf.create_dataset("events/mask", data=masks, dtype=np.uint8,maxshape=maxshape_mask,fletcher32=True,chunks=True)
                dset.attrs.create('CLASS', np.string_('IMAGE'))
                dset.attrs.create('IMAGE_VERSION', np.string_('1.2'))
                dset.attrs.create('IMAGE_SUBCLASS', np.string_('IMAGE_GRAYSCALE'))
            
                hdf.create_dataset("events/index_online", data=index_orig,dtype=np.int32)
                hdf.create_dataset("events/pos_x", data=pos_x, dtype=np.int32)
                hdf.create_dataset("events/pos_y", data=pos_y, dtype=np.int32)
                hdf.create_dataset("events/bright_avg", data=bright_avg, dtype=np.float32)
                hdf.create_dataset("events/bright_sd", data=bright_sd, dtype=np.float32)
                hdf.create_dataset("events/factor_ch0", data=factor_ch0, dtype=np.float32)
            
                hdf.create_dataset("events/circ", data=circularity, dtype=np.float32)
                hdf.create_dataset("events/inert_ratio_raw", data=inert_ratio_raw, dtype=np.float32)
                hdf.create_dataset("events/area_ratio", data=area_ratio, dtype=np.float32)
                hdf.create_dataset("events/area_msd", data=area_orig, dtype=np.float32)
                hdf.create_dataset("events/area_cvx", data=area_hull, dtype=np.float32)       
                hdf.create_dataset("events/area_um", data=area_um, dtype=np.float32)       
                
                #Adjust metadata:
                #"experiment:event count" = Nr. of images
                hdf.attrs["experiment:event count"] = images_ch0_save.shape[0]
                hdf.attrs["imaging:roi size x"] = images_ch0_save.shape[1]
                hdf.attrs["imaging:roi size y"] = images_ch0_save.shape[2]
            
                hdf.attrs["imaging:pixel size"] = pixel_size
                hdf.attrs["experiment:date"] = time.strftime("%Y-%m-%d")
                hdf.attrs["experiment:time"] = time.strftime("%H:%M:%S")
                #hdf.attrs["setup:identifier"] = "iIACS_2.0"
                hdf.close()

        ###################### show messagebox ######## 
        
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("")
        text = "\n".join(save_paths)
        msg.setText("Successfully export .rtdc files in \n" + text)
        
        x = msg.exec_()
            
    
    def iacs_save_rtdc_2ch(self):
        rowPosition = self.table_iacs.rowCount()
        ch0_paths = []
        ch1_paths = []
        save_paths = []
        for i in range(rowPosition):
            path = self.table_iacs.item(i, 2).text()
            if path.endswith("CH0.png"):
                name = path.split("/")[-1]
                ch0_path = path
                ch1_path = path.split("CH0.png")[0]+"CH1.png"
                if os.path.isfile(ch1_path):
                    ch0_paths.append(ch0_path)
                    ch1_paths.append(ch1_path)
                else:
                    print("Could not find corresponding CH1 image for:" + ch0_path)
                    exit()
                    
                #path = self.table_iacs.item(0, 2).text()
                path_target = os.path.dirname(path) + os.sep + os.path.basename(path).split("_CH0.png")[0] + ".rtdc" #filename of resulting file
                
                save_paths.append(path_target)
                #check if the file that should be written may already exists (e.g. after runnig script twice)
                if os.path.isfile(path_target):
                    print("Following file already exists and will be overwritten: "+path_target)
                    #delete the file
                    os.remove(path_target)
                
                #Initialize lists for all properties
                folder_names,index_orig,images_ch0_save,images_ch1_save,masks,contours,\
                pos_x,pos_y,size_x,size_y,\
                bright_avg,bright_sd,factor_ch0,factor_ch1,\
                area_um,area_orig,area_hull,\
                area_ratio,circularity,inert_ratio_raw = \
                [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
                
                t1 = time.time()
                #get all images from a large tiled image
                images_ch0,images_ch1 = [],[]
                
                for index in range(len(ch0_paths)):
                    ch0_path = ch0_paths[index]
                    ch1_path = ch1_paths
                
                    img_ch0 = cv2.imread(ch0_path,-1) #Load one big (tiled) image
                    img_ch0 = tiled_2_list(img_ch0) #separate tiled img into individual images
                    images_ch0.append(img_ch0)
                
                    img_ch1 = cv2.imread(ch1_path,-1)
                    img_ch1 = tiled_2_list(img_ch1) #separate tiled img into individual images
                    images_ch1.append(img_ch1)
                
                images_ch0 = np.concatenate(images_ch0)
                images_ch0 = list(images_ch0)
                images_ch1 = np.concatenate(images_ch1)
                images_ch1 = list(images_ch1)
                
                
                ###############Parameters##########################
                filter_len = self.checkBox_iacs_cl.isChecked()
                len_min = self.spinBox_iacs_ca_min.value()
                len_max = self.spinBox_iacs_cl_max.value()
            
                filter_area = self.checkBox_iacs_ca.isChecked()
                area_min = self.spinBox_iacs_ca_min.value()
                area_max = self.spinBox_iacs_ca_max.value()
                
                filter_n = self.checkBox_iacs_cn.isChecked()
                nr_contours = self.spinBox_iacs_cn.value()
                
                pixel_size = self.SpinBox_iacs_pixel.value()
                ###############Parameters##########################
                
                
                for index in range(len(images_ch0)):
                    img_ch0 = images_ch0[index]
                    img_ch0,factor0 = uint16_2_unit8(img_ch0)
                    img_ch0_bg_removed = vstripes_removal(img_ch0)
                
                    img_ch1 = images_ch1[index]    
                    img_ch1,factor1 = uint16_2_unit8(img_ch1)
                    img_ch1_bg_removed = vstripes_removal(img_ch1)
                    
                    #Create a superposition
                    img_sup = cv2.add(img_ch0_bg_removed,img_ch1_bg_removed)
                    #get list of conoturs and masks in image using the superposition
                    
                    contours_,masks_ = get_masks_iacs(img_sup, filter_len, len_min, len_max, 
                                                           filter_area, area_min, area_max, 
                                                           filter_n, nr_contours)
                    
                    
                    for contour,mask in zip(contours_,masks_):
                        output = get_boundingbox_features(img_ch0,contour,pixel_size)
                
                        if type(contour)!=np.ndarray or np.isnan(output[0]) or np.isnan(output[1]):
                            folder_names.append(ch0_path)
                            index_orig.append(index)
                            images_ch0_save.append(np.zeros(shape=img_ch0.shape))
                            images_ch1_save.append(np.zeros(shape=img_ch1.shape))
                            masks.append(np.zeros(shape=img_ch0.shape))
                            contours.append(np.nan)
                            factor_ch0.append(np.nan)
                            factor_ch1.append(np.nan)
                            pos_x.append(np.nan)
                            pos_y.append(np.nan)
                            size_x.append(np.nan)
                            size_y.append(np.nan)
                            bright_avg.append(np.nan)
                            bright_sd.append(np.nan)
                            area_orig.append(np.nan)
                            area_hull.append(np.nan)
                            area_um.append(np.nan)
                            area_ratio.append(np.nan)
                            circularity.append(np.nan)
                            inert_ratio_raw.append(np.nan)
                
                        else:
                            folder_names.append(ch0_path)
                            index_orig.append(index)
                            images_ch0_save.append(img_ch0)
                            images_ch1_save.append(img_ch1)
                            masks.append(mask)
                            contours.append(contour)
                            factor_ch0.append(factor0)
                            factor_ch1.append(factor1)
                            pos_x.append(output[0])
                            pos_y.append(output[1])
                            size_x.append(output[2])
                            size_y.append(output[3])
                            output = get_brightness(img_ch0,mask)
                            bright_avg.append(output["bright_avg"]/factor0)
                            bright_sd.append(output["bright_sd"]/factor0)
                            
                            output = get_contourfeatures(contour, pixel_size)
                            area_orig.append(output["area_orig"])
                            area_hull.append(output["area_hull"])
                            area_um.append(output["area_um"])
                            area_ratio.append(output["area_ratio"])
                            circularity.append(output["circularity"])
                            inert_ratio_raw.append(output["inert_ratio_raw"])
                
                t2 = time.time()
                dt = t2-t1
                print("Required time to compute features: " +str(np.round(dt,2) )+"s ("+str(np.round(dt/len(images_ch0_save)*1000,2) )+"ms per cell)")
                
                
                #remove events where no contours were found (pos_x and pos_y is nan)
                ind_nan = np.isnan(pos_x)
                ind_nan = np.where(ind_nan==False)[0]
                if len(ind_nan)>0:
                    index_orig = list(np.array(index_orig)[ind_nan])
                    pos_x = list(np.array(pos_x)[ind_nan])
                    pos_y = list(np.array(pos_y)[ind_nan])
                    images_ch0_save = list(np.array(images_ch0_save)[ind_nan])
                    images_ch1_save = list(np.array(images_ch1_save)[ind_nan])
                    masks = list(np.array(masks)[ind_nan])
                    contours = list(np.array(contours)[ind_nan])
                    factor_ch0 = list(np.array(factor_ch0)[ind_nan])
                    factor_ch1 = list(np.array(factor_ch1)[ind_nan])
                    bright_avg = list(np.array(bright_avg)[ind_nan])
                    bright_sd = list(np.array(bright_sd)[ind_nan])
                    area_orig = list(np.array(area_orig)[ind_nan])
                    area_hull = list(np.array(area_hull)[ind_nan])
                    area_um = list(np.array(area_um)[ind_nan])
                    area_ratio = list(np.array(area_ratio)[ind_nan])
                    circularity = list(np.array(circularity)[ind_nan])
                    inert_ratio_raw = list(np.array(inert_ratio_raw)[ind_nan])
                    
                    #Save images and corresponding pos_x and pos_y to an hdf5 file for AIDeveloper
                    images_ch0_save = np.r_[images_ch0_save].astype(np.uint8)
                    images_ch1_save = np.r_[images_ch1_save].astype(np.uint8)
                    masks = (np.r_[masks]*255).astype(np.uint8)
                
                    # #Create RGB images
                    #zero_green = np.ones(shape=images_ch0_save.shape)+100
                    images_3channels = np.stack([images_ch0_save,images_ch1_save,masks],axis=-1)
        # =============================================================================
        #             
        #             #copy the empty Empty.rtdc
        #             shutil.copy("Empty.rtdc",path_target)
        #             
        # =============================================================================
                    maxshape_1channel = (None, images_ch0_save.shape[1], images_ch0_save.shape[2])
                    #maxshape_3channels = (None, images_3channels.shape[1], images_3channels.shape[2],3)
                    maxshape_mask = (None, masks.shape[1], masks.shape[2])
                    
                    #Create hdf5 dataset
                    hdf = h5py.File(path_target,'a')
                    #events = hdf.require_group("events")
                
                    dset = hdf.create_dataset("events/image", data=images_ch0_save, dtype=np.uint8,maxshape=maxshape_1channel,fletcher32=True,chunks=True)
                    dset.attrs.create('CLASS', np.string_('IMAGE'))
                    dset.attrs.create('IMAGE_VERSION', np.string_('1.2'))
                    dset.attrs.create('IMAGE_SUBCLASS', np.string_('IMAGE_GRAYSCALE'))
                
                    # dset = hdf.create_dataset("events/image_ch0", data=images_ch0_save, dtype=np.uint8,maxshape=maxshape_1channel,fletcher32=True,chunks=True)        
                    # dset.attrs.create('CLASS', np.string_('IMAGE'))
                    # dset.attrs.create('IMAGE_VERSION', np.string_('1.2'))
                    # dset.attrs.create('IMAGE_SUBCLASS', np.string_('IMAGE_GRAYSCALE'))
                
                    dset = hdf.create_dataset("events/image_ch1", data=images_ch1_save, dtype=np.uint8,maxshape=maxshape_1channel,fletcher32=True,chunks=True)        
                    dset.attrs.create('CLASS', np.string_('IMAGE'))
                    dset.attrs.create('IMAGE_VERSION', np.string_('1.2'))
                    dset.attrs.create('IMAGE_SUBCLASS', np.string_('IMAGE_GRAYSCALE'))
                
                    #store_mask(h5group=events,name=feat, data=mask, compression="gzip")
                
                    dset = hdf.create_dataset("events/mask", data=masks, dtype=np.uint8,maxshape=maxshape_mask,fletcher32=True,chunks=True)
                    dset.attrs.create('CLASS', np.string_('IMAGE'))
                    dset.attrs.create('IMAGE_VERSION', np.string_('1.2'))
                    dset.attrs.create('IMAGE_SUBCLASS', np.string_('IMAGE_GRAYSCALE'))
                    
                    #Contours are omitted (Masks contain the same information
                    # for ii, cc in enumerate(contours):
                    #     hdf.create_dataset("events/contour/"+"{}".format(ii),
                    #                        data=cc.reshape(cc.shape[0],cc.shape[2]),
                    #                        fletcher32=True)
                
                    hdf.create_dataset("events/index_online", data=index_orig,dtype=np.int32)
                    hdf.create_dataset("events/pos_x", data=pos_x, dtype=np.int32)
                    hdf.create_dataset("events/pos_y", data=pos_y, dtype=np.int32)
                    hdf.create_dataset("events/bright_avg", data=bright_avg, dtype=np.float32)
                    hdf.create_dataset("events/bright_sd", data=bright_sd, dtype=np.float32)
                    hdf.create_dataset("events/factor_ch0", data=factor_ch0, dtype=np.float32)
                    hdf.create_dataset("events/factor_ch1", data=factor_ch1, dtype=np.float32)
                
                    hdf.create_dataset("events/circ", data=circularity, dtype=np.float32)
                    hdf.create_dataset("events/inert_ratio_raw", data=inert_ratio_raw, dtype=np.float32)
                    hdf.create_dataset("events/area_ratio", data=area_ratio, dtype=np.float32)
                    hdf.create_dataset("events/area_msd", data=area_orig, dtype=np.float32)
                    hdf.create_dataset("events/area_cvx", data=area_hull, dtype=np.float32)       
                    hdf.create_dataset("events/area_um", data=area_um, dtype=np.float32)       
                    
                    #Adjust metadata:
                    #"experiment:event count" = Nr. of images
                    hdf.attrs["experiment:event count"] = images_ch0_save.shape[0]
                    #hdf.attrs["experiment:sample"] = sample_name
                    hdf.attrs["imaging:roi size x"] = images_ch0_save.shape[1]
                    hdf.attrs["imaging:roi size y"] = images_ch0_save.shape[2]
                
                    hdf.attrs["imaging:pixel size"] = pixel_size
                    hdf.attrs["experiment:date"] = time.strftime("%Y-%m-%d")
                    hdf.attrs["experiment:time"] = time.strftime("%H:%M:%S")
                    #hdf.attrs["setup:identifier"] = "iIACS_2.0"
                    hdf.close()
    
        ###################### show messagebox ######## 
        
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("")
        text = "\n".join(save_paths)
        msg.setText("Successfully export .rtdc files in \n" + text)
        
        x = msg.exec_()
                
            


# =============================================================================
# iPAC
# =============================================================================
    
    def dataDropped_ipac(self, l):
        #Iterate over l and check if it is a folder or a file (directory)    
        isfile = [os.path.isfile(str(url)) for url in l]
        isfolder = [os.path.isdir(str(url)) for url in l]


        #####################For folders with images:##########################            
        #where are folders?
        ind_true = np.where(np.array(isfolder)==True)[0]
        foldernames = list(np.array(l)[ind_true]) #select the indices that are valid
        #On mac, there is a trailing / in case of folders; remove them
        foldernames = [os.path.normpath(url) for url in foldernames]

        basename = [os.path.basename(f) for f in foldernames]
        #Look quickly inside the folders and ask the user if he wants to convert
        #to .rtdc (might take a while!)
        if len(foldernames)>0: #User dropped (also) folders (which may contain images)
            url_converted = []
            for url in foldernames:
                #get a list of tiff files inside this directory:
                images = []
                for root, dirs, files in os.walk(url):
                    for file in files:
                        if file.endswith(".bin"):
                            url_converted.append(os.path.join(root, file)) 
                                  
            self.dataDropped_ipac(url_converted)

        #####################For .bin files:##################################            
        #where are files?
        ind_true = np.where(np.array(isfile)==True)[0]
        filenames = list(np.array(l)[ind_true]) #select the indices that are valid
        filenames = [x for x in filenames if x.endswith(".bin")]
        fileinfo = []
        for binary_path in filenames:
            binary = np.fromfile(binary_path, dtype='>H')
            n,w,h = binary[1],binary[3],binary[5]
            images = binary[6:].reshape(n,h,w) #reshape to get the images       
            shape = images.shape
            
            fileinfo.append({"binary_path":binary_path,"shape":shape})
            #print(fileinfo)
        
        for rowNumber in range(len(fileinfo)):#for url in l:
            url = fileinfo[rowNumber]["binary_path"]
            shape = str(fileinfo[rowNumber]["shape"])

            #add to table
            rowPosition = self.table_ipac.rowCount()
            self.table_ipac.insertRow(rowPosition)

            columnPosition = 0
            #for each item, also create 2 checkboxes (train/valid)
            item = QtWidgets.QTableWidgetItem()#("item {0} {1}".format(rowNumber, columnNumber))
            item.setFlags( QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled  )
            item.setCheckState(QtCore.Qt.Unchecked)
            self.table_ipac.setItem(rowPosition, columnPosition, item)

            columnPosition = 1
            #Place a button which allows to send to napari for viewing
            btn = QtWidgets.QPushButton(self.table_ipac)
            btn.setObjectName("btn_load")
            btn.setMinimumSize(QtCore.QSize(30, 30))
            btn.setMaximumSize(QtCore.QSize(100, 100))
            btn.clicked.connect(self.send_to_napari_ipac)
            icon_path = os.path.join(dir_root,"art","eye.png")
            btn.setIcon(QtGui.QIcon(str(icon_path)))
            self.table_ipac.setCellWidget(rowPosition, columnPosition, btn) 
            self.table_ipac.resizeRowsToContents()

            columnPosition = 2
            line = QtWidgets.QTableWidgetItem(str(url)) 
            line.setFlags( QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.table_ipac.setItem(rowPosition,columnPosition, line) 
            
            
            columnPosition = 3
            line = QtWidgets.QLabel(self.table_ipac)
            line.setText(shape)
            line.setDisabled(True)
            line.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            self.table_ipac.setCellWidget(rowPosition, columnPosition, line) 
            
            columnPosition = 4
            #Place a button which allows to send to napari for viewing
            btn_delete = QtWidgets.QPushButton(self.table_ipac)
            btn_delete.setMinimumSize(QtCore.QSize(30, 30))
            btn_delete.setMaximumSize(QtCore.QSize(100, 100))
            icon_path = os.path.join(dir_root,"art","delete.png")
            btn_delete.setIcon(QtGui.QIcon(str(icon_path)))
            self.table_ipac.setCellWidget(rowPosition, columnPosition, btn_delete) 
            self.table_ipac.resizeRowsToContents()
            btn_delete.clicked.connect(self.delete_item)


    def send_to_napari_ipac(self, item):
        buttonClicked = self.sender()
        index = self.table_ipac.indexAt(buttonClicked.pos())
        rowPosition = index.row()
    
        path_image = self.table_ipac.item(rowPosition, 2).text()
        path_image = str(path_image)
        #send image to napari 
        binary = np.fromfile(path_image, dtype='>H')
        n,w,h = binary[1],binary[3],binary[5]
        images = binary[6:].reshape(n,h,w)
        name = os.path.basename(path_image)[:-4] #remove .bin
        
        new_layer = self.viewer.add_image(images,name=name)
    
    
    def export_stack_ipac(self):
        #contour_img_list,layer_name = self.read_image_ipac()
        index = self.read_image_ipac()
        for i in range(len(index)):
            contour_img_list,layer_name = index[i]
            img_stack_ipac = np.array(contour_img_list)
            self.viewer.add_image(img_stack_ipac, name =str(layer_name)+"_cnt_stack")
        

    def ipac_save_rtdc(self):
        ###############Parameters##########################
        filter_len = self.checkBox_ipac_cl.isChecked()
        len_min = self.spinBox_ipac_cl_min.value()
        len_max = self.spinBox_ipac_cl_max.value()
        
        filter_area = self.checkBox_ipac_ca.isChecked()
        area_min = self.spinBox_ipac_ca_min.value()
        area_max = self.spinBox_ipac_ca_max.value()
        noise_level = self.doubleSpinBox_ipac_noise.value() 
        
        filter_n = self.checkBox_ipac_cn.isChecked()
        nr_contours = self.spinBox_ipac_cn.value()
        pixel_size = self.ipac_SpinBox_pixel.value()
        
        bg_intensity = 16381
        ###############Parameters##########################
        
        # load data paths
        rowPosition = self.table_ipac.rowCount()
        paths = []
        save_paths = []
        for i in range(rowPosition):
            path = self.table_ipac.item(i, 2).text()
            paths.append(path)
        
        for path in paths:
            path_target = os.path.dirname(path) + os.sep + os.path.basename(path).split(".bin")[0] + ".rtdc" #filename of resulting file
            save_paths.append(path_target)
            #check if the file that should be written may already exists (e.g. after runnig script twice)
            if os.path.isfile(path_target):
                print("Following file already exists and will be overwritten: "+path_target)
                #delete the file
                os.remove(path_target)
            
            
            #load binary
            binary = np.fromfile(path, dtype='>H')
            n,w,h = binary[1],binary[3],binary[5]
            images = binary[6:].reshape(n,h,w)
            
            images = images.astype(np.float32) #for conversion to unit8, first make it float (float32 is sufficient)
            factor = 128/bg_intensity
            images = np.multiply(images, factor)
            images.astype(np.uint8)

            #Cell can be darker and brighter than background->Subtract background and take absolute
            images_abs = images.astype(np.int)-128 #after that, the background is approx 0
            images_abs = abs(images_abs).astype(np.uint8) #absolute. Neccessary for contour finding
            

            #Initialize lists for all properties
            folder_names,index_orig,images_save,masks,contours,\
            pos_x,pos_y,size_x,size_y,\
            bright_avg,bright_sd,factor_ch0,factor_ch1,\
            area_um,area_orig,area_hull,\
            area_ratio,circularity,inert_ratio_raw = \
            [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
            
            t1 = time.time()
            #get all images located in one particular folder
            for img_index in range(len(images)):
                #load image
                image = images[img_index]
                image, factor = uint16_2_unit8(image) 
                image_abs = images_abs[img_index]
                        
                #get list of conoturs and masks in image using the superposition
                contours_,masks_ = get_masks_ipac(image_abs, noise_level,filter_len, len_min, len_max, 
                                                     filter_area, area_min, area_max, filter_n, nr_contours)
                del image_abs
                
                for contour,mask in zip(contours_,masks_):
                    output = get_boundingbox_features(image,contour,pixel_size)
            
                    if type(contour)!=np.ndarray or np.isnan(output[0]) or np.isnan(output[1]):
                        index_orig.append(img_index)
                        images_save.append(np.zeros(shape=image.shape))
                        masks.append(np.zeros(shape=image.shape))
                        contours.append(np.nan)
                        pos_x.append(np.nan)
                        pos_y.append(np.nan)
                        size_x.append(np.nan)
                        size_y.append(np.nan)
            
                        bright_avg.append(np.nan)
                        bright_sd.append(np.nan)
                        area_orig.append(np.nan)
                        area_hull.append(np.nan)
                        area_um.append(np.nan)
                        area_ratio.append(np.nan)
                        circularity.append(np.nan)
                        inert_ratio_raw.append(np.nan)
            
                    else:
                        index_orig.append(img_index)
                        images_save.append(image)
                        masks.append(mask)
                        contours.append(contour)
                        pos_x.append(output[0])
                        pos_y.append(output[1])
                        size_x.append(output[2])
                        size_y.append(output[3])
                        
                        output = get_brightness(image,mask)
                        bright_avg.append(output["bright_avg"])
                        bright_sd.append(output["bright_sd"])
                        
                        output = get_contourfeatures(contour,pixel_size)
                        area_orig.append(output["area_orig"])
                        area_hull.append(output["area_hull"])
                        area_um.append(output["area_um"])
                        area_ratio.append(output["area_ratio"])
                        circularity.append(output["circularity"])
                        inert_ratio_raw.append(output["inert_ratio_raw"])
            
            t2 = time.time()
            dt = t2-t1
            print("Required time to compute features: " +str(np.round(dt,2) )+"s ("+str(np.round(dt/len(images_save)*1000,2) )+"ms per cell)")
            
            
            #remove events where no contours were found (pos_x and pos_y is nan)
            ind_nan = np.isnan(pos_x)
            ind_nan = np.where(ind_nan==False)[0]
            if len(ind_nan)>0:
                index_orig = list(np.array(index_orig)[ind_nan])
                pos_x = list(np.array(pos_x)[ind_nan])
                pos_y = list(np.array(pos_y)[ind_nan])
                size_x = list(np.array(size_x)[ind_nan])
                size_y = list(np.array(size_y)[ind_nan])
            
                images_save = list(np.array(images_save)[ind_nan])
                masks = list(np.array(masks)[ind_nan])
                contours = list(np.array(contours)[ind_nan])
                bright_avg = list(np.array(bright_avg)[ind_nan])
                bright_sd = list(np.array(bright_sd)[ind_nan])
                area_orig = list(np.array(area_orig)[ind_nan])
                area_hull = list(np.array(area_hull)[ind_nan])
                area_um = list(np.array(area_um)[ind_nan])
                area_ratio = list(np.array(area_ratio)[ind_nan])
                circularity = list(np.array(circularity)[ind_nan])
                inert_ratio_raw = list(np.array(inert_ratio_raw)[ind_nan])
                
                #Save images and corresponding pos_x and pos_y to an hdf5 file for AIDeveloper
                images_save = np.r_[images_save]
                masks = np.r_[masks]
                
                maxshape_img = (None, images_save.shape[1], images_save.shape[2])
                maxshape_mask = (None, masks.shape[1], masks.shape[2])
                
                #Create rtdc_dataset; valid feature names can be found via dclab.dfn.feature_names
                hdf = h5py.File(path_target,'a')
                dset = hdf.create_dataset("events/image", data=images_save, dtype=np.uint8,maxshape=maxshape_img,fletcher32=True,chunks=True)
                dset.attrs.create('CLASS', np.string_('IMAGE'))
                dset.attrs.create('IMAGE_VERSION', np.string_('1.2'))
                dset.attrs.create('IMAGE_SUBCLASS', np.string_('IMAGE_GRAYSCALE'))
            
                dset = hdf.create_dataset("events/mask", data=masks, dtype=np.uint8,maxshape=maxshape_mask,fletcher32=True,chunks=True)
                dset.attrs.create('CLASS', np.string_('IMAGE'))
                dset.attrs.create('IMAGE_VERSION', np.string_('1.2'))
                dset.attrs.create('IMAGE_SUBCLASS', np.string_('IMAGE_GRAYSCALE'))
                
                hdf.create_dataset("events/index_online", data=index_orig,dtype=np.int32)
                hdf.create_dataset("events/pos_x", data=pos_x, dtype=np.int32)
                hdf.create_dataset("events/pos_y", data=pos_y, dtype=np.int32)
                hdf.create_dataset("events/size_x", data=size_x, dtype=np.int32)
                hdf.create_dataset("events/size_y", data=size_y, dtype=np.int32)
                
                hdf.create_dataset("events/bright_avg", data=bright_avg, dtype=np.float32)
                hdf.create_dataset("events/bright_sd", data=bright_sd, dtype=np.float32)
            
                hdf.create_dataset("events/circ", data=circularity, dtype=np.float32)
                hdf.create_dataset("events/inert_ratio_raw", data=inert_ratio_raw, dtype=np.float32)
                hdf.create_dataset("events/area_ratio", data=area_ratio, dtype=np.float32)
                hdf.create_dataset("events/area_msd", data=area_orig, dtype=np.float32)
                hdf.create_dataset("events/area_cvx", data=area_hull, dtype=np.float32)       
                hdf.create_dataset("events/area_um", data=area_um, dtype=np.float32)       
            
                #Adjust metadata:
# =============================================================================
#                 #"experiment:event count" = Nr. of images
#                 hdf.attrs["experiment:run index"] = m_number
#                 m_number += 1 #increase measurement number 
# =============================================================================
                hdf.attrs["experiment:event count"] = images_save.shape[0]
                #hdf.attrs["experiment:sample"] = condition #Blood draw date
                hdf.attrs["imaging:pixel size"] = pixel_size
                hdf.attrs["experiment:date"] = time.strftime("%Y-%m-%d")
                hdf.attrs["experiment:time"] = time.strftime("%H:%M:%S")

                hdf.attrs["imaging:roi size x"] = images_save.shape[2]
                hdf.attrs["imaging:roi size y"] = images_save.shape[1]
                hdf.attrs["online_contour:bin kernel"] = 2*int(3)+1
                hdf.attrs["online_contour:bin threshold"] = noise_level

                hdf.close()
                
        ###################### show messagebox ######## 
        
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("")
        text = "\n".join(save_paths)
        msg.setText("Successfully export .rtdc files in \n" + text)
        
        x = msg.exec_()
        

   

# =============================================================================
# AID 
# =============================================================================

    def dataDropped_aid(self, l):
        #Iterate over l and check if it is a folder or a file (directory)    
        isfile = [os.path.isfile(str(url)) for url in l]
        isfolder = [os.path.isdir(str(url)) for url in l]


        #####################For folders with rtdc or bin:##########################            
        #where are folders?
        ind_true = np.where(np.array(isfolder)==True)[0]
        foldernames = list(np.array(l)[ind_true]) #select the indices that are valid
        #On mac, there is a trailing / in case of folders; remove them
        foldernames = [os.path.normpath(url) for url in foldernames]

        basename = [os.path.basename(f) for f in foldernames]
        #Look quickly inside the folders and ask the user if he wants to convert
        #to .rtdc (might take a while!)
        if len(foldernames)>0: #User dropped (also) folders (which may contain images)
            url_converted = []
            for url in foldernames:
                #get a list of tiff files inside this directory:
                images = []
                for root, dirs, files in os.walk(url):
                    for file in files:
                        if file.endswith(".rtdc"):
                            url_converted.append(os.path.join(root, file)) 
                        if file.endswith(".bin"):
                            url_converted.append(os.path.join(root, file)) 
                                  
            self.dataDropped_aid(url_converted)

        #####################For .rtdc or .bin files:##################################            
        #where are files?
        ind_true = np.where(np.array(isfile)==True)[0]
        filenames = list(np.array(l)[ind_true]) #select the indices that are valid
        filenames = [x for x in filenames if x.endswith(".rtdc") or x.endswith(".bin")]
        
        fileinfo = []
        for file_path in filenames:
                fileinfo.append({"file_path":file_path})
        
        
        for rowNumber in range(len(fileinfo)):#for url in l:
            url = fileinfo[rowNumber]["file_path"]

            #add to table
            rowPosition = self.table_aid_files.rowCount()
            self.table_aid_files.insertRow(rowPosition)

            columnPosition = 0
            #for each item, also create 2 checkboxes (train/valid)
            item = QtWidgets.QTableWidgetItem()#("item {0} {1}".format(rowNumber, columnNumber))
            item.setFlags( QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled  )
            item.setCheckState(QtCore.Qt.Unchecked)
            self.table_aid_files.setItem(rowPosition, columnPosition, item)

            columnPosition = 1
            #Place a button which allows to send to napari for viewing
            btn = QtWidgets.QPushButton(self.table_aid_files)
            btn.setObjectName("btn_load")
            btn.setMinimumSize(QtCore.QSize(30, 30))
            btn.setMaximumSize(QtCore.QSize(100, 100))
            btn.clicked.connect(self.load_rtdc_images)
            icon_path = os.path.join(dir_root,"art","eye.png")
            btn.setIcon(QtGui.QIcon(str(icon_path)))
            self.table_aid_files.setCellWidget(rowPosition, columnPosition, btn) 
            self.table_aid_files.resizeRowsToContents()

            columnPosition = 2
            line = QtWidgets.QTableWidgetItem(str(url)) 
            line.setFlags( QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.table_aid_files.setItem(rowPosition,columnPosition, line) 
            
            columnPosition = 3
            btn_delete = QtWidgets.QPushButton(self.table_aid_files)
            
            btn_delete.setMinimumSize(QtCore.QSize(30, 30))
            btn_delete.setMaximumSize(QtCore.QSize(100, 100))
            icon_path = os.path.join(dir_root,"art","delete.png")
            btn_delete.setIcon(QtGui.QIcon(str(icon_path)))
            self.table_aid_files.setCellWidget(rowPosition, columnPosition, btn_delete) 
            self.table_aid_files.resizeRowsToContents()
            btn_delete.clicked.connect(self.delete_item)        

    def select_all_aid(self,col):
        """
        Check/Uncheck items on table_dragdrop
        """
        parent = self.sender().parent()
        if col == 0:
            rows = range(parent.rowCount()) #Number of rows of the table
            tableitems = [parent.item(row, col) for row in rows]
            
            checkStates = [tableitem.checkState() for tableitem in tableitems]
            checked = [state==QtCore.Qt.Checked for state in checkStates]
            if set(checked)=={True}:#all are checked!
                #Uncheck all!
                for tableitem in tableitems:
                    tableitem.setCheckState(QtCore.Qt.Unchecked)
            else:#otherwise check all   
                for tableitem in tableitems:
                    tableitem.setCheckState(QtCore.Qt.Checked)
                    
        if col == 1: #load all
            children= parent.findChildren(QtWidgets.QPushButton,"btn_load")
            for button in children:
                button.click() #click load button

        if col == 3: #delete all
            for i in range(parent.rowCount()):
                parent.removeRow(0)
    

    
    def anti_vowel(self,c):
        newstr = ""
        vowels = ('a', 'e', 'i', 'o', 'u','A', 'E', 'I', 'O', 'U')
        for x in c.lower():
            if x in vowels:
                newstr = ''.join([l for l in c if l not in vowels])    
        return newstr
    
    
    def load_rtdc_images(self):
        buttonClicked = self.sender()
        index = self.table_aid_files.indexAt(buttonClicked.pos())
        rowPosition = index.row()
    
        file_path = self.table_aid_files.item(rowPosition, 2).text()
        
        if file_path.endswith(".rtdc"):
            rtdc_ds = h5py.File(file_path,"r")
    
            #Get the images from .rtdc file
            images = np.array(rtdc_ds["events"]["image"]) 
        
            name = os.path.basename(rtdc_path)
            new_layer = self.viewer.add_image(images,name=name)
        
        if file_path.endswith(".bin"):
            binary = np.fromfile(file_path, dtype='>H')
            #Get the images from .bin
            n,w,h = binary[1],binary[3],binary[5]
            images = binary[6:].reshape(n,h,w)
            
            name = os.path.basename(file_path)[:-4] #remove .bin
            new_layer = self.viewer.add_image(images,name=name)
            
        
    
    def aid_load_model(self):
        openfile_name = QtWidgets.QFileDialog.getExistingDirectory(self)
        if openfile_name:
            self.lineEdit_aid_model_path.setText(openfile_name)
    
  
            
    def aid_classify_rtdc(self,model_path,rtdc_path):

        #model_path = "/" + self.lineEdit_aid_model_path.text()
        #find .pb and mate files in model folder
        for path,dir_list,file_list in os.walk(model_path):
            for file_name in file_list:
                if file_name.endswith(".xlsx"):
                    meta_path = os.path.join(model_path,file_name) 
                if file_name.endswith(".pb"):
                    model_pb_path = os.path.join(model_path,file_name) 
            if not meta_path:
                print("Could not fine meta file.")
            if not model_pb_path:
                print("Could not fine .pb file.")
        
        #Load model
        model_pb = cv2.dnn.readNet(model_pb_path)
        # Extract image preprocessing settings from meta file
        img_processing_settings = load_model_meta(meta_path)
        
        #Load .rtdc
        rtdc_ds = h5py.File(rtdc_path,"r")
        images = np.array(rtdc_ds["events"]["image"]) # get the images
        pos_x, pos_y = rtdc_ds["events"]["pos_x"][:], rtdc_ds["events"]["pos_y"][:] 
        pix = rtdc_ds.attrs["imaging:pixel size"] # pixelation (um/pix)

        # Compute the predictions
        scores = forward_images_cv2(model_pb,img_processing_settings,images,pos_x,pos_y,pix)
        prediction = np.argmax(scores,axis=1)
        
        #predict disease
        rtdc_ds_len = rtdc_ds["events"]["image"].shape[0] #this way is actually faster than asking any other feature for its len :)
        prediction_fillnan = np.full([rtdc_ds_len], np.nan)#put initially np.nan for all cells

        classes = scores.shape[1]
        if classes>9:
            classes = 9#set the max number of classes to 9. It cannot saved more to .rtdc due to limitation of userdef

        #Make sure the predictions get again to the same length as the initial data set
        #Fill array with corresponding predictions
        index = range(len(images))

        for i in range(len(prediction)):
            indx = index[i]
            prediction_fillnan[indx] = prediction[i]

        #Predictions are integers
        prediction_fillnan = prediction_fillnan.astype(int)

        #Get area, area_ratio values for each cell type
        X_features = getdata2(rtdc_path,prediction_fillnan)
        values = np.array(X_features[0])
        values = values.reshape(-1,values.shape[0])
        FeatNames = X_features[1]
        X_features = pd.DataFrame(values,columns=FeatNames)

        #re-order the features (Random forest expexts a certain order)
        features_path = os.path.join(model_path,"02_rf_model","09_RF_v02.3_features.csv")
        features_select = pd.read_csv(features_path)
        features = [row for index, row in features_select.itertuples()]

        #Disease classification using Rndom forest model
        model_path = os.path.join(model_path,"02_rf_model","09_RF_v02.3.sav")
        #load the random forest model
        rf_model = pickle.load(open(model_path, 'rb'))

        X_features = X_features[features]
        disease_probab = rf_model.predict_proba(X_features)
        
        
        return prediction, images, (pos_x/pix).astype(int), (pos_y/pix).astype(int), scores, model_pb_path, disease_probab
    
    
    def show_results(self,prediction,disease_probab):
        #Statistics
        result = pd.value_counts(prediction).sort_index()
        percentage = pd.value_counts(prediction,normalize=True).sort_index()
        
        indexs = [index for index,v in result.items()]
        values = [value for i,value in result.items()]
        pcts = [pct for i,pct in percentage.items()]
        Name = ["Noise","single Platelet", "multiple Platelets","single WBC", 
                "single WBC+Platelets", "multiple WBC", "multiple WBC+Platelets" ]
        
        #clear content before add new
        if not self.table_aid_analy.rowCount() ==0:
            for i in range(self.table_aid_analy.rowCount()):
                self.table_aid_analy.removeRow(0) 
        
        
        for rowNumber in range(len(result)):
            index = indexs[rowNumber]
            value = values[rowNumber]
            pct = np.round(pcts[rowNumber],4)
            
            #add to table
            rowPosition = self.table_aid_analy.rowCount()
            self.table_aid_analy.insertRow(rowPosition)
            
            columnPosition = 0 # load
            #Place a button which allows to send to napari for viewing
            btn = QtWidgets.QPushButton(self.table_aid_analy)
            btn.setObjectName("btn_load")
            btn.setMinimumSize(QtCore.QSize(30, 30))
            btn.setMaximumSize(QtCore.QSize(100, 100))
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(os.path.join(dir_root,"art","visibility_off.png")), QtGui.QIcon.Active, QtGui.QIcon.On)
            icon.addPixmap(QtGui.QPixmap(os.path.join(dir_root,"art","eye.png")), QtGui.QIcon.Active, QtGui.QIcon.Off)
            btn.setIcon(icon)
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.toggle()
            btn.clicked.connect(self.send_to_napari_pred_class)
            
            self.table_aid_analy.setCellWidget(rowPosition, columnPosition, btn) 
            self.table_aid_analy.resizeRowsToContents()
            
            columnPosition = 1 # class column
            line = QtWidgets.QTableWidgetItem(str(index)) 
            
            line.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            line.setFlags( QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.table_aid_analy.setItem(rowPosition,columnPosition, line) 
            
            columnPosition = 2 # name
            line = QtWidgets.QTableWidgetItem(Name[index]) 
            line.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            line.setFlags( QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.table_aid_analy.setItem(rowPosition,columnPosition, line) 
            
            
            columnPosition = 3 # number
            line = QtWidgets.QTableWidgetItem(str(value)) 
            line.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            line.setFlags( QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.table_aid_analy.setItem(rowPosition,columnPosition, line) 
            
            columnPosition = 4 # percentage
            line = QtWidgets.QTableWidgetItem(str(pct)) 
            line.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            line.setFlags( QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.table_aid_analy.setItem(rowPosition,columnPosition, line) 
            
            pro_bar = QtWidgets.QProgressBar(self.table_aid_analy)
            pro_bar.setRange(0, 100)
            pro_bar.setValue(int(pct*100))
            self.table_aid_analy.setCellWidget(rowPosition, columnPosition, pro_bar) 
        
        Name = ["Thrombosis" , "COVID-19"]
        probability = disease_probab[0]
        for rowNumber in range(2):
            #add to table
            rowPosition = self.table_dise_class.rowCount()
            self.table_dise_class.insertRow(rowPosition)
            
            columnPosition = 0 # disease name
            line = QtWidgets.QTableWidgetItem(Name[rowNumber]) 
            line.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            line.setFlags( QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.table_dise_class.setItem(rowPosition,columnPosition, line) 
            
            columnPosition = 1 # percentage
            line = QtWidgets.QTableWidgetItem(probability[rowNumber]) 
            line.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            line.setFlags( QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.table_dise_class.setItem(rowPosition,columnPosition, line) 
            
            pro_bar = QtWidgets.QProgressBar(self.table_dise_class)
            pro_bar.setRange(0, 100)
            pro_bar.setValue(int(probability[rowNumber]*100))
            self.table_dise_class.setCellWidget(rowPosition, columnPosition, pro_bar) 
            
            
            
    def run_classify(self):
        model_path = "/" + self.lineEdit_aid_model_path.text()        
        rowPosition = self.table_aid_files.rowCount()
        rtdc_paths = []
        bin_paths = []
        
        for i in range(rowPosition): # read all checked data
            if self.table_aid_files.item(i, 0).checkState(): #return 0 or 2,,0:uncheck, 2:checked
                file_path = self.table_aid_files.item(i, 2).text()
                if file_path.endswith('.rtdc'):
                    rtdc_paths.append(file_path)
                if file_path.endswith('.bin'):
                    bin_paths.append(file_path)
                
        if len(rtdc_paths)>0:
            for rtdc_path in rtdc_paths:
                self.prediction, self.images, self.pos_x, self.pos_y, scores, model_pb_path, disease_probab = self.aid_classify_rtdc(model_path,rtdc_path)
                self.show_results(self.prediction,disease_probab)
        
        if len(bin_paths)>0:
            for bin_path in bin_paths:
                rtdc_path = bin_2_rtdc(bin_path)
                self.prediction, self.images, self.pos_x, self.pos_y, scores, model_pb_path, disease_probab = self.aid_classify_rtdc(model_path,rtdc_path)
                self.show_results(self.prediction,disease_probab)

        
    
    def send_to_napari_pred_class(self):
        #get class images
        buttonClicked = self.sender()
        index = self.table_aid_analy.indexAt(buttonClicked.pos())
        rowPosition = index.row()
        class_num = self.table_aid_analy.item(rowPosition,1).text()
        images = self.images
        class_ind = np.where(self.prediction==int(class_num))
        class_images = images[class_ind]
        class_name = "Class " + class_num
        
        #add class label
        x_ = self.pos_x[class_ind]
        y_ = self.pos_y[class_ind]
        label = []
        for i in range(len(class_images)):
            mask = np.zeros((67,67,4),dtype=np.uint8) 
            x = x_[i]
            y = y_[i] 
            cv2.circle(mask, (x,y), 1, (0,255,0,255),-1)
            cv2.putText(mask,class_num,(2,11),cv2.FONT_HERSHEY_DUPLEX, 0.4, (255,255,255,255),1)
            label.append(mask)
        label = np.array(label)
        label_name = "Label " + class_num
        
        #button status
        if buttonClicked.isChecked(): #add layer
            new_layer = self.viewer.add_image(class_images,name=class_name)
            lable_layer = self.viewer.add_image(label, name=label_name)     
        else: #delete layer
            existing_layers = {layer.name for layer in self.viewer.layers} 
            if class_name in existing_layers:
                self.viewer.layers.remove(class_name)
            if label_name in existing_layers:
                self.viewer.layers.remove(label_name)
            
        
    def send_to_napari_all_class(self,col):
        if col==0:
            parent = self.sender().parent()
            children= parent.findChildren(QtWidgets.QPushButton,"btn_load")
            for button in children:
                button.click() #click load button


    def aid_add_rtdc(self):
        model_path = "/" + self.lineEdit_aid_model_path.text() 
        rowPosition = self.table_aid_files.rowCount()
        file_paths = []
        save_paths = []
        for i in range(rowPosition): # read all data
            #if self.table_aid_files.item(i, 0).checkState(): #return 0 or 2,,0:uncheck, 2:checked
                file_path = self.table_aid_files.item(i, 2).text()
                file_paths.append(file_path)
        
        for file_path in file_paths:
            if file_path.endswith('.rtdc'):
                rtdc_path = file_path
            if file_path.endswith('.bin'):
                rtdc_path = bin_2_rtdc(file_path)
                
            prediction, images, pos_x, pos_y, scores, model_pb_path,disease_probab = self.aid_classify_rtdc(model_path,rtdc_path)
            rtdc_ds = h5py.File(rtdc_path,"r")
            
            ###################append scores and pred to .rtdc file########################
            rtdc_ds_len = rtdc_ds["events"]["image"].shape[0] #this way is actually faster than asking any other feature for its len :)
            prediction_fillnan = np.full([rtdc_ds_len], np.nan)#put initially np.nan for all cells
            
            classes = scores.shape[1]
            if classes>9:
                classes = 9#set the max number of classes to 9. It cannot saved more to .rtdc due to limitation of userdef
            scores_fillnan = np.full([rtdc_ds_len,classes], np.nan)
    
            #Make sure the predictions get again to the same length as the initial data set
            #Fill array with corresponding predictions
            index = range(len(images))
    
            for i in range(len(prediction)):
                indx = index[i]
                prediction_fillnan[indx] = prediction[i]
                #if export_option == "Append to .rtdc":
                #for class_ in range(classes):
                scores_fillnan[indx,0:classes] = scores[i,0:classes]
    
            #Get savename
            path, rtdc_file = os.path.split(rtdc_path)
            
            fname_addon = os.path.split(model_pb_path)[-1]#the filename of the model
            fname_addon = fname_addon.split(".pb")[0]
            fname_addon = self.anti_vowel(fname_addon)#remove the vowels to make it shorter
            savename = rtdc_path.split(".rtdc")[0]
            savename = savename+"_"+str(fname_addon)+".rtdc"
            save_paths.append(savename)
            
            if not os.path.isfile(savename):#if such a file does not yet exist...
                savename = savename
            else:#such a file already exists!!!
                #Avoid to overwriting an existing file:
                print("Adding additional number since file exists!")
                addon = 1
                while os.path.isfile(savename):
                    savename = savename.split(".rtdc")[0]
                    if addon>1:
                        savename = savename.split("_"+str(addon-1))[0]
                    savename = savename+"_"+str(addon)+".rtdc"
                    addon += 1        
    
            print(f"Save as {savename}")                    
            shutil.copy(rtdc_path, savename) #copy original file
            #append to hdf5 file
            with h5py.File(savename, mode="a") as h5:
                h5["events/userdef0"] = prediction_fillnan
                #add the scores to userdef1...9
                userdef_ind = 1
                for class_ in range(classes):
                    scores_i = scores_fillnan[:,class_]
                    h5["events/userdef"+str(userdef_ind)] = scores_i
                    userdef_ind += 1
            
            
        ###################### show messagebox ######## 
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("")
        text = "\n".join(save_paths)
        msg.setText("Successfully export .rtdc files in \n" + text)
        
        x = msg.exec_()


        
        
    
    

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [iacs_ipac_reader]


