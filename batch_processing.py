import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import threading
import time
import pyxas
from PyQt5 import QtGui
from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QRadioButton, QApplication,QWidget,
                             QLineEdit, QPlainTextEdit, QWidget, QPushButton, QLabel, QCheckBox, QGroupBox,
                             QScrollBar, QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget,
                             QListWidget, QListWidgetItem, QAbstractItemView, QScrollArea,
                             QSlider, QComboBox, QButtonGroup, QMessageBox, QSizePolicy)
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5 import QtCore
from skimage import io
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from copy import deepcopy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyxas.image_util import img_smooth, rm_abnormal, bin_ndarray, rm_noise, kmean_mask
from pyxas.align3D import *
from scipy.ndimage.interpolation import shift
from scipy.signal import medfilt2d, medfilt
from scipy.interpolate import interp1d, UnivariateSpline
from scipy import ndimage
from pyxas.xanes_util import (fit_2D_xanes_non_iter, fit_2D_xanes_iter, fit_2D_xanes_iter2, normalize_2D_xanes2, normalize_2D_xanes_old, normalize_1D_xanes, find_nearest, normalize_2D_xanes_regulation)
from multiprocessing import cpu_count


global xanes


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Fitting Parameter'
        screen_resolution = QApplication.desktop().screenGeometry()
        width, height = screen_resolution.width(), screen_resolution.height()
        self.width = 1020
        self.height = 800
        self.left = (width - self.width) // 2
        self.top = (height - self.height) // 2
        self.initUI()
        #self.default_layout()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.font1 = QtGui.QFont('Arial', 11, QtGui.QFont.Bold)
        self.font2 = QtGui.QFont('Arial', 11, QtGui.QFont.Normal)
        self.lb_empty = QLabel()
        self.fpath = os.getcwd()
        self.spectrum_ref = {}
        self.xanes_files = []
        self.elem_label = []
        self.num_ref = 0
        self.load_eng_successful = 0
        self.load_file_successful = 0
        self.save_fit_param_successful = 0
        self.load_fit_param_successful = 0

        gpbox_param_file = self.layout_param_open_filefolder()
        gpbox_msg = self.layout_msg()
        gpbox_fit_param = self.layout_param_fit()


        grid = QGridLayout()
        grid.addWidget(gpbox_param_file, 0, 1)
        grid.addLayout(gpbox_msg, 1, 1)
        grid.addLayout(gpbox_fit_param, 2, 1)
        '''
        grid.addWidget(gpbox_param_txm_norm, 1, 1)
        grid.addWidget(gpbox_param_edge_fit, 2, 1)
        grid.addWidget(gpbox_param_peak_reg, 3, 1)
        grid.addWidget(gpbox_param_image_align, 4, 1)
        grid.addWidget(gpbox_param_xanes_fit, 5, 1)
        grid.addWidget(gpbox_param_mask, 6, 1)
        #grid.addLayout(gpbox_msg, 2, 1)
        '''

        layout = QVBoxLayout()
        layout.addLayout(grid)
        layout.addWidget(QLabel())

        self.setLayout(layout)

        #self.connect(SIGNAL("returnPressed(void)"), self.run_command)

    def layout_param_open_filefolder(self):
        gpbox = QGroupBox('Files')
        gpbox.setFont(self.font1)

        # file directory
        lb_param_filepath = QLabel()
        lb_param_filepath.setFont(self.font2)
        lb_param_filepath.setText('File path:')
        lb_param_filepath.setFixedWidth(80)

        self.tx_param_folder = QLineEdit()
        self.tx_param_folder.setFixedWidth(250)
        #self.tx_param_folder.setEnabled(False)
        self.tx_param_folder.setFont(self.font2)

        self.pb_file = QPushButton('Open')
        self.pb_file.setFont(self.font2)
        self.pb_file.clicked.connect(self.open_folder)
        self.pb_file.setFixedWidth(80)
        self.pb_file.setFixedWidth(80)

        hbox_param_open = QHBoxLayout()
        hbox_param_open.addWidget(lb_param_filepath)
        hbox_param_open.addWidget(self.tx_param_folder)
        hbox_param_open.addWidget(self.pb_file)
        hbox_param_open.setAlignment(QtCore.Qt.AlignLeft)

        self.tx_param_pre_s = QLineEdit()
        self.tx_param_pre_s.setFixedWidth(60)
        self.tx_param_pre_s.setFont(self.font2)

        # file name and type
        lb_param_file_name= QLabel()
        lb_param_file_name.setFont(self.font2)
        lb_param_file_name.setText('File name start with:')
        lb_param_file_name.setFixedWidth(140)

        self.tx_param_file_prefix = QLineEdit()
        self.tx_param_file_prefix.setFixedWidth(60)
        self.tx_param_file_prefix.setText('xanes')
        self.tx_param_file_prefix.setFont(self.font2)

        lb_param_file_type = QLabel()
        lb_param_file_type.setFont(self.font2)
        lb_param_file_type.setText('  format:')
        lb_param_file_type.setFixedWidth(60)

        self.tx_param_file_type = QLineEdit()
        self.tx_param_file_type.setFixedWidth(60)
        self.tx_param_file_type.setText('.tiff')
        self.tx_param_file_type.setFont(self.font2)

        self.pb_load_file = QPushButton('Load')
        self.pb_load_file.setFont(self.font2)
        self.pb_load_file.clicked.connect(self.load_xanes_image_file)
        self.pb_load_file.setFixedWidth(80)

        # file type: .h5 or .tiff
        lb_param_true_file_type = QLabel()
        lb_param_true_file_type.setText('  File type:')
        lb_param_true_file_type.setFont(self.font2)
        lb_param_true_file_type.setFixedWidth(80)

        self.file_group = QButtonGroup()
        self.file_group.setExclusive(True)
        self.rd_hdf = QRadioButton('hdf')
        self.rd_hdf.setFixedWidth(60)
        self.rd_hdf.setChecked(True)

        self.rd_tif = QRadioButton('tiff')
        self.rd_tif.setFixedWidth(60)

        self.file_group.addButton(self.rd_hdf)
        self.file_group.addButton(self.rd_tif)

        lb_param_fit_hdf = QLabel()
        lb_param_fit_hdf.setText('Dataset:')
        lb_param_fit_hdf.setFixedWidth(80)
        lb_param_fit_hdf.setFont(self.font2)

        self.tx_param_hdf = QLineEdit()
        self.tx_param_hdf.setFixedWidth(80)
        self.tx_param_hdf.setText('img_xanes')
        self.tx_param_hdf.setFont(self.font2)

        hbox_param_file = QHBoxLayout()
        hbox_param_file.addWidget(lb_param_file_name)
        hbox_param_file.addWidget(self.tx_param_file_prefix)
        hbox_param_file.addWidget(lb_param_file_type)
        hbox_param_file.addWidget(self.tx_param_file_type)
        hbox_param_file.addWidget(self.pb_load_file)
        hbox_param_file.addWidget(lb_param_true_file_type)
        hbox_param_file.addWidget(self.rd_tif)
        hbox_param_file.addWidget(self.rd_hdf)
        hbox_param_file.addWidget(self.tx_param_hdf)
        hbox_param_file.setAlignment(QtCore.Qt.AlignLeft)

        vbox_param_file = QVBoxLayout()
        vbox_param_file.addLayout(hbox_param_open)
        vbox_param_file.addLayout(hbox_param_file)
        vbox_param_file.setAlignment(QtCore.Qt.AlignTop)

        gpbox.setLayout(vbox_param_file)

        return gpbox

    def layout_param_default(self):
        pass

    def layout_msg(self):
        self.lb_ip = QLabel()
        self.lb_ip.setFont(self.font2)
        self.lb_ip.setStyleSheet('color: rgb(200, 50, 50);')
        self.lb_ip.setText('File loaded:')

        self.lb_msg = QLabel()
        self.lb_msg.setFont(self.font1)
        self.lb_msg.setStyleSheet('color: rgb(200, 50, 50);')
        self.lb_msg.setText('Message: Load File first before batch fitting')

        vbox_msg = QVBoxLayout()
        #vbox_msg.addWidget(self.lb_ip)
        vbox_msg.addWidget(self.lb_msg)
        vbox_msg.setAlignment(QtCore.Qt.AlignLeft)
        return vbox_msg

    def layout_param_fit(self):

        param_cpu = self.layout_param_cpu()
        param_txm_norm = self.layout_param_txm_norm()
        param_edge_fit = self.layout_param_edge_norm()
        param_peak_reg = self.layout_param_edge_regulation()
        param_image_align = self.layout_param_align()
        param_xanes_fit = self.layout_param_fitting()
        param_mask = self.layout_param_mask()
        param_ref = self.layout_param_ref()
        param_execute = self.layout_param_execute()
        param_colormix = self.layout_param_colormix()
        param_comb_color_ref = QVBoxLayout()
        param_comb_color_ref.addLayout(param_colormix)
        param_comb_color_ref.addLayout(param_ref)

        grid = QGridLayout()
        grid.addLayout(param_cpu, 0, 1)
        grid.addLayout(param_txm_norm, 0, 2)
        grid.addLayout(param_edge_fit, 1, 1)
        grid.addLayout(param_peak_reg, 1, 2)
        grid.addLayout(param_image_align, 2, 1)
        grid.addLayout(param_mask, 2, 2)
        grid.addLayout(param_xanes_fit, 3, 1)
        grid.addLayout(param_comb_color_ref, 3, 2)
        #grid.addLayout(param_ref, 4, 2)
        grid.addLayout(param_execute, 4, 1, 1, 2)
        '''
        hbox1 = QHBoxLayout()
        hbox1.addLayout(param_cpu)
        hbox1.addLayout(param_txm_norm)
        hbox1.addWidget(lb_empty)
        hbox1.setAlignment(QtCore.Qt.AlignLeft)

        hbox2 = QHBoxLayout()
        hbox2.addLayout(param_edge_fit)
        hbox2.addLayout(param_peak_reg)
        hbox2.addWidget(lb_empty)
        hbox2.setAlignment(QtCore.Qt.AlignLeft)

        hbox3 = QHBoxLayout()
        hbox3.addLayout(param_image_align)
        hbox3.addLayout(param_xanes_fit)
        hbox3.addWidget(lb_empty)
        hbox3.setAlignment(QtCore.Qt.AlignLeft)

        vbox = QVBoxLayout()
        #vbox.addLayout(param_cpu)
        vbox.addWidget(lb_empty)
        #vbox.addLayout(param_txm_norm)
        #vbox.addWidget(lb_empty)
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        #vbox.addLayout(hbox3)
        
        return vbox
        '''
        return grid

    def layout_param_cpu(self):
        lb_param_parellel = QLabel()
        lb_param_parellel.setFont(self.font1)
        lb_param_parellel.setText('Parallel computing')

        lb_param_cpu = QLabel()
        lb_param_cpu.setFont(self.font2)
        lb_param_cpu.setText('Number of CPU:')
        lb_param_cpu.setFixedWidth(140)

        self.tx_param_cpu = QLineEdit()
        self.tx_param_cpu.setFixedWidth(60)
        self.tx_param_cpu.setText('1')
        self.tx_param_cpu.setFont(self.font2)

        hbox_param_cpu = QHBoxLayout()
        hbox_param_cpu.addWidget(lb_param_cpu)
        hbox_param_cpu.addWidget(self.tx_param_cpu)
        hbox_param_cpu.setAlignment(QtCore.Qt.AlignLeft)

        vbox_param_cpu = QVBoxLayout()
        vbox_param_cpu.addWidget(lb_param_parellel)
        vbox_param_cpu.addLayout(hbox_param_cpu)
        vbox_param_cpu.addWidget(self.lb_empty)
        vbox_param_cpu.setAlignment(QtCore.Qt.AlignTop)
        return vbox_param_cpu

    def layout_param_txm_norm(self):
        lb_param_txm_norm0 = QLabel()
        lb_param_txm_norm0.setFont(self.font1)
        lb_param_txm_norm0.setText('TXM normalization')
        lb_param_txm_norm0.setFixedWidth(140)

        lb_param_txm_norm = QLabel()
        lb_param_txm_norm.setFont(self.font2)
        lb_param_txm_norm.setText('Take -log()')
        lb_param_txm_norm.setFixedWidth(80)

        self.param_txm_norm_group = QButtonGroup()
        self.param_txm_norm_group.setExclusive(True)
        self.rd_param_txm_norm_yes = QRadioButton('Yes')
        self.rd_param_txm_norm_yes.setFixedWidth(80)

        self.rd_param_txm_norm_no = QRadioButton('No')
        self.rd_param_txm_norm_no.setFixedWidth(80)

        self.param_txm_norm_group.addButton(self.rd_param_txm_norm_yes)
        self.param_txm_norm_group.addButton(self.rd_param_txm_norm_no)
        self.param_txm_norm_group.setExclusive(True)
        self.rd_param_txm_norm_yes.setChecked(True)

        hbox_param_txm_norm = QHBoxLayout()
        hbox_param_txm_norm.addWidget(lb_param_txm_norm)
        hbox_param_txm_norm.addWidget(self.rd_param_txm_norm_yes)
        hbox_param_txm_norm.addWidget(self.rd_param_txm_norm_no)
        hbox_param_txm_norm.addWidget(self.lb_empty)
        hbox_param_txm_norm.setAlignment(QtCore.Qt.AlignLeft)

        vbox_param_txm_norm = QVBoxLayout()
        vbox_param_txm_norm.addWidget(lb_param_txm_norm0)
        vbox_param_txm_norm.addLayout(hbox_param_txm_norm)
        vbox_param_txm_norm.addWidget(self.lb_empty)
        vbox_param_txm_norm.setAlignment(QtCore.Qt.AlignTop)
        return vbox_param_txm_norm

    def layout_param_edge_norm(self):
        #gpbox = QGroupBox('Edge normalization')
        #gpbox.setFont(self.font1)

        #self.chkbox_param_edge_norm = QCheckBox('Edge normalization')
        #self.chkbox_param_edge_norm.setFixedWidth(190)
        #self.chkbox_param_edge_norm.setFont(self.font2)

        lb_space = QLabel()
        lb_space.setFixedWidth(40)

        lb_param_fit_edge0 = QLabel()
        lb_param_fit_edge0.setFont(self.font1)
        lb_param_fit_edge0.setText('Edge normalization')
        lb_param_fit_edge0.setFixedWidth(140)

        # normalization method
        lb_param_fit_edge_method = QLabel()
        lb_param_fit_edge_method.setFont(self.font2)
        lb_param_fit_edge_method.setText('Method:')
        lb_param_fit_edge_method.setFixedWidth(80)

        self.chkbox_param_fit_edge_method = QCheckBox('Global slope')
        self.chkbox_param_fit_edge_method.setFixedWidth(140)
        self.chkbox_param_fit_edge_method.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.chkbox_param_fit_edge_method.setFont(self.font2)

        hbox_param_fit_edge_method = QHBoxLayout()
        hbox_param_fit_edge_method.addWidget(lb_param_fit_edge0)
        #hbox_param_fit_edge_method.addWidget(lb_space)
        hbox_param_fit_edge_method.addWidget(self.chkbox_param_fit_edge_method)
        hbox_param_fit_edge_method.setAlignment(QtCore.Qt.AlignLeft)

        # normalize or not
        lb_param_fit_edge = QLabel()
        lb_param_fit_edge.setFont(self.font2)
        lb_param_fit_edge.setText('Norm absorp. edge')
        lb_param_fit_edge.setFixedWidth(140)

        self.fit_edge_group = QButtonGroup()
        self.fit_edge_group.setExclusive(True)
        self.rd_param_fit_edge_yes = QRadioButton('Yes')
        self.rd_param_fit_edge_yes.setFixedWidth(80)

        self.rd_param_fit_edge_no = QRadioButton('No')
        self.rd_param_fit_edge_no.setFixedWidth(80)

        self.fit_edge_group.addButton(self.rd_param_fit_edge_yes)
        self.fit_edge_group.addButton(self.rd_param_fit_edge_no)
        self.fit_edge_group.setExclusive(True)
        self.rd_param_fit_edge_yes.setChecked(True)

        hbox_param_fit_edge_flag = QHBoxLayout()
        hbox_param_fit_edge_flag.addWidget(lb_param_fit_edge)
        hbox_param_fit_edge_flag.addWidget(self.rd_param_fit_edge_yes)
        hbox_param_fit_edge_flag.addWidget(self.rd_param_fit_edge_no)
        hbox_param_fit_edge_flag.setAlignment(QtCore.Qt.AlignLeft)



        # pre-edge
        lb_param_pre = QLabel()
        lb_param_pre.setFont(self.font2)
        lb_param_pre.setText('Pre-edge')
        lb_param_pre.setFixedWidth(80)

        lb_param_pre_start = QLabel()
        lb_param_pre_start.setFont(self.font2)
        lb_param_pre_start.setText('start:')
        lb_param_pre_start.setFixedWidth(60)

        lb_param_pre_end = QLabel()
        lb_param_pre_end.setFont(self.font2)
        lb_param_pre_end.setText('  end:')
        lb_param_pre_end.setFixedWidth(60)

        self.tx_param_pre_s = QLineEdit()
        self.tx_param_pre_s.setFixedWidth(60)
        self.tx_param_pre_s.setFont(self.font2)

        self.tx_param_pre_e = QLineEdit()
        self.tx_param_pre_e.setFixedWidth(60)
        self.tx_param_pre_e.setFont(self.font2)

        hbox_param_pre = QHBoxLayout()
        hbox_param_pre.addWidget(lb_param_pre)
        hbox_param_pre.addWidget(lb_param_pre_start)
        hbox_param_pre.addWidget(self.tx_param_pre_s)
        hbox_param_pre.addWidget(lb_param_pre_end)
        hbox_param_pre.addWidget(self.tx_param_pre_e)
        hbox_param_pre.setAlignment(QtCore.Qt.AlignLeft)

        # post edge
        lb_param_post = QLabel()
        lb_param_post.setFont(self.font2)
        lb_param_post.setText('Post-edge')
        lb_param_post.setFixedWidth(80)

        lb_param_post_start = QLabel()
        lb_param_post_start.setFont(self.font2)
        lb_param_post_start.setText('start:')
        lb_param_post_start.setFixedWidth(60)

        lb_param_post_end = QLabel()
        lb_param_post_end.setFont(self.font2)
        lb_param_post_end.setText('  end:')
        lb_param_post_end.setFixedWidth(60)

        self.tx_param_post_s = QLineEdit()
        self.tx_param_post_s.setFixedWidth(60)
        self.tx_param_post_s.setFont(self.font2)

        self.tx_param_post_e = QLineEdit()
        self.tx_param_post_e.setFixedWidth(60)
        self.tx_param_post_e.setFont(self.font2)

        hbox_param_post = QHBoxLayout()
        hbox_param_post.addWidget(lb_param_post)
        hbox_param_post.addWidget(lb_param_post_start)
        hbox_param_post.addWidget(self.tx_param_post_s)
        hbox_param_post.addWidget(lb_param_post_end)
        hbox_param_post.addWidget(self.tx_param_post_e)
        hbox_param_post.setAlignment(QtCore.Qt.AlignLeft)

        vbox_param_edge = QVBoxLayout()
        vbox_param_edge.addLayout(hbox_param_fit_edge_method)
        vbox_param_edge.addLayout(hbox_param_fit_edge_flag)
        vbox_param_edge.addLayout(hbox_param_fit_edge_method)
        vbox_param_edge.addLayout(hbox_param_pre)
        vbox_param_edge.addLayout(hbox_param_post)
        vbox_param_edge.addWidget(self.lb_empty)
        vbox_param_edge.setAlignment(QtCore.Qt.AlignTop)


        return vbox_param_edge

    def layout_param_edge_regulation(self):
        #gpbox = QGroupBox('Peak regulation')
        #gpbox.setFont(self.font1)

        lb_param_reg_edge0 = QLabel()
        lb_param_reg_edge0.setFont(self.font1)
        lb_param_reg_edge0.setText('Peak regulation')
        lb_param_reg_edge0.setFixedWidth(140)
        # normalize or not
        lb_param_reg_edge = QLabel()
        lb_param_reg_edge.setFont(self.font2)
        lb_param_reg_edge.setText('Peak regulation')
        lb_param_reg_edge.setFixedWidth(140)

        self.reg_edge_group = QButtonGroup()
        self.reg_edge_group.setExclusive(True)
        self.rd_param_reg_edge_yes = QRadioButton('Yes')
        self.rd_param_reg_edge_yes.setFixedWidth(80)

        self.rd_param_reg_edge_no = QRadioButton('No')
        self.rd_param_reg_edge_no.setFixedWidth(80)
        self.rd_param_reg_edge_no.setChecked(True)

        self.reg_edge_group.addButton(self.rd_param_reg_edge_yes)
        self.reg_edge_group.addButton(self.rd_param_reg_edge_no)
        self.reg_edge_group.setExclusive(True)

        hbox_param_reg_edge_flag = QHBoxLayout()
        hbox_param_reg_edge_flag.addWidget(lb_param_reg_edge)
        hbox_param_reg_edge_flag.addWidget(self.rd_param_reg_edge_yes)
        hbox_param_reg_edge_flag.addWidget(self.rd_param_reg_edge_no)
        hbox_param_reg_edge_flag.setAlignment(QtCore.Qt.AlignLeft)

        # regulation peak max
        lb_param_reg_peak = QLabel()
        lb_param_reg_peak.setFont(self.font2)
        lb_param_reg_peak.setText('Peak maximum:')
        lb_param_reg_peak.setFixedWidth(140)

        self.tx_param_reg_peak = QLineEdit()
        self.tx_param_reg_peak.setFixedWidth(60)
        self.tx_param_reg_peak.setFont(self.font2)

        # regulation gamma
        lb_param_reg_gamma = QLabel()
        lb_param_reg_gamma.setFont(self.font2)
        lb_param_reg_gamma.setText('  Width:')
        lb_param_reg_gamma.setFixedWidth(60)

        self.tx_param_reg_gamma = QLineEdit()
        self.tx_param_reg_gamma.setText('0.05')
        self.tx_param_reg_gamma.setFixedWidth(60)
        self.tx_param_reg_gamma.setFont(self.font2)

        hbox_param_reg_peak = QHBoxLayout()
        hbox_param_reg_peak.addWidget(lb_param_reg_peak)
        hbox_param_reg_peak.addWidget(self.tx_param_reg_peak)
        hbox_param_reg_peak.addWidget(lb_param_reg_gamma)
        hbox_param_reg_peak.addWidget(self.tx_param_reg_gamma)
        hbox_param_reg_peak.setAlignment(QtCore.Qt.AlignLeft)

        vbox_param_reg = QVBoxLayout()
        vbox_param_reg.addWidget(lb_param_reg_edge0)
        vbox_param_reg.addLayout(hbox_param_reg_edge_flag)
        vbox_param_reg.addLayout(hbox_param_reg_peak)
        vbox_param_reg.addWidget(self.lb_empty)
        vbox_param_reg.setAlignment(QtCore.Qt.AlignTop)

        #gpbox.setLayout(vbox_param_reg)
        return vbox_param_reg

    def layout_param_align(self):
        #gpbox = QGroupBox('Image alignment')
        #gpbox.setFont(self.font1)

        lb_param_align0 = QLabel()
        lb_param_align0.setFont(self.font1)
        lb_param_align0.setText('Image alignment')
        lb_param_align0.setFixedWidth(140)
        # Alignment
        lb_param_align = QLabel()
        lb_param_align.setFont(self.font2)
        lb_param_align.setText('Image alignment')
        lb_param_align.setFixedWidth(140)

        self.align_group = QButtonGroup()
        self.align_group.setExclusive(True)
        self.rd_param_align_yes = QRadioButton('Yes')
        self.rd_param_align_yes.setFixedWidth(80)

        self.rd_param_align_no = QRadioButton('No')
        self.rd_param_align_no.setFixedWidth(80)

        self.align_group.addButton(self.rd_param_align_yes)
        self.align_group.addButton(self.rd_param_align_no)
        self.align_group.setExclusive(True)
        self.rd_param_align_no.setChecked(True)

        hbox_param_align_flag = QHBoxLayout()
        hbox_param_align_flag.addWidget(lb_param_align)
        hbox_param_align_flag.addWidget(self.rd_param_align_yes)
        hbox_param_align_flag.addWidget(self.rd_param_align_no)
        hbox_param_align_flag.setAlignment(QtCore.Qt.AlignLeft)

        # ROI ratio
        lb_param_align_roi = QLabel()
        lb_param_align_roi.setFont(self.font2)
        lb_param_align_roi.setText('  Ratio:')
        lb_param_align_roi.setFixedWidth(60)

        self.tx_param_align_roi = QLineEdit()
        self.tx_param_align_roi.setFixedWidth(60)
        self.tx_param_align_roi.setText('0.9')
        self.tx_param_align_roi.setFont(self.font2)

        # reference index
        lb_param_align_ref = QLabel()
        lb_param_align_ref.setFont(self.font2)
        lb_param_align_ref.setText('Reference index:')
        lb_param_align_ref.setFixedWidth(140)

        self.tx_param_align_ref = QLineEdit()
        self.tx_param_align_ref.setFixedWidth(60)
        self.tx_param_align_ref.setText('-1')
        self.tx_param_align_ref.setFont(self.font2)

        hbox_param_align_ref = QHBoxLayout()
        hbox_param_align_ref.addWidget(lb_param_align_ref)
        hbox_param_align_ref.addWidget(self.tx_param_align_ref)
        hbox_param_align_ref.addWidget(lb_param_align_roi)
        hbox_param_align_ref.addWidget(self.tx_param_align_roi)
        hbox_param_align_ref.setAlignment(QtCore.Qt.AlignLeft)

        vbox_param_align = QVBoxLayout()
        vbox_param_align.addWidget(lb_param_align0)
        vbox_param_align.addLayout(hbox_param_align_flag)
        vbox_param_align.addLayout(hbox_param_align_ref)
        vbox_param_align.addWidget(self.lb_empty)
        vbox_param_align.setAlignment(QtCore.Qt.AlignTop)

        #gpbox.setLayout(vbox_param_align)
        return vbox_param_align

    def layout_param_fitting(self):
        #gpbox = QGroupBox('Fitting')
        #gpbox.setFont(self.font1)
        lb_param_fit_iter0 = QLabel()
        lb_param_fit_iter0.setFont(self.font1)
        lb_param_fit_iter0.setText('Fitting')
        lb_param_fit_iter0.setFixedWidth(140)

        # Fit edge
        self.chkbox_fit_pre_edge = QCheckBox('Fit pre-edge')
        #self.chkbox_fit_pre_edge.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.chkbox_fit_pre_edge.setFont(self.font2)
        self.chkbox_fit_pre_edge.setChecked(True)
        self.chkbox_fit_pre_edge.setFixedWidth(140)

        self.chkbox_fit_post_edge = QCheckBox('Fit post-edge')
        #self.chkbox_fit_post_edge.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.chkbox_fit_post_edge.setFont(self.font2)
        self.chkbox_fit_post_edge.setChecked(True)
        self.chkbox_fit_post_edge.setFixedWidth(140)

        hbox_param_fit_edge = QHBoxLayout()
        hbox_param_fit_edge.addWidget(self.chkbox_fit_pre_edge)
        hbox_param_fit_edge.addWidget(self.chkbox_fit_post_edge)
        hbox_param_fit_edge.setAlignment(QtCore.Qt.AlignLeft)

        # Fit energy range
        lb_param_fit_range = QLabel()
        lb_param_fit_range.setFont(self.font2)
        lb_param_fit_range.setText('Eng. range')
        lb_param_fit_range.setFixedWidth(80)

        lb_param_fit_range_start = QLabel()
        lb_param_fit_range_start.setFont(self.font2)
        lb_param_fit_range_start.setText('start:')
        lb_param_fit_range_start.setFixedWidth(55)

        lb_param_fit_range_end = QLabel()
        lb_param_fit_range_end.setFont(self.font2)
        lb_param_fit_range_end.setText('  end:')
        lb_param_fit_range_end.setFixedWidth(60)

        self.tx_param_fit_range_s = QLineEdit()
        self.tx_param_fit_range_s.setFixedWidth(60)
        self.tx_param_fit_range_s.setFont(self.font2)

        self.tx_param_fit_range_e = QLineEdit()
        self.tx_param_fit_range_e.setFixedWidth(60)
        self.tx_param_fit_range_e.setFont(self.font2)

        hbox_param_fit_range = QHBoxLayout()
        hbox_param_fit_range.addWidget(lb_param_fit_range)
        hbox_param_fit_range.addWidget(lb_param_fit_range_start)
        hbox_param_fit_range.addWidget(self.tx_param_fit_range_s)
        hbox_param_fit_range.addWidget(lb_param_fit_range_end)
        hbox_param_fit_range.addWidget(self.tx_param_fit_range_e)
        hbox_param_fit_range.setAlignment(QtCore.Qt.AlignLeft)

        # XANES fitting
        lb_param_fit_iter = QLabel()
        lb_param_fit_iter.setFont(self.font2)
        lb_param_fit_iter.setText('Fitting')
        lb_param_fit_iter.setFixedWidth(140)

        self.rd_param_fit_iter_yes = QRadioButton('Yes')
        self.rd_param_fit_iter_yes.setFixedWidth(80)
        self.rd_param_fit_iter_yes.setChecked(True)

        self.rd_param_fit_iter_no = QRadioButton('No')
        self.rd_param_fit_iter_no.setFixedWidth(80)

        self.fit_iter_group = QButtonGroup()
        self.fit_iter_group.setExclusive(True)
        self.fit_iter_group.addButton(self.rd_param_fit_iter_yes)
        self.fit_iter_group.addButton(self.rd_param_fit_iter_no)
        self.fit_iter_group.setExclusive(True)

        hbox_param_fit_iter_flag = QHBoxLayout()
        hbox_param_fit_iter_flag.addWidget(lb_param_fit_iter)
        hbox_param_fit_iter_flag.addWidget(self.rd_param_fit_iter_yes)
        hbox_param_fit_iter_flag.addWidget(self.rd_param_fit_iter_no)
        hbox_param_fit_iter_flag.setAlignment(QtCore.Qt.AlignLeft)

        # leaning rate
        self.chkbox_fit_iter = QCheckBox('Fit iter')
        # self.chkbox_fit_post_edge.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.chkbox_fit_iter.setFont(self.font2)
        self.chkbox_fit_iter.setChecked(True)
        self.chkbox_fit_iter.setFixedWidth(140)

        lb_param_iter_rate = QLabel()
        lb_param_iter_rate.setFont(self.font2)
        lb_param_iter_rate.setText('Updating rate:')
        lb_param_iter_rate.setFixedWidth(140)

        self.tx_param_iter_rate = QLineEdit()
        self.tx_param_iter_rate.setFixedWidth(60)
        self.tx_param_iter_rate.setText('0.005')
        self.tx_param_iter_rate.setFont(self.font2)

        # Iters
        lb_param_iter_num = QLabel()
        lb_param_iter_num.setFont(self.font2)
        lb_param_iter_num.setText(' # Iter:')
        lb_param_iter_num.setFixedWidth(60)

        self.tx_param_iter_num = QLineEdit()
        self.tx_param_iter_num.setFixedWidth(60)
        self.tx_param_iter_num.setText('5')
        self.tx_param_iter_num.setFont(self.font2)

        hbox_param_fit_iter_p = QHBoxLayout()
        hbox_param_fit_iter_p.addWidget(lb_param_iter_rate)
        hbox_param_fit_iter_p.addWidget(self.tx_param_iter_rate)
        hbox_param_fit_iter_p.addWidget(lb_param_iter_num)
        hbox_param_fit_iter_p.addWidget(self.tx_param_iter_num)
        hbox_param_fit_iter_p.setAlignment(QtCore.Qt.AlignLeft)

        # Bounds
        lb_param_iter_bounds = QLabel()
        lb_param_iter_bounds.setFont(self.font2)
        lb_param_iter_bounds.setText('Bounds to:')
        lb_param_iter_bounds.setFixedWidth(80)

        lb_param_iter_bounds_low = QLabel()
        lb_param_iter_bounds_low.setFont(self.font2)
        lb_param_iter_bounds_low.setText('low:')
        lb_param_iter_bounds_low.setFixedWidth(55)

        self.tx_param_iter_bounds_low = QLineEdit()
        self.tx_param_iter_bounds_low.setFixedWidth(60)
        self.tx_param_iter_bounds_low.setText('0')
        self.tx_param_iter_bounds_low.setFont(self.font2)

        lb_param_iter_bounds_high = QLabel()
        lb_param_iter_bounds_high.setFont(self.font2)
        lb_param_iter_bounds_high.setText('  high:')
        lb_param_iter_bounds_high.setFixedWidth(60)

        self.tx_param_iter_bounds_high = QLineEdit()
        self.tx_param_iter_bounds_high.setFixedWidth(60)
        self.tx_param_iter_bounds_high.setText('1')
        self.tx_param_iter_bounds_high.setFont(self.font2)

        hbox_param_fit_iter_bounds = QHBoxLayout()
        hbox_param_fit_iter_bounds.addWidget(lb_param_iter_bounds)
        hbox_param_fit_iter_bounds.addWidget(lb_param_iter_bounds_low)
        hbox_param_fit_iter_bounds.addWidget(self.tx_param_iter_bounds_low)
        hbox_param_fit_iter_bounds.addWidget(lb_param_iter_bounds_high)
        hbox_param_fit_iter_bounds.addWidget(self.tx_param_iter_bounds_high)
        hbox_param_fit_iter_bounds.setAlignment(QtCore.Qt.AlignLeft)

        lb_param_iter_lambda = QLabel()
        lb_param_iter_lambda.setFont(self.font2)
        lb_param_iter_lambda.setText('lambda:')
        lb_param_iter_lambda.setFixedWidth(140)

        self.tx_param_iter_lambda = QLineEdit()
        self.tx_param_iter_lambda.setFixedWidth(60)
        self.tx_param_iter_lambda.setText('0.01')
        self.tx_param_iter_lambda.setFont(self.font2)

        hbox_param_fit_iter_lambda = QHBoxLayout()
        hbox_param_fit_iter_lambda.addWidget(lb_param_iter_lambda)
        hbox_param_fit_iter_lambda.addWidget(self.tx_param_iter_lambda)
        hbox_param_fit_iter_lambda.setAlignment(QtCore.Qt.AlignLeft)

        vbox_param_fit = QVBoxLayout()
        vbox_param_fit.addWidget(lb_param_fit_iter0)
        vbox_param_fit.addLayout(hbox_param_fit_edge)
        vbox_param_fit.addLayout(hbox_param_fit_range)
        vbox_param_fit.addWidget(self.chkbox_fit_iter)
        vbox_param_fit.addLayout(hbox_param_fit_iter_p)
        vbox_param_fit.addLayout(hbox_param_fit_iter_bounds)
        vbox_param_fit.addLayout(hbox_param_fit_iter_lambda)
        vbox_param_fit.addWidget(self.lb_empty)
        vbox_param_fit.setAlignment(QtCore.Qt.AlignTop)

        #gpbox.setLayout(vbox_param_fit)
        return vbox_param_fit

    def layout_param_mask(self):
        #gpbox = QGroupBox('Mask')
        #gpbox.setFont(self.font1)
        lb_param_smart_mask0 = QLabel()
        lb_param_smart_mask0.setFont(self.font1)
        lb_param_smart_mask0.setText('Cluster Mask comp.')
        lb_param_smart_mask0.setFixedWidth(140)
        # smart mask
        lb_param_smart_mask = QLabel()
        lb_param_smart_mask.setFont(self.font2)
        lb_param_smart_mask.setText('Cluster Mask comp.')
        lb_param_smart_mask.setFixedWidth(140)

        self.tx_param_smart_mask_comp = QLineEdit()
        self.tx_param_smart_mask_comp.setFixedWidth(60)
        self.tx_param_smart_mask_comp.setText('0')
        self.tx_param_smart_mask_comp.setFont(self.font2)

        hbox_param_smart_mask_comp = QHBoxLayout()
        hbox_param_smart_mask_comp.addWidget(lb_param_smart_mask)
        hbox_param_smart_mask_comp.addWidget(self.tx_param_smart_mask_comp)
        hbox_param_smart_mask_comp.setAlignment(QtCore.Qt.AlignLeft)

        # XANES fitting mask
        lb_param_threshold = QLabel()
        lb_param_threshold.setFont(self.font2)
        lb_param_threshold.setText('Threshold: thickness:')
        lb_param_threshold.setFixedWidth(140)

        self.tx_param_threshold_thick = QLineEdit()
        self.tx_param_threshold_thick.setFixedWidth(60)
        self.tx_param_threshold_thick.setText('0')
        self.tx_param_threshold_thick.setFont(self.font2)

        lb_param_threshold_error = QLabel()
        lb_param_threshold_error.setFont(self.font2)
        lb_param_threshold_error.setText('  error:')
        lb_param_threshold_error.setFixedWidth(60)

        self.tx_param_threshold_error = QLineEdit()
        self.tx_param_threshold_error.setFixedWidth(60)
        self.tx_param_threshold_error.setText('100')
        self.tx_param_threshold_error.setFont(self.font2)

        hbox_param_fit_threshold_mask = QHBoxLayout()
        hbox_param_fit_threshold_mask.addWidget(lb_param_threshold)
        hbox_param_fit_threshold_mask.addWidget(self.tx_param_threshold_thick)
        hbox_param_fit_threshold_mask.addWidget(lb_param_threshold_error)
        hbox_param_fit_threshold_mask.addWidget(self.tx_param_threshold_error)
        hbox_param_fit_threshold_mask.setAlignment(QtCore.Qt.AlignLeft)

        vbox_param_mask = QVBoxLayout()
        vbox_param_mask.addWidget(lb_param_smart_mask0)
        vbox_param_mask.addLayout(hbox_param_smart_mask_comp)
        vbox_param_mask.addLayout(hbox_param_fit_threshold_mask)
        vbox_param_mask.addWidget(self.lb_empty)
        vbox_param_mask.setAlignment(QtCore.Qt.AlignTop)

        #gpbox.setLayout(vbox_param_mask)
        return vbox_param_mask

    def layout_param_ref(self):
        lb_param_ref = QLabel()
        lb_param_ref.setFont(self.font1)
        lb_param_ref.setText('Reference spec. & X-ray energy')
        lb_param_ref.setFixedWidth(240)

        # ref
        self.pb_param_load_ref = QPushButton('Load Ref.')
        self.pb_param_load_ref.setFont(self.font2)
        self.pb_param_load_ref.clicked.connect(self.load_reference)
        self.pb_param_load_ref.setFixedWidth(80)

        self.pb_param_plot_ref = QPushButton('Plot Ref.')
        self.pb_param_plot_ref.setFont(self.font2)
        self.pb_param_plot_ref.clicked.connect(self.plot_reference)
        self.pb_param_plot_ref.setFixedWidth(80)

        self.pb_param_reset_ref = QPushButton('Reset Ref.')
        self.pb_param_reset_ref.setFont(self.font2)
        self.pb_param_reset_ref.clicked.connect(self.reset_reference)
        self.pb_param_reset_ref.setFixedWidth(80)

        self.pb_param_eng = QPushButton('Load XEng')
        self.pb_param_eng.setFont(self.font2)
        self.pb_param_eng.clicked.connect(self.load_energy)
        self.pb_param_eng.setFixedWidth(80)

        self.lb_param_ref_info = QLabel()
        self.lb_param_ref_info.setFont(self.font2)
        self.lb_param_ref_info.setStyleSheet('color: rgb(200, 50, 50);')
        self.lb_param_ref_info.setText('Reference spectrum: ')
        self.lb_param_ref_info.setFixedWidth(450)

        self.lb_param_eng_info = QLabel()
        self.lb_param_eng_info.setFont(self.font2)
        self.lb_param_eng_info.setStyleSheet('color: rgb(200, 50, 50);')
        self.lb_param_eng_info.setText('Energy: ')
        self.lb_param_eng_info.setFixedWidth(450)

        hbox_param_load_ref = QHBoxLayout()
        hbox_param_load_ref.addWidget(self.pb_param_load_ref)
        hbox_param_load_ref.addWidget(self.pb_param_plot_ref)
        hbox_param_load_ref.addWidget(self.pb_param_reset_ref)
        hbox_param_load_ref.addWidget(self.pb_param_eng)
        hbox_param_load_ref.setAlignment(QtCore.Qt.AlignLeft)

        vbox_param_load_ref = QVBoxLayout()
        vbox_param_load_ref.addWidget(lb_param_ref)
        vbox_param_load_ref.addLayout(hbox_param_load_ref)
        vbox_param_load_ref.addWidget(self.lb_param_ref_info)
        vbox_param_load_ref.addWidget(self.lb_param_eng_info)
        vbox_param_load_ref.setAlignment(QtCore.Qt.AlignLeft)

        return vbox_param_load_ref

    def layout_param_colormix(self):
        lb_param_colormix = QLabel()
        lb_param_colormix.setFont(self.font1)
        lb_param_colormix.setText('Color Mix')
        lb_param_colormix.setFixedWidth(240)

        lb_param_color = QLabel()
        lb_param_color.setFont(self.font2)
        lb_param_color.setText('Color Mix')
        lb_param_color.setFixedWidth(80)

        self.tx_param_color = QLineEdit()
        self.tx_param_color.setFixedWidth(60)
        self.tx_param_color.setText('r, g, b')
        self.tx_param_color.setFont(self.font2)

        lb_param_color_hint = QLabel()
        lb_param_color_hint.setFont(self.font2)
        lb_param_color_hint.setText("   choose from 'r, g, b, c, p, y'")

        hbox_param_colormix = QHBoxLayout()
        hbox_param_colormix.addWidget(lb_param_color)
        hbox_param_colormix.addWidget(self.tx_param_color)
        hbox_param_colormix.addWidget(lb_param_color_hint)
        hbox_param_colormix.setAlignment(QtCore.Qt.AlignLeft)

        vbox_param_colormix = QVBoxLayout()
        vbox_param_colormix.addWidget(lb_param_colormix)
        vbox_param_colormix.addLayout(hbox_param_colormix)
        vbox_param_colormix.addWidget(self.lb_empty)
        vbox_param_colormix.setAlignment(QtCore.Qt.AlignLeft)

        return vbox_param_colormix


    def layout_param_execute(self):
        self.pb_param_save_param = QPushButton('Save parameter')
        self.pb_param_save_param.setFont(self.font2)
        self.pb_param_save_param.clicked.connect(self.save_fit_param)
        self.pb_param_save_param.setFixedWidth(120)

        self.pb_param_load_param = QPushButton('Load parameter')
        self.pb_param_load_param.setFont(self.font2)
        self.pb_param_load_param.clicked.connect(self.load_fit_param)
        self.pb_param_load_param.setFixedWidth(120)

        self.pb_param_batch_fit = QPushButton('Batch fitting')
        self.pb_param_batch_fit.setFont(self.font2)
        self.pb_param_batch_fit.clicked.connect(self.batch_fitting)
        #self.pb_param_batch_fit.clicked.connect(self.run_command)
        self.pb_param_batch_fit.setFixedWidth(120)

        self.tx_param_output = QPlainTextEdit()
        self.tx_param_output.resize(6, 200)

        self.lb_execute_output = QLabel()
        self.lb_execute_output.setFixedWidth(800)
        self.lb_execute_output.setStyleSheet('color: rgb(200, 50, 50);')

        hbox_param_execute = QHBoxLayout()
        hbox_param_execute.addWidget(self.pb_param_save_param)
        hbox_param_execute.addWidget(self.pb_param_load_param)
        hbox_param_execute.addWidget(self.pb_param_batch_fit)
        hbox_param_execute.setAlignment(QtCore.Qt.AlignLeft)

        vbox_param_execute = QVBoxLayout()
        vbox_param_execute.addLayout(hbox_param_execute)
        vbox_param_execute.addWidget(self.lb_execute_output)
        #vbox_param_execute.addWidget(self.tx_param_output)
        vbox_param_execute.setAlignment(QtCore.Qt.AlignTop)

        return vbox_param_execute

    def open_folder(self):
        options = QFileDialog.Option()
        options |= QFileDialog.DontUseNativeDialog
        file_type = '*.*'
        fn, _ = QFileDialog.getOpenFileName(xanes, "QFileDialog.getOpenFileName()", "", '*', options=options)
        fn_tmp = fn.split('/')
        self.file_path = '/'.join(t for t in fn_tmp[:-1])
        #self.file_path = QFileDialog.getExistingDirectory(None, 'Select a folder:', self.fpath, QFileDialog.ShowDirsOnly)
        self.tx_param_folder.setText(self.file_path)
        self.tx_param_file_prefix.setText(fn_tmp[-1][:3])
        self.tx_param_file_type.setText('.' + fn_tmp[-1].split('.')[-1])
        self.lb_execute_output.setText('')
        self.lb_msg.setText(f'Message: ')



    def load_xanes_image_file(self):
        self.load_file_successful = 0
        try:
            self.file_path = self.tx_param_folder.text()
            self.file_prefix = self.tx_param_file_prefix.text()
            self.file_type = self.tx_param_file_type.text()
            self.xanes_files = pyxas.retrieve_file_type(self.file_path, self.file_prefix, self.file_type)
            print('file load in sequence:')
            for fn in self.xanes_files:
                print(fn.split("/")[-1])
            msg = f'{self.xanes_files[0].split("/")[-1]}  ...  {self.xanes_files[-1].split("/")[-1]}'
            self.lb_msg.setText(f'Message: {len(self.xanes_files)} files loaded:   [{msg}]')
            self.load_file_successful = 1
        except:
            self.load_file_successful = 0

    def load_reference(self):
        self.lb_param_ref_info.setStyleSheet('color: rgb(200, 50, 50);')
        self.load_reference_successful = 0
        options = QFileDialog.Option()
        options |= QFileDialog.DontUseNativeDialog
        file_type = 'txt files (*.txt)'
        fn, _ = QFileDialog.getOpenFileName(xanes, "QFileDialog.getOpenFileName()", "", file_type, options=options)
        if fn:
            try:
                print(fn)
                fn_tmp = fn.split('/')
                fn_ref = '/'.join(t for t in fn_tmp[-6:])
                fn_ref = f'.../{fn_ref}'
                print(f'selected reference: {fn_ref}')
                self.lb_param_ref_info.setText(self.lb_param_ref_info.text() + '\n' + f'ref #{self.num_ref}: ' + fn_ref)
                QApplication.processEvents()
                self.spectrum_ref[f'ref{self.num_ref}'] = np.loadtxt(fn)
                self.num_ref += 1
                self.load_reference_successful = 1
            except:
                print('un-supported xanes reference format')
                self.load_reference_successful = 0

    def load_energy(self):
        self.load_eng_successful = 0
        options = QFileDialog.Option()
        options |= QFileDialog.DontUseNativeDialog
        file_type = '*.*'
        fn, _ = QFileDialog.getOpenFileName(xanes, "QFileDialog.getOpenFileName()", "", ' txt files (*.txt)', options=options)
        try:
            self.eng = np.loadtxt(fn)
            n_eng = len(self.eng)
            self.lb_param_eng_info.setText(f'Energy: {n_eng} energies: {self.eng[0]}, {self.eng[1]},   ...   {self.eng[-1]} keV')
            self.load_eng_successful = 1
        except:
            self.lb_param_eng_info = 'Energy load fails'
            self.load_eng_successful = 0

    def plot_reference(self):
        plt.figure()
        legend = []
        try:
            for i in range(self.num_ref):
                plot_label = f'ref_{i}'
                spec = self.spectrum_ref[f'ref{i}']
                line, = plt.plot(spec[:, 0], spec[:, 1], label=plot_label)
                legend.append(line)
            print(legend)
            plt.legend(handles=legend)
            plt.show()
        except:
            self.lb_param_ref_info = 'un-recognized reference spectrum format'

    def reset_reference(self):
        self.num_ref = 0
        self.lb_param_ref_info.setText('Reference spectrum:')
        self.spectrum_ref = {}
        self.elem_label = []

    def save_fit_param(self):
        self.save_fit_param_successful = 0
        save_successful = 1
        self.fit_param = {}
        try:
            self.fit_param['align_flag'] = 1 if self.rd_param_align_yes.isChecked() else 0
            self.fit_param['align_ref_index'] = int(self.tx_param_align_ref.text())

            roi_ratio = self.tx_param_align_roi.text()
            roi_ratio = float(roi_ratio) if roi_ratio else 1
            self.fit_param['roi_ratio'] = roi_ratio
            self.tx_param_align_roi.setText(str(roi_ratio))

            # fit_eng
            try:
                fit_eng_s = self.tx_param_fit_range_s.text()
                fit_eng_e = self.tx_param_fit_range_e.text()
                fit_eng_s = float(fit_eng_s) if fit_eng_s else self.eng[0]
                fit_eng_e = float(fit_eng_e) if fit_eng_e else self.eng[-1]
                self.fit_param['fit_eng'] = [fit_eng_s, fit_eng_e]
                self.tx_param_fit_range_s.setText(f'{fit_eng_s:2.4f}')
                self.tx_param_fit_range_e.setText(f'{fit_eng_e:2.4f}')
            except:
                print('errors in saving "fitting energy range"')
                save_successful = 0

            # fit_iter_bound
            fit_bound_s = self.tx_param_iter_bounds_low.text()
            fit_bound_e = self.tx_param_iter_bounds_high.text()
            fit_bound_s = float(fit_bound_s) if fit_bound_s else 0
            fit_bound_e = float(fit_bound_e) if fit_bound_s else 100
            self.fit_param['fit_iter_bound'] = [fit_bound_s, fit_bound_e]
            self.tx_param_iter_bounds_low.setText(str(fit_bound_s))
            self.tx_param_iter_bounds_high.setText(str(fit_bound_e))

            self.fit_param['fit_iter_flag'] = 1 if self.chkbox_fit_iter.isChecked() else 0
            self.fit_param['fit_iter_learning_rate'] = float(self.tx_param_iter_rate.text())
            self.fit_param['fit_iter_num'] = int(self.tx_param_iter_num.text())
            self.fit_param['fit_iter_lambda'] = float(self.tx_param_iter_lambda.text())
            self.fit_param['fit_pre_edge_flag'] = 1 if self.chkbox_fit_pre_edge.isChecked() else 0
            self.fit_param['fit_post_edge_flag'] = 1 if self.chkbox_fit_post_edge.isChecked() else 0
            self.fit_param['fit_mask_thickness_threshold'] = float(self.tx_param_threshold_thick.text())
            self.fit_param['fit_mask_cost_threshold'] = float(self.tx_param_threshold_error.text())

            cluster_comp = int(self.tx_param_smart_mask_comp.text())
            self.fit_param['mask_xanes_flag'] = 1 if cluster_comp else 0
            self.fit_param['n_comp'] = cluster_comp

            self.fit_param['norm_edge_method'] = 'new' if self.chkbox_param_fit_edge_method.isChecked() else 'old'
            self.fit_param['norm_txm_flag'] = 1 if self.rd_param_txm_norm_yes.isChecked() else 0

            if self.rd_param_fit_edge_yes.isChecked():
                try:
                    pre_edge_s = self.tx_param_pre_s.text()
                    pre_edge_s = float(pre_edge_s) if pre_edge_s else self.eng[0]
                    pre_edge_e = self.tx_param_pre_e.text()
                    pre_edge_e = float(pre_edge_e) if pre_edge_e else pre_edge_s + 0.02
                    self.fit_param['pre_edge'] = [pre_edge_s, pre_edge_e]
                    self.tx_param_pre_s.setText(f'{pre_edge_s:2.4f}')
                    self.tx_param_pre_e.setText(f'{pre_edge_e:2.4f}')
                except:
                    print('fails in save pre-edge energy')
                    save_successful = 0

                try:
                    post_edge_e = self.tx_param_post_e.text()
                    post_edge_e = float(post_edge_e) if post_edge_e else self.eng[-1]
                    post_edge_s = self.tx_param_post_s.text()
                    post_edge_s = float(post_edge_s) if post_edge_s else post_edge_e - 0.2
                    self.fit_param['post_edge'] = [post_edge_s, post_edge_e]
                    self.tx_param_post_s.setText(f'{post_edge_s:2.4f}')
                    self.tx_param_post_e.setText(f'{post_edge_e:2.4f}')
                except:
                    print('fails in save post-edge energy')
                    save_successful = 0

            self.fit_param['regulation_flag'] = 1 if self.rd_param_reg_edge_yes.isChecked() else 0
            peak_max = self.tx_param_reg_peak.text()
            peak_max = float(peak_max) if peak_max else 1.6
            self.fit_param['regulation_designed_max'] = peak_max
            self.tx_param_reg_peak.setText(f'{peak_max:2.4f}')

            peak_width = self.tx_param_reg_gamma.text()
            peak_width = float(peak_width) if peak_width else 0.05
            self.fit_param['regulation_gamma'] = peak_width
            self.tx_param_reg_gamma.setText(f'{peak_width:2.4f}')

            self.num_cpu = min(int(self.tx_param_cpu.text()), round(cpu_count() * 0.8))
            self.tx_param_cpu.setText(str(self.num_cpu))

            if self.rd_tif.isChecked():
                self.fit_param['file_type'] = 'tiff'
            else:
                self.fit_param['file_type'] = 'h5'
            self.fit_param['hdf_attr'] = self.tx_param_hdf.text()
            self.fit_param['num_cpu'] = self.num_cpu

            self.fit_param['color'] = self.tx_param_color.text()

        except:
            save_successful = 0
        # save to .csv
        if save_successful:
            options = QFileDialog.Option()
            options |= QFileDialog.DontUseNativeDialog
            file_type = 'csv (*.csv)'
            fn, _ = QFileDialog.getSaveFileName(self, 'Save File', "", file_type, options=options)
            if fn.split('.')[-1] != 'csv':
                fn += '.csv'
            pyxas.save_xanes_fit_param_file(self.fit_param, fn)
            self.lb_execute_output.setText(f'{fn} saved')
            self.tx_param_output.appendPlainText(f'{fn} saved')
            self.save_fit_param_successful = 1
        else:
            print('fails to save fitting parameter')
            self.lb_execute_output.setText('fails to save fitting parameter')
            self.save_fit_param_successful = 0


    def load_fit_param(self):
        self.load_fit_param_successful = 0
        try:
            options = QFileDialog.Option()
            options |= QFileDialog.DontUseNativeDialog
            file_type = ' csv files (*.csv)'
            fn, _ = QFileDialog.getOpenFileName(xanes, "QFileDialog.getOpenFileName()", "", file_type, options=options)
            if fn:
                self.fit_param = pyxas.load_xanes_fit_param_file(fn, num_items=0)
                txt = self.lb_execute_output.text() + f'\nfit_param loaded: {fn}'
                self.lb_execute_output.setText(txt)
                self.load_fit_param_successful = 1


                if self.fit_param['align_flag']:
                    self.rd_param_align_yes.setChecked(1)
                    align_index = int(self.fit_param['align_ref_index'])
                    self.tx_param_align_ref.setText(str(align_index))
                    roi_ratio = self.fit_param['roi_ratio']
                    self.tx_param_align_roi.setText(str(roi_ratio))
                else:
                    self.rd_param_align_no.setChecked(1)

                fit_pre_edge_flag = self.fit_param['fit_pre_edge_flag']
                if fit_pre_edge_flag:
                    self.chkbox_fit_pre_edge.setChecked(1)
                else:
                    self.chkbox_fit_pre_edge.setChecked(0)
                fit_post_edge_flag = self.fit_param['fit_pre_edge_flag']
                if fit_post_edge_flag:
                    self.chkbox_fit_post_edge.setChecked(1)
                else:
                    self.chkbox_fit_post_edge.setChecked(0)

                # fit_eng
                try:
                    fit_eng_s = self.fit_param['fit_eng'][0]
                    self.tx_param_fit_range_s.setText(f'{fit_eng_s:2.4f}')
                    fit_eng_e = self.fit_param['fit_eng'][1]
                    self.tx_param_fit_range_e.setText(f'{fit_eng_e:2.4f}')
                except:
                    pass

                # fit_iter_bound
                fit_iter_flag = self.fit_param['fit_iter_flag']
                if fit_iter_flag:
                    self.chkbox_fit_iter.setChecked(1)
                    fit_bound_s, fit_bound_e = self.fit_param['fit_iter_bound']
                    self.tx_param_iter_bounds_low.setText(str(fit_bound_s))
                    self.tx_param_iter_bounds_high.setText(str(fit_bound_e))
                    learn_rate = float(self.fit_param['fit_iter_learning_rate'])
                    self.tx_param_iter_rate.setText(f'{learn_rate:1.4f}')
                    fit_iter_num = int(self.fit_param['fit_iter_num'])
                    self.tx_param_iter_num.setText(str(fit_iter_num))
                    try:
                        fit_iter_lambda = float(self.fit_param['fit_iter_lambda'])
                    except:
                        fit_iter_lambda = 0.5
                    self.tx_param_iter_lambda.setText(str(fit_iter_lambda))
                else:
                    self.chkbox_fit_iter.setChecked(0)

                self.tx_param_threshold_thick.setText(str(self.fit_param['fit_mask_thickness_threshold']))
                self.tx_param_threshold_error.setText(str(self.fit_param['fit_mask_cost_threshold']))

                mask_xanes_flag = self.fit_param['mask_xanes_flag']
                if mask_xanes_flag:
                    n_comp = self.fit_param['n_comp']
                else:
                    n_comp = 0
                self.tx_param_smart_mask_comp.setText(str(n_comp))

                if self.fit_param['norm_edge_method'] == 'new':
                    self.chkbox_param_fit_edge_method.setChecked(1)
                else:
                    self.chkbox_param_fit_edge_method.setChecked(0)

                if self.fit_param['norm_txm_flag']:
                    self.rd_param_txm_norm_yes.setChecked(1)
                else:
                    self.rd_param_txm_norm_no.setChecked(1)

                try:
                    pre_edge_s, pre_edge_e = self.fit_param['pre_edge']
                    self.tx_param_pre_s.setText(f'{pre_edge_s:2.4f}')
                    self.tx_param_pre_e.setText(f'{pre_edge_e:2.4f}')
                    self.rd_param_fit_edge_yes.setChecked(1)
                    self.chkbox_fit_pre_edge.setChecked(0)
                except:
                    self.rd_param_fit_edge_no.setChecked(1)
                    self.fit_param['fit_pre_edge_flag'] = 0

                try:
                    post_edge_s, post_edge_e = self.fit_param['post_edge']
                    self.tx_param_post_s.setText(f'{post_edge_s:2.4f}')
                    self.tx_param_post_e.setText(f'{post_edge_e:2.4f}')
                    self.rd_param_fit_edge_yes.setChecked(1)
                    self.chkbox_fit_post_edge.setChecked(0)
                except:
                    self.rd_param_fit_edge_no.setChecked(1)
                    self.fit_param['fit_post_edge_flag'] = 0


                regulation_flag = self.fit_param['regulation_flag']
                if regulation_flag:
                    self.rd_param_reg_edge_yes.setChecked(1)
                    peak_max = self.fit_param['regulation_designed_max']
                    self.tx_param_reg_peak.setText(f'{peak_max:2.4f}')

                    peak_width = self.fit_param['regulation_gamma']
                    self.tx_param_reg_gamma.text()
                    self.tx_param_reg_gamma.setText(f'{peak_width:2.4f}')
                else:
                    self.rd_param_reg_edge_no.setChecked(1)


                num_cpu = int(self.fit_param['num_cpu'])
                self.tx_param_cpu.setText(str(num_cpu))

                if self.fit_param['file_type'] == 'tiff':
                    self.rd_tif.setChecked(1)
                elif self.fit_param['file_type'] == 'h5':
                    self.rd_hdf.setChecked(1)
                    self.tx_param_hdf.setText(str(self.fit_param['hdf_attr']))
                else:
                    print('un-recongnized file type')

                self.tx_param_color.setText(self.fit_param['color'])
        except:
            self.load_fit_param_successful = 0




    def run_command(self):
        stdouterr = os.popen4(self.batch_fitting())[1].read()
        self.self.lb_execute_output.setText(stdouterr)


    def batch_fitting(self):
        self.lb_execute_output.setText('Fitting in progress ... ')
        if self.load_eng_successful and self.load_file_successful and self.load_reference_successful:
            if self.save_fit_param_successful or self.load_fit_param_successful:
                fit_param = self.fit_param
                xanes_eng = self.eng
                file_path = self.file_path
                file_type = self.file_type
                file_prefix = self.file_prefix
                spectrum_ref = self.spectrum_ref
                try:
                    num_cpu = int(self.tx_param_cpu.text())
                except:
                    num_cpu = self.fit_param['num_cpu']
                try:
                    if num_cpu == 1:
                        pyxas.fit_2D_xanes_file(file_path, file_prefix,
                                                file_type, fit_param,
                                                xanes_eng, spectrum_ref,
                                                file_range=[], save_hdf=0)
                    else:
                        pyxas.fit_2D_xanes_file_mpi(file_path, file_prefix,
                                                file_type, fit_param,
                                                xanes_eng, spectrum_ref,
                                                file_range=[], save_hdf=0, num_cpu=num_cpu)
                except:
                    txt = 'something wrong in fitting'
                    self.lb_execute_output.setText(txt)
                    print(txt)
                    return 0
            self.lb_execute_output.setText('Fitting finished !')
            return 1
        else:
            txt = 'something wrong in fitting'
            self.lb_execute_output.setText(txt)
            print(txt)
            return 0



def longest_commonPrefix(strs):
    if not strs:
        return ''
    for i, lett in enumerate(zip(*strs)):
        if len(set(lett)) > 1:
            return strs[0][:i]
        else:
            return min(strs)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    xanes = App()
    xanes.show()
    sys.exit(app.exec_())
