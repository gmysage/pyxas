# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 09:07:42 2018

@author: Mingyuan Ge
Email: gmysage@gmail.com
"""

import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import threading
import time
import textwrap
from PyQt5 import QtGui
from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QRadioButton, QApplication,QWidget,
                             QLineEdit, QWidget, QPushButton, QLabel, QCheckBox, QGroupBox,
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
from pyxas.align3D import align_img, align_img_stackreg
from scipy.ndimage.interpolation import shift
from scipy.signal import medfilt2d, medfilt
from scipy.interpolate import interp1d, UnivariateSpline
from scipy import ndimage
from pyxas.xanes_util import (fit_2D_xanes_non_iter, fit_2D_xanes_iter, fit_2D_xanes_iter2, normalize_2D_xanes2, normalize_2D_xanes_old, normalize_1D_xanes, find_nearest, normalize_2D_xanes_regulation)


global xanes


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'XANES Control'
        screen_resolution = QApplication.desktop().screenGeometry()
        width, height = screen_resolution.width(), screen_resolution.height()
        self.width = 1020
        self.height = 800
        self.left = (width - self.width) // 2
        self.top = (height - self.height) // 2
        self.initUI()
        self.bkg_memory_check()
        self.default_layout()

    def bkg_memory_check(self):
        thread = threading.Thread(target=self.bkg_memory_check_run, args=())
        thread.daemon = True
        thread.start()

    def bkg_memory_check_run(self):
        pass
        '''
        while True:
            PID = os.getpid()
            py = psutil.Process(PID)
            MEM = list(psutil.virtual_memory())
            prog_used_mem = py.memory_info()[0]
            MEM.append(prog_used_mem)
            prog_used = '{:4.1f}'.format(MEM[-1] / 2. ** 30)  # unit in Gb
            tot_mem = '{:4.1f}'.format(MEM[0] / 2. ** 30)
            mem_pecent = '{:2.1f}'.format(100 - MEM[2])
            self.lb_pid_display.setText(str(PID))
            self.lb_mem_prog_used.setText(prog_used + ' / ' + tot_mem + 'Gb')
            self.lb_mem_avail.setText(mem_pecent + ' %')
            if float(mem_pecent) < 10:
                self.lb_mem_avail.setStyleSheet('color: rgb(200, 50, 50);')
            else:
                self.lb_mem_avail.setStyleSheet('color: rgb(0, 0, 0);')
            time.sleep(1)
        '''

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.font1 = QtGui.QFont('Arial', 11, QtGui.QFont.Bold)
        self.font2 = QtGui.QFont('Arial', 11, QtGui.QFont.Normal)
        self.fpath = os.getcwd()
        self.roi_file_id = 0
        self.spectrum_ref = {}
        grid = QGridLayout()
        gpbox_prep = self.layout_GP_prepare()
        gpbox_msg = self.layout_msg()
        gpbox_xanes = self.layout_xanes()

        grid.addWidget(gpbox_prep, 0, 1)
        grid.addLayout(gpbox_msg, 1, 1)
        grid.addWidget(gpbox_xanes, 2, 1)

        layout = QVBoxLayout()
        layout.addLayout(grid)
        layout.addWidget(QLabel())
        self.setLayout(layout)

    def default_layout(self):
        try:
            del self.img_xanes, self.img_update, self.xanes_2d_fit, self.xanes_fit_cost # self.img_bkg, self.img_bkg_removed, self.img_bkg_update
        except:
            pass
        default_img = np.zeros([1,500, 500])
        self.fn_raw_image = ''
        self.save_version = 0
        self.xanes_eng = np.array([0])
        self.img_xanes = deepcopy(default_img)
        self.img_update = deepcopy(default_img)
        self.current_img = deepcopy(default_img)
        self.img_regulation = deepcopy(default_img)
        self.dataset_used_for_fitting = 0
        self.img_colormix_raw = np.array([])
        self.edge_normalized_flag = 0
        self.mask1 = np.array([1])
        self.mask2 = np.array([1])
        self.mask3 = np.array([1])
        self.mask = np.array([1])
        self.smart_mask = np.array([1])
        self.smart_mask_comp = 2
        self.smart_mask_current = np.array([1])
        self.img_compress = np.array([1])
        self.img_labels = np.array([1])
        self.img_rm_noise = np.array([1])
        self.roi_spec = np.array([0])
        self.external_spec = np.array([0, 0])
        self.msg = ''
        self.smooth_param = {'flag': 0, 'kernal_size': 3}
        self.shift_list = []
        self.lst_roi.clear()
        self.roi_spec_dif = ''
        self.lb_eng1.setText('No energy data ...')
        self.data_summary= {}
        self.pb_mask1.setStyleSheet('color: rgb(0, 0, 0);')
        self.pb_mask2.setStyleSheet('color: rgb(0, 0, 0);')
        self.pb_mask3.setStyleSheet('color: rgb(0, 0, 0);')
        self.pb_smart_mask.setStyleSheet('color: rgb(0, 0, 0);')
        try:
            self.num_ref = self.num_ref
        except:
            self.num_ref= 0
        try:
            self.spectrum_ref = self.spectrum_ref
        except:
            self.spectrum_ref = {}
        self.fitting_method = 1
        self.xanes_2d_fit = None
        self.xanes_2d_fit_offset = 0
        self.xanes_fit_cost = 0
        self.img_pre_edge_sub_mean = np.array([1])
        self.elem_label = []
        self.figure = {}

        self.pb_plot_roi.setEnabled(False)
        self.pb_export_roi_fit.setEnabled(False)
        self.pb_colormix.setEnabled(True)
        self.pb_save.setEnabled(False)

        self.canvas1.cmax = 1
        self.canvas1.cmin = 0
        self.canvas1.current_img_index = 0
        self.canvas1.mask = np.array([1])
        self.canvas1.rgb_mask = np.array([1])
        self.canvas1.colorbar_on_flag = True
        self.canvas1.colormap = 'viridis'
        self.canvas1.title = []
        self.canvas1.draw_line = False
        self.canvas1.overlay_flag = True
        self.canvas1.x, self.y, = [], []
        self.canvas1.plot_label = ''
        self.canvas1.legend_flag = False
        self.canvas1.roi_list = {}
        self.canvas1.roi_color = {}
        self.canvas1.roi_count = 0
        self.canvas1.show_roi_flag = False
        self.canvas1.current_roi = [0, 0, 0, 0, '0'] # x1, y1, x2, y1, roi_name
        self.canvas1.color_list = ['red', 'brown', 'orange', 'olive', 'green', 'cyan', 'blue', 'pink', 'purple', 'gray']
        self.canvas1.current_color = 'red'
        self.canvas1.special_info = None

        self.cb1.setCurrentText('Raw image')
        QApplication.processEvents()
        count = self.cb1.count()
        for i in reversed(range(count)):
            self.cb1.removeItem(i)
        count = self.cb_color_channel.count()
        #for i in range(count):
        #    self.cb_color_channel.removeItem(i)
        self.cb_color_channel.clear()
        self.update_canvas_img()

    def layout_msg(self):
        self.lb_ip = QLabel()
        self.lb_ip.setFont(self.font2)
        self.lb_ip.setStyleSheet('color: rgb(200, 50, 50);')
        self.lb_ip.setText('File loaded:')

        self.lb_msg = QLabel()
        self.lb_msg.setFont(self.font1)
        self.lb_msg.setStyleSheet('color: rgb(200, 50, 50);')
        self.lb_msg.setText('Message:')

        vbox_msg = QVBoxLayout()
        vbox_msg.addWidget(self.lb_ip)
        vbox_msg.addWidget(self.lb_msg)
        vbox_msg.setAlignment(QtCore.Qt.AlignLeft)
        return vbox_msg

    def gpbox_system_info(self):
        lb_empty1 = QLabel()
        lb_empty1.setFixedWidth(80)
        lb_mem = QLabel()
        lb_mem.setFont(self.font1)
        lb_mem.setText('Memory:')
        lb_mem.setFixedWidth(80)

        lb_mem_prog_used = QLabel()
        lb_mem_prog_used.setFont(self.font2)
        lb_mem_prog_used.setText('Prog. used:')
        lb_mem_prog_used.setFixedWidth(80)

        lb_mem_avail = QLabel()
        lb_mem_avail.setFont(self.font2)
        lb_mem_avail.setText('Available:')
        lb_mem_avail.setFixedWidth(80)

        self.lb_mem_avail = QLabel()
        self.lb_mem_avail.setFont(self.font2)
        self.lb_mem_avail.setFixedWidth(60)

        self.lb_mem_prog_used = QLabel()
        self.lb_mem_prog_used.setFont(self.font2)
        self.lb_mem_prog_used.setFixedWidth(100)

        lb_pid = QLabel()
        lb_pid.setFont(self.font1)
        lb_pid.setText('PID:')
        lb_pid.setFixedWidth(80)

        self.lb_pid_display = QLabel()
        self.lb_pid_display.setFont(self.font2)
        self.lb_pid_display.setFixedWidth(80)

        hbox1 = QHBoxLayout()
        hbox1.addWidget(lb_pid)
        hbox1.addWidget(self.lb_pid_display)
        hbox1.setAlignment(QtCore.Qt.AlignLeft)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(lb_mem)
        hbox2.addWidget(lb_mem_prog_used)
        hbox2.addWidget(self.lb_mem_prog_used)

        hbox2.setAlignment(QtCore.Qt.AlignLeft)

        hbox3 = QHBoxLayout()
        hbox3.addWidget(lb_empty1)
        hbox3.addWidget(lb_mem_avail)
        hbox3.addWidget(self.lb_mem_avail)
        hbox3.setAlignment(QtCore.Qt.AlignLeft)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addLayout(hbox3)
        vbox.setAlignment(QtCore.Qt.AlignTop)
        return vbox

    def layout_GP_prepare(self):
        lb_empty = QLabel()
        lb_empty2 = QLabel()
        lb_empty2.setFixedWidth(10)
        lb_empty3 = QLabel()
        lb_empty3.setFixedWidth(100)

        gpbox = QGroupBox('Load image')
        gpbox.setFont(self.font1)

        lb_ld = QLabel()
        lb_ld.setFont(self.font2)
        lb_ld.setText('Image file:')
        lb_ld.setFixedWidth(100)

        lb_ld_eng = QLabel()
        lb_ld_eng.setFont(self.font2)
        lb_ld_eng.setText('Energy file:')
        lb_ld_eng.setFixedWidth(100)

        self.pb_ld = QPushButton('Load image')
        self.pb_ld.setToolTip('image type: .hdf, .tiff')
        self.pb_ld.setFont(self.font2)
        self.pb_ld.clicked.connect(self.load_image)
        self.pb_ld.setFixedWidth(140)
        self.pb_ld.setFixedWidth(140)

        self.pb_ld_eng = QPushButton('Load energy')
        self.pb_ld_eng.setToolTip('File: .txt')
        self.pb_ld_eng.setFont(self.font2)
        self.pb_ld_eng.clicked.connect(self.load_energy)
        self.pb_ld_eng.setFixedWidth(140)

        lb_eng = QLabel()
        lb_eng.setFont(self.font2)
        lb_eng.setText('XANES energy:')
        lb_eng.setFixedWidth(120)

        self.lb_eng1 = QLabel()
        self.lb_eng1.setFont(self.font2)
        self.lb_eng1.setText('No energy data ...')
        self.lb_eng1.setFixedWidth(400)

        self.lb_eng2 = QLabel()
        self.lb_eng2.setFont(self.font2)
        self.lb_eng2.setText('Manual input  (python command):')
        self.lb_eng2.setFixedWidth(245)

        self.tx_eng = QLineEdit()
        self.tx_eng.setFixedWidth(280)
        self.tx_eng.setFont(self.font2)
        # self.tx_eng.setVisible(False)

        self.pb_eng = QPushButton('Execute')
        self.pb_eng.setFont(self.font2)
        self.pb_eng.clicked.connect(self.manu_energy_input)
        self.pb_eng.setFixedWidth(80)

        self.pb_fiji = QPushButton('ImageJ')
        self.pb_fiji.setFont(self.font2)
        self.pb_fiji.clicked.connect(self.open_imagej)
        self.pb_fiji.setFixedWidth(80)

        self.pb_closeImg = QPushButton('Close Fig.')
        self.pb_closeImg.setFont(self.font2)
        self.pb_closeImg.clicked.connect(self.close_all_figures)
        self.pb_closeImg.setFixedWidth(80)

        lb_mod = QLabel()
        lb_mod.setFont(self.font2)
        lb_mod.setText('Image mode:')
        lb_mod.setFixedWidth(100)
        
        # radio button for loading image
        self.file_group = QButtonGroup()
        self.file_group.setExclusive(True)
        self.rd_hdf = QRadioButton('hdf')
        self.rd_hdf.setFixedWidth(60)
        self.rd_hdf.setChecked(True)
        self.rd_hdf.toggled.connect(self.select_file)

        self.rd_tif = QRadioButton('tif')
        self.rd_tif.setFixedWidth(60)
        self.rd_tif.toggled.connect(self.select_file)

        self.file_group.addButton(self.rd_hdf)
        self.file_group.addButton(self.rd_tif)
        self.file_group_eng = QButtonGroup()
        self.file_group_eng.setExclusive(True)

        self.rd_hdf_eng = QRadioButton('hdf')
        self.rd_hdf_eng.setFixedWidth(60)
        self.rd_hdf_eng.setChecked(True)
        self.rd_hdf_eng.toggled.connect(self.select_file)

        self.rd_txt_eng = QRadioButton('txt')
        self.rd_txt_eng.setFixedWidth(60)
        self.rd_txt_eng.toggled.connect(self.select_file)
        self.file_group_eng.addButton(self.rd_hdf_eng)
        self.file_group_eng.addButton(self.rd_txt_eng)

        lb_hdf_xanes = QLabel()
        lb_hdf_xanes.setFont(self.font2)
        lb_hdf_xanes.setText('Dataset for XANES:')
        lb_hdf_xanes.setFixedWidth(140)

        lb_hdf_eng = QLabel()
        lb_hdf_eng.setFont(self.font2)
        lb_hdf_eng.setText('Dataset for Energy:')
        lb_hdf_eng.setFixedWidth(140)
        
        lb_db_xanes = QLabel()
        lb_db_xanes.setFont(self.font2)
        lb_db_xanes.setText('Scan id:')
        lb_db_xanes.setFixedWidth(60)
        
        self.tx_db_xanes = QLineEdit()
        self.tx_db_xanes.setText('-1')
        self.tx_db_xanes.setFixedWidth(85)
        self.tx_db_xanes.setFont(self.font2)

        self.tx_hdf_xanes = QLineEdit()
        self.tx_hdf_xanes.setText('img_xanes')
        self.tx_hdf_xanes.setFixedWidth(85)
        self.tx_hdf_xanes.setFont(self.font2)

        self.tx_hdf_eng = QLineEdit()
        self.tx_hdf_eng.setText('X_eng')
        self.tx_hdf_eng.setFixedWidth(85)
        self.tx_hdf_eng.setFont(self.font2)

        self.type_group = QButtonGroup()
        self.type_group.setExclusive(True)
        self.rd_absp = QRadioButton('Absorption')
        self.rd_absp.setFont(self.font2)
        self.rd_absp.setFixedWidth(100)
        self.rd_absp.setChecked(True)
        self.rd_flrc = QRadioButton('Fluorescence')
        self.rd_flrc.setFont(self.font2)
        self.rd_flrc.setFixedWidth(120)
        self.rd_flrc.setChecked(False)
        self.type_group.addButton(self.rd_absp)
        self.type_group.addButton(self.rd_flrc)

        lb_fp = QLabel()
        lb_fp.setFont(self.font2)
        lb_fp.setText('')
        lb_fp.setFixedWidth(100)

        self.pb_bin = QPushButton('XANES Binning')
        self.pb_bin.setFont(self.font2)
        self.pb_bin.clicked.connect(self.bin_image)
        self.pb_bin.setEnabled(True)
        self.pb_bin.setFixedWidth(140)

        self.cb_bin = QComboBox()
        self.cb_bin.setFont(self.font2)
        self.cb_bin.addItem('2 x 2')
        self.cb_bin.addItem('4 x 4')
        self.cb_bin.setFixedWidth(70)

        self.tx_bin = QLineEdit()
        self.tx_bin.setText('X_eng')
        self.tx_bin.setFixedWidth(85)
        self.tx_bin.setFont(self.font2)

        hbox_bin = QHBoxLayout()
        hbox_bin.addWidget(self.pb_bin)
        hbox_bin.addWidget(self.cb_bin)
        hbox_bin.setAlignment(QtCore.Qt.AlignLeft)

        # gpbox_sys = self.gpbox_system_info()

        hbox1 = QHBoxLayout()
        hbox1.addWidget(lb_ld)
        hbox1.addWidget(self.rd_tif)
        hbox1.addWidget(self.rd_hdf)
        hbox1.addWidget(lb_hdf_xanes)
        hbox1.addWidget(self.tx_hdf_xanes)
        hbox1.addWidget(self.pb_ld)
        hbox1.addLayout(hbox_bin)
        hbox1.addWidget(lb_empty)
        hbox1.setAlignment(QtCore.Qt.AlignLeft)

        # load fitted file
        self.pb_ld_fitted = QPushButton('Load Fitted')
        self.pb_ld_fitted.setToolTip('File: .h5')
        self.pb_ld_fitted.setFont(self.font2)
        self.pb_ld_fitted.clicked.connect(self.load_fitted_file)
        self.pb_ld_fitted.setFixedWidth(140)


        # databroker 
        hbox2 = QHBoxLayout()
        hbox2.addWidget(lb_empty3)
        hbox2.addWidget(lb_empty2)
        hbox2.addWidget(self.pb_ld)
        hbox2.addWidget(self.pb_ld_eng)
        hbox2.addWidget(lb_empty)
        hbox2.setAlignment(QtCore.Qt.AlignLeft)

        hbox3 = QHBoxLayout()
        hbox3.addWidget(lb_ld_eng)
        hbox3.addWidget(self.rd_txt_eng)
        hbox3.addWidget(self.rd_hdf_eng)
        hbox3.addWidget(lb_hdf_eng)
        hbox3.addWidget(self.tx_hdf_eng)
        hbox3.addWidget(self.pb_ld_eng)
        hbox3.addWidget(self.pb_ld_fitted)
        hbox3.addWidget(lb_empty)
        hbox3.setAlignment(QtCore.Qt.AlignLeft)

        # XANES energy: no energy data
        hbox_eng = QHBoxLayout()
        hbox_eng.addWidget(lb_eng)
        hbox_eng.addWidget(self.lb_eng1)
        hbox_eng.setAlignment(QtCore.Qt.AlignLeft)

        hbox_manul_input = QHBoxLayout()
        hbox_manul_input.addWidget(self.lb_eng2)
        hbox_manul_input.addWidget(self.tx_eng)
        hbox_manul_input.addWidget(self.pb_eng)
        hbox_manul_input.addWidget(self.pb_fiji)
        hbox_manul_input.addWidget(self.pb_closeImg)
        hbox_manul_input.setAlignment(QtCore.Qt.AlignLeft)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addLayout(hbox3)
        vbox.addLayout(hbox_eng)
        vbox.addLayout(hbox_manul_input)
        vbox.setAlignment(QtCore.Qt.AlignLeft)

        hbox_tot = QHBoxLayout()
        hbox_tot.addLayout(vbox)
        # hbox_tot.addLayout(gpbox_sys)
        hbox_tot.addWidget(lb_empty)
        hbox_tot.setAlignment(QtCore.Qt.AlignLeft)

        gpbox.setLayout(hbox_tot)
        return gpbox

    def layout_xanes(self):
        lb_empty = QLabel()
        lb_empty2 = QLabel()
        lb_empty2.setFixedWidth(5)

        gpbox = QGroupBox('XANES fitting')
        gpbox.setFont(self.font1)

        tabs = QTabWidget()
        tab1 = QWidget()
        tab2 = QWidget()
        tab3 = QWidget()
        tab4 = QWidget()
        tab5 = QWidget()

        lay1 = QVBoxLayout()
        lay2 = QVBoxLayout()
        lay3 = QVBoxLayout()
        lay4 = QVBoxLayout()
        lay5 = QVBoxLayout()

        scroll1 = QScrollArea()
        scroll1.setWidgetResizable(True)
        scroll1.setFixedHeight(600)
        scroll1.setFixedWidth(400)

        scroll2 = QScrollArea()
        scroll2.setWidgetResizable(True)
        scroll2.setFixedHeight(600)
        scroll2.setFixedWidth(400)

        scroll3 = QScrollArea()
        scroll3.setWidgetResizable(True)
        scroll3.setFixedHeight(600)
        scroll3.setFixedWidth(400)

        scroll4 = QScrollArea()
        scroll4.setWidgetResizable(True)
        scroll4.setFixedHeight(600)
        scroll4.setFixedWidth(400)

        scroll5 = QScrollArea()
        scroll5.setWidgetResizable(True)
        scroll5.setFixedHeight(600)
        scroll5.setFixedWidth(400)

        xanes_prep_layout = self.layout_xanes_prep()
        xanes_roi_layout = self.layout_plot_spec
        xanes_roi_norm_layout = self.layout_roi_normalization()
        xanes_fit2d_layout = self.layout_fit2d()
        img_tools_layout = self.layout_img_tools()
        canvas_layout = self.layout_canvas()
        analysis_layout = self.layout_find_edge()

        vbox_lay1 = QVBoxLayout()
        vbox_lay2 = QVBoxLayout()
        vbox_lay3 = QVBoxLayout()
        vbox_lay4 = QVBoxLayout()
        vbox_lay5 = QVBoxLayout()

        vbox_lay1.addLayout(xanes_prep_layout)
        vbox_lay1.setAlignment(QtCore.Qt.AlignTop)
        vbox_lay1.addLayout(xanes_roi_layout)

        vbox_lay2.addLayout(xanes_roi_norm_layout)
        vbox_lay2.setAlignment(QtCore.Qt.AlignTop)

        vbox_lay3.addLayout(xanes_fit2d_layout)
        vbox_lay3.setAlignment(QtCore.Qt.AlignTop)

        vbox_lay4.addLayout(img_tools_layout)
        vbox_lay4.setAlignment(QtCore.Qt.AlignTop)

        vbox_lay5.addLayout(analysis_layout)
        vbox_lay5.setAlignment(QtCore.Qt.AlignTop)

        gp_box1 = QGroupBox()
        gp_box2 = QGroupBox()
        gp_box3 = QGroupBox()
        gp_box4 = QGroupBox()
        gp_box5 = QGroupBox()

        gp_box1.setLayout(vbox_lay1)
        gp_box2.setLayout(vbox_lay2)
        gp_box3.setLayout(vbox_lay3)
        gp_box4.setLayout(vbox_lay4)
        gp_box5.setLayout(vbox_lay5)

        scroll1.setWidget(gp_box1)
        scroll2.setWidget(gp_box2)
        scroll3.setWidget(gp_box3)
        scroll4.setWidget(gp_box4)
        scroll5.setWidget(gp_box5)

        lay1.addWidget(scroll1)
        lay2.addWidget(scroll2)
        lay3.addWidget(scroll3)
        lay4.addWidget(scroll4)
        lay5.addWidget(scroll5)

        tab1.setLayout(lay1)
        tab2.setLayout(lay2)
        tab3.setLayout(lay3)
        tab4.setLayout(lay4)
        tab5.setLayout(lay5)

        tabs.addTab(tab1, 'Prep.')
        tabs.addTab(tab2, 'Norm.')
        tabs.addTab(tab3, 'Fit Spec.')
        tabs.addTab(tab4, 'Img. Tools')
        tabs.addTab(tab5, 'Others')

        hbox = QHBoxLayout()
        hbox.addWidget(tabs)
        hbox.addLayout(canvas_layout)
        hbox.setAlignment(QtCore.Qt.AlignLeft)
        gpbox.setLayout(hbox)
        return gpbox

    @property
    def layout_plot_spec(self):
        lb_empty = QLabel()
        lb_roi = QLabel()
        lb_roi.setFont(self.font1)
        lb_roi.setText('ROI for Spec.')
        lb_roi.setFixedWidth(100)

        lb_info = QLabel()
        lb_info.setFont(self.font2)
        lb_info.setStyleSheet('color: rgb(200, 50, 50);')
        lb_info.setText('Spectrum calc. based on current image stack')

        lb_roi_x1 = QLabel()
        lb_roi_x1.setText('Top-left  x:')
        lb_roi_x1.setFont(self.font2)
        lb_roi_x1.setFixedWidth(80)

        lb_roi_y1 = QLabel()
        lb_roi_y1.setText('y:')
        lb_roi_y1.setFont(self.font2)
        lb_roi_y1.setFixedWidth(20)

        lb_roi_x2 = QLabel()
        lb_roi_x2.setText('Bot-right x:')
        lb_roi_x2.setFont(self.font2)
        lb_roi_x2.setFixedWidth(80)

        lb_roi_y2 = QLabel()
        lb_roi_y2.setText('y:')
        lb_roi_y2.setFont(self.font2)
        lb_roi_y2.setFixedWidth(20)

        self.tx_roi_x1 = QLineEdit()
        self.tx_roi_x1.setText('0')
        self.tx_roi_x1.setFont(self.font2)
        self.tx_roi_x1.setFixedWidth(50)

        self.tx_roi_y1 = QLineEdit()
        self.tx_roi_y1.setText('0')
        self.tx_roi_y1.setFont(self.font2)
        self.tx_roi_y1.setFixedWidth(50)

        self.tx_roi_x2 = QLineEdit()
        self.tx_roi_x2.setText('1')
        self.tx_roi_x2.setFont(self.font2)
        self.tx_roi_x2.setFixedWidth(50)

        self.tx_roi_y2 = QLineEdit()
        self.tx_roi_y2.setText('1')
        self.tx_roi_y2.setFont(self.font2)
        self.tx_roi_y2.setFixedWidth(50)

        self.pb_roi_draw = QPushButton('Draw ROI')
        self.pb_roi_draw.setFont(self.font2)
        self.pb_roi_draw.clicked.connect(self.draw_roi)
        self.pb_roi_draw.setFixedWidth(105)

        self.pb_roi_plot = QPushButton('Plot Spec.')
        self.pb_roi_plot.setFont(self.font2)
        self.pb_roi_plot.clicked.connect(self.plot_spectrum)
        self.pb_roi_plot.setFixedWidth(105)

        self.pb_roi_hide = QPushButton('Hide ROI')
        self.pb_roi_hide.setFont(self.font2)
        self.pb_roi_hide.clicked.connect(self.hide_roi)
        self.pb_roi_hide.setFixedWidth(105)

        self.pb_roi_show = QPushButton('Show ROI')
        self.pb_roi_show.setFont(self.font2)
        self.pb_roi_show.clicked.connect(self.show_roi)
        self.pb_roi_show.setFixedWidth(105)

        self.pb_roi_reset = QPushButton('Reset ROI')
        self.pb_roi_reset.setFont(self.font2)
        self.pb_roi_reset.clicked.connect(self.reset_roi)
        self.pb_roi_reset.setFixedWidth(105)

        self.pb_roi_export = QPushButton('Export Spec.')
        self.pb_roi_export.setFont(self.font2)
        self.pb_roi_export.clicked.connect(self.export_spectrum)
        self.pb_roi_export.setFixedWidth(105)

        self.pb_eval_glitch = QPushButton('Eval. Glitch')
        self.pb_eval_glitch.setFont(self.font2)
        self.pb_eval_glitch.clicked.connect(self.evaluate_glitch)
        self.pb_eval_glitch.setFixedWidth(105)

        self.pb_rm_glitch = QPushButton('Remov Glitch')
        self.pb_rm_glitch.setFont(self.font2)
        self.pb_rm_glitch.clicked.connect(self.remove_glitch)
        self.pb_rm_glitch.setFixedWidth(105)

        lb_glitch_thresh = QLabel()
        lb_glitch_thresh.setFont(self.font2)
        lb_glitch_thresh.setText('  Glitch threshold: ')
        lb_glitch_thresh.setFixedWidth(155)

        self.tx_glitch_thresh = QLineEdit(self)
        self.tx_glitch_thresh.setFont(self.font2)
        self.tx_glitch_thresh.setText('<0.8')
        self.tx_glitch_thresh.setFixedWidth(50)

        lb_file_index = QLabel()
        lb_file_index.setFont(self.font2)
        lb_file_index.setText('  File index for export:')
        lb_file_index.setFixedWidth(155)

        self.tx_file_index = QLineEdit()
        self.tx_file_index.setFixedWidth(50)
        self.tx_file_index.setFont(self.font2)
        self.tx_file_index.setText(str(self.roi_file_id))

        self.lst_roi = QListWidget()
        self.lst_roi.setFont(self.font2)
        self.lst_roi.setSelectionMode(QAbstractItemView.MultiSelection)
        self.lst_roi.setFixedWidth(90)
        self.lst_roi.setFixedHeight(140)

        lb_lst_roi = QLabel()
        lb_lst_roi.setFont(self.font2)
        lb_lst_roi.setText('ROI list:')
        lb_lst_roi.setFixedWidth(80)

        hbox_roi_button1 = QHBoxLayout()
        hbox_roi_button1.addWidget(self.pb_roi_draw)
        hbox_roi_button1.addWidget(self.pb_roi_reset)
        hbox_roi_button1.setAlignment(QtCore.Qt.AlignLeft)

        hbox_roi_button2 = QHBoxLayout()
        hbox_roi_button2.addWidget(self.pb_roi_show)
        hbox_roi_button2.addWidget(self.pb_roi_hide)
        hbox_roi_button2.setAlignment(QtCore.Qt.AlignLeft)

        hbox_roi_button3 = QHBoxLayout()
        hbox_roi_button3.addWidget(self.pb_roi_plot)
        hbox_roi_button3.addWidget(self.pb_roi_export)
        hbox_roi_button3.setAlignment(QtCore.Qt.AlignLeft)

        hbox_roi_button4 = QHBoxLayout()
        hbox_roi_button4.addWidget(self.pb_eval_glitch)
        hbox_roi_button4.addWidget(self.pb_rm_glitch)
        hbox_roi_button4.setAlignment(QtCore.Qt.AlignLeft)

        hbox_roi_button5 = QHBoxLayout()
        hbox_roi_button5.addWidget(lb_glitch_thresh)
        hbox_roi_button5.addWidget(self.tx_glitch_thresh)
        hbox_roi_button5.setAlignment(QtCore.Qt.AlignLeft)

        hbox_roi_button6 = QHBoxLayout()
        hbox_roi_button6.addWidget(lb_file_index)
        hbox_roi_button6.addWidget(self.tx_file_index)
        hbox_roi_button6.setAlignment(QtCore.Qt.AlignLeft)

        vbox_roi = QVBoxLayout()
        vbox_roi.setContentsMargins(0, 0, 0, 0)
        vbox_roi.addLayout(hbox_roi_button1)
        vbox_roi.addLayout(hbox_roi_button2)
        vbox_roi.addLayout(hbox_roi_button3)
        vbox_roi.addLayout(hbox_roi_button4)
        vbox_roi.addLayout(hbox_roi_button5)
        vbox_roi.addLayout(hbox_roi_button6)
        vbox_roi.setAlignment(QtCore.Qt.AlignLeft)

        vbox_lst = QVBoxLayout()
        vbox_lst.addWidget(lb_lst_roi, 0, QtCore.Qt.AlignTop)
        vbox_lst.addWidget(self.lst_roi, 0, QtCore.Qt.AlignTop)
        vbox_lst.addWidget(lb_empty)
        vbox_lst.setAlignment(QtCore.Qt.AlignLeft)

        box_roi = QHBoxLayout()
        box_roi.addLayout(vbox_roi)
        box_roi.addLayout(vbox_lst)
        box_roi.addWidget(lb_empty, 0, QtCore.Qt.AlignLeft)
        box_roi.setAlignment(QtCore.Qt.AlignLeft)

        box_roi_tot = QVBoxLayout()
        box_roi_tot.addWidget(lb_roi)
        box_roi_tot.addWidget(lb_info)
        box_roi_tot.addLayout(box_roi)
        box_roi_tot.addWidget(lb_empty)
        box_roi_tot.setAlignment(QtCore.Qt.AlignLeft)
        return box_roi_tot

    def layout_find_edge(self):
        lb_empty = QLabel()
        lb_empty.setFixedWidth(100)
        lb_empty1 = QLabel()
        lb_empty1.setFixedWidth(20)

        lb_find_edge = QLabel()
        lb_find_edge.setFont(self.font1)
        lb_find_edge.setText('Find absorption edge')
        lb_find_edge.setFixedWidth(200)

        lb_edge_range = QLabel()
        lb_edge_range.setFont(self.font2)
        lb_edge_range.setText('Energy range:')
        lb_edge_range.setFixedWidth(100)

        lb_edge_range_s = QLabel()
        lb_edge_range_s.setFont(self.font2)
        lb_edge_range_s.setText('start:')
        lb_edge_range_s.setFixedWidth(50)

        lb_edge_range_e = QLabel()
        lb_edge_range_e.setFont(self.font2)
        lb_edge_range_e.setText('end:')
        lb_edge_range_e.setFixedWidth(50)

        self.tx_edge_s = QLineEdit()
        self.tx_edge_s.setFont(self.font2)
        self.tx_edge_s.setFixedWidth(60)
        self.tx_edge_s.setValidator(QDoubleValidator())

        self.tx_edge_e = QLineEdit()
        self.tx_edge_e.setFont(self.font2)
        self.tx_edge_e.setFixedWidth(60)
        self.tx_edge_e.setValidator(QDoubleValidator())

        hbox_edge_range = QHBoxLayout()
        hbox_edge_range.addWidget(lb_edge_range)
        hbox_edge_range.addWidget(lb_edge_range_s)
        hbox_edge_range.addWidget(self.tx_edge_s)
        hbox_edge_range.addWidget(lb_empty1)
        hbox_edge_range.addWidget(lb_edge_range_e)
        hbox_edge_range.addWidget(self.tx_edge_e)
        hbox_edge_range.setAlignment(QtCore.Qt.AlignLeft)

        #
        lb_edge_est = QLabel()
        lb_edge_est.setText('Pre-edge:')
        lb_edge_est.setFont(self.font2)
        lb_edge_est.setFixedWidth(100)

        lb_edge_est_pos = QLabel()
        lb_edge_est_pos.setText('pos.:')
        lb_edge_est_pos.setFont(self.font2)
        lb_edge_est_pos.setFixedWidth(50)

        self.tx_edge_pos = QLineEdit()
        self.tx_edge_pos.setFont(self.font2)
        self.tx_edge_pos.setText('')
        self.tx_edge_pos.setFixedWidth(60)
        self.tx_edge_pos.setValidator(QDoubleValidator())

        lb_pre_edge_wt = QLabel()
        lb_pre_edge_wt.setText('weight:')
        lb_pre_edge_wt.setFont(self.font2)
        lb_pre_edge_wt.setFixedWidth(50)

        self.tx_pre_edge_wt = QLineEdit()
        self.tx_pre_edge_wt.setText('1.0')
        self.tx_pre_edge_wt.setFont(self.font2)
        self.tx_pre_edge_wt.setValidator(QDoubleValidator())
        self.tx_pre_edge_wt.setFixedWidth(60)

        lb_empty2 = QLabel()
        lb_empty2.setFixedWidth(20)

        hbox_edge_pos = QHBoxLayout()
        hbox_edge_pos.addWidget(lb_edge_est)
        hbox_edge_pos.addWidget(lb_edge_est_pos)
        hbox_edge_pos.addWidget(self.tx_edge_pos)
        hbox_edge_pos.addWidget(lb_empty2)
        hbox_edge_pos.addWidget(lb_pre_edge_wt)
        hbox_edge_pos.addWidget(self.tx_pre_edge_wt)
        hbox_edge_pos.setAlignment(QtCore.Qt.AlignLeft)

        # fit parameter

        lb_empty3 = QLabel()
        lb_empty3.setFixedWidth(100)

        lb_edge_param = QLabel()
        lb_edge_param.setText('Fit param:')
        lb_edge_param.setFont(self.font2)
        lb_edge_param.setFixedWidth(100)

        lb_edge_smooth = QLabel()
        lb_edge_smooth.setText('smooth:')
        lb_edge_smooth.setFixedWidth(50)
        lb_edge_smooth.setFont(self.font2)

        self.tx_edge_smooth = QLineEdit()
        self.tx_edge_smooth.setText('0.002')
        self.tx_edge_smooth.setFont(self.font2)
        self.tx_edge_smooth.setValidator(QDoubleValidator())
        self.tx_edge_smooth.setFixedWidth(60)

        lb_edge_order = QLabel()
        lb_edge_order.setText('order:')
        lb_edge_order.setFixedWidth(50)
        lb_edge_order.setFont(self.font2)

        self.tx_edge_order = QLineEdit()
        self.tx_edge_order.setText('2')
        self.tx_edge_order.setFont(self.font2)
        self.tx_edge_order.setValidator(QIntValidator())
        self.tx_edge_order.setFixedWidth(60)

        hbox_edge_param = QHBoxLayout()
        hbox_edge_param.addWidget(lb_edge_param)
        hbox_edge_param.addWidget(lb_edge_smooth)
        hbox_edge_param.addWidget(self.tx_edge_smooth)
        hbox_edge_param.addWidget(lb_empty2)
        hbox_edge_param.addWidget(lb_edge_order)
        hbox_edge_param.addWidget(self.tx_edge_order)
        hbox_edge_param.setAlignment(QtCore.Qt.AlignLeft)

        #
        self.chkbox_edge = QCheckBox('Fitting edge')
        self.chkbox_edge.setFont(self.font2)
        self.chkbox_edge.setFixedWidth(105)
        self.chkbox_edge.setChecked(True)

        self.chkbox_peak = QCheckBox('Fitting peak')
        self.chkbox_peak.setFont(self.font2)
        self.chkbox_peak.setFixedWidth(105)
        self.chkbox_peak.setChecked(True)

        self.peak_maxmin_group = QButtonGroup()
        self.peak_maxmin_group.setExclusive(True)
        self.rd_peak_max = QRadioButton('max')
        self.rd_peak_max.setFixedWidth(60)
        self.rd_peak_max.setChecked(True)

        self.rd_peak_min = QRadioButton('min')
        self.rd_peak_min.setFixedWidth(60)
        self.rd_peak_min.setChecked(False)
        self.peak_maxmin_group.addButton(self.rd_peak_max)
        self.peak_maxmin_group.addButton(self.rd_peak_min)

        hbox_fit_peak = QHBoxLayout()
        hbox_fit_peak.addWidget(self.chkbox_peak)
        hbox_fit_peak.addWidget(self.rd_peak_max)
        hbox_fit_peak.addWidget(self.rd_peak_min)
        hbox_fit_peak.setAlignment(QtCore.Qt.AlignLeft)

        vbox_fit_edge_peak = QVBoxLayout()
        vbox_fit_edge_peak.addWidget(self.chkbox_edge)
        vbox_fit_edge_peak.addLayout(hbox_fit_peak)
        vbox_fit_edge_peak.setAlignment(QtCore.Qt.AlignTop)

        # scale image
        lb_empty3 = QLabel()
        lb_empty3.setFixedWidth(80)

        lb_scale_img = QLabel()
        lb_scale_img.setText('scale  x')
        lb_scale_img.setFixedWidth(50)
        lb_scale_img.setFont(self.font2)

        self.tx_scale_img = QLineEdit()
        self.tx_scale_img.setText('1.0')
        self.tx_scale_img.setFont(self.font2)
        self.tx_scale_img.setValidator(QDoubleValidator())
        self.tx_scale_img.setFixedWidth(60)

        self.pb_scale_img = QPushButton()
        self.pb_scale_img.setText('Scale Img.')
        self.pb_scale_img.setFont(self.font2)
        self.pb_scale_img.setFixedWidth(100)
        self.pb_scale_img.clicked.connect(self.scale_image)

        lb_overlay_roi = QLabel()
        lb_overlay_roi.setText('ROI #')
        lb_overlay_roi.setFixedWidth(50)
        lb_overlay_roi.setFont(self.font2)

        self.tx_overlay_roi = QLineEdit()
        self.tx_overlay_roi.setText('-1')
        self.tx_overlay_roi.setFont(self.font2)
        self.tx_overlay_roi.setValidator(QIntValidator())
        self.tx_overlay_roi.setFixedWidth(60)

        hbox_scale_img = QHBoxLayout()
        hbox_scale_img.addWidget(self.pb_scale_img)
        hbox_scale_img.addWidget(lb_scale_img)
        hbox_scale_img.addWidget(self.tx_scale_img)
        hbox_scale_img.addWidget(lb_empty2)
        hbox_scale_img.addWidget(lb_overlay_roi)
        hbox_scale_img.addWidget(self.tx_overlay_roi)
        hbox_scale_img.setAlignment(QtCore.Qt.AlignLeft)

        #

        lb_empty2 = QLabel()
        lb_empty2.setFixedWidth(5)

        self.pb_load_edge_curve = QPushButton()
        self.pb_load_edge_curve.setText('Load curve')
        self.pb_load_edge_curve.setFont(self.font2)
        self.pb_load_edge_curve.setFixedWidth(100)
        self.pb_load_edge_curve.clicked.connect(self.load_external_spec)

        self.pb_fit_edge_curve = QPushButton()
        self.pb_fit_edge_curve.setText('Fit curve')
        self.pb_fit_edge_curve.setFont(self.font2)
        self.pb_fit_edge_curve.setFixedWidth(100)
        self.pb_fit_edge_curve.clicked.connect(self.fit_edge_curve)

        hbox_fit_edge_curve = QHBoxLayout()
        hbox_fit_edge_curve.addWidget(self.pb_load_edge_curve)
        hbox_fit_edge_curve.addWidget(self.pb_fit_edge_curve)
        hbox_fit_edge_curve.setAlignment(QtCore.Qt.AlignLeft)

        #
        lb_empty2 = QLabel()
        lb_empty2.setFixedWidth(5)

        self.pb_find_edge = QPushButton()
        self.pb_find_edge.setText('Fit ROI')
        self.pb_find_edge.setFont(self.font2)
        self.pb_find_edge.setFixedWidth(100)
        self.pb_find_edge.clicked.connect(self.find_edge_peak_single)

        self.pb_find_edge_img = QPushButton()
        self.pb_find_edge_img.setText('Fit image')
        self.pb_find_edge_img.setFont(self.font2)
        self.pb_find_edge_img.setFixedWidth(100)
        self.pb_find_edge_img.clicked.connect(self.find_edge_peak_image)

        self.pb_plot_edge_roi = QPushButton()
        self.pb_plot_edge_roi.setText('Plot fit ROI')
        self.pb_plot_edge_roi.setFont(self.font2)
        self.pb_plot_edge_roi.setFixedWidth(100)
        self.pb_plot_edge_roi.clicked.connect(self.plot_fit_edge_peak_roi)

        hbox_find_edge = QHBoxLayout()
        hbox_find_edge.addWidget(self.pb_find_edge)
        hbox_find_edge.addWidget(self.pb_find_edge_img)
        hbox_find_edge.setAlignment(QtCore.Qt.AlignLeft)

        # convert peak position to percentage
        lb_empty2 = QLabel()
        lb_empty2.setFixedWidth(15)

        self.pb_cvt_percentage = QPushButton()
        self.pb_cvt_percentage.setText('Cvt peak %')
        self.pb_cvt_percentage.setFont(self.font2)
        self.pb_cvt_percentage.setFixedWidth(100)
        self.pb_cvt_percentage.clicked.connect(self.convert_percentage_image)

        lb_cvt_min = QLabel()
        lb_cvt_min.setText('pk_min')
        lb_cvt_min.setFixedWidth(50)
        lb_cvt_min.setFont(self.font2)

        self.tx_cvt_min = QLineEdit()
        self.tx_cvt_min.setText('')
        self.tx_cvt_min.setFont(self.font2)
        self.tx_cvt_min.setValidator(QDoubleValidator())
        self.tx_cvt_min.setFixedWidth(60)

        lb_cvt_max = QLabel()
        lb_cvt_max.setText('pk_max')
        lb_cvt_max.setFixedWidth(55)
        lb_cvt_max.setFont(self.font2)

        self.tx_cvt_max = QLineEdit()
        self.tx_cvt_max.setText('')
        self.tx_cvt_max.setFont(self.font2)
        self.tx_cvt_max.setValidator(QDoubleValidator())
        self.tx_cvt_max.setFixedWidth(60)

        hbox_cvt_percentage = QHBoxLayout()
        hbox_cvt_percentage.addWidget(self.pb_cvt_percentage)
        hbox_cvt_percentage.addWidget(lb_cvt_min)
        hbox_cvt_percentage.addWidget(self.tx_cvt_min)
        hbox_cvt_percentage.addWidget(lb_empty2)
        hbox_cvt_percentage.addWidget(lb_cvt_max)
        hbox_cvt_percentage.addWidget(self.tx_cvt_max)
        hbox_cvt_percentage.setAlignment(QtCore.Qt.AlignLeft)

        # assemble
        vbox_find_edge = QVBoxLayout()
        vbox_find_edge.addWidget(lb_find_edge)
        vbox_find_edge.addLayout(vbox_fit_edge_peak)
        vbox_find_edge.addLayout(hbox_edge_range)
        vbox_find_edge.addLayout(hbox_edge_pos)
        vbox_find_edge.addLayout(hbox_edge_param)
        vbox_find_edge.addLayout(hbox_scale_img)
        vbox_find_edge.addLayout(hbox_find_edge)
        vbox_find_edge.addLayout(hbox_cvt_percentage)
        vbox_find_edge.addWidget(lb_empty)
        vbox_find_edge.addLayout(hbox_fit_edge_curve)
        vbox_find_edge.setAlignment(QtCore.Qt.AlignTop)

        return vbox_find_edge

    def layout_roi_normalization(self):
        lb_empty = QLabel()
        lb_empty2 = QLabel()
        lb_empty2.setFixedWidth(120)

        lb_fit_edge = QLabel()
        lb_fit_edge.setFont(self.font1)
        lb_fit_edge.setText('ROI normalization')
        lb_fit_edge.setFixedWidth(150)

        lb_fit_pre_s = QLabel()
        lb_fit_pre_s.setText('Pre -edge start:')
        lb_fit_pre_s.setFont(self.font2)
        lb_fit_pre_s.setFixedWidth(120)

        lb_fit_pre_e = QLabel()
        lb_fit_pre_e.setText(' end: ')
        lb_fit_pre_e.setFont(self.font2)
        lb_fit_pre_e.setFixedWidth(50)

        lb_fit_post_s = QLabel()
        lb_fit_post_s.setText('Post-edge start:')
        lb_fit_post_s.setFont(self.font2)
        lb_fit_post_s.setFixedWidth(120)

        lb_fit_post_e = QLabel()
        lb_fit_post_e.setText(' end: ')
        lb_fit_post_e.setFont(self.font2)
        lb_fit_post_e.setFixedWidth(50)

        self.tx_fit_pre_s = QLineEdit()
        self.tx_fit_pre_s.setFont(self.font2)
        self.tx_fit_pre_s.setFixedWidth(60)
        self.tx_fit_pre_s.setValidator(QDoubleValidator())

        self.tx_fit_pre_e = QLineEdit()
        self.tx_fit_pre_e.setFont(self.font2)
        self.tx_fit_pre_e.setFixedWidth(60)
        self.tx_fit_pre_e.setValidator(QDoubleValidator())

        self.tx_fit_post_s = QLineEdit()
        self.tx_fit_post_s.setFont(self.font2)
        self.tx_fit_post_s.setFixedWidth(60)
        self.tx_fit_post_s.setValidator(QDoubleValidator())

        self.tx_fit_post_e = QLineEdit()
        self.tx_fit_post_e.setFont(self.font2)
        self.tx_fit_post_e.setFixedWidth(60)
        self.tx_fit_post_e.setValidator(QDoubleValidator())

        lb_fit_roi = QLabel()
        lb_fit_roi.setFont(self.font2)
        lb_fit_roi.setText('Norm Spec(ROI)')
        lb_fit_roi.setFixedWidth(120)

        self.pb_fit_roi = QPushButton('Norm Spec')
        self.pb_fit_roi.setFont(self.font2)
        self.pb_fit_roi.clicked.connect(self.fit_edge)
        self.pb_fit_roi.setFixedWidth(90)

        self.pb_save_fit_roi = QPushButton('Save Spec')
        self.pb_save_fit_roi.setFont(self.font2)
        self.pb_save_fit_roi.clicked.connect(self.save_normed_roi)
        self.pb_save_fit_roi.setFixedWidth(90)

        lb_fit_img = QLabel()
        lb_fit_img.setFont(self.font2)
        lb_fit_img.setText('Norm Image')
        lb_fit_img.setFixedWidth(120)

        self.pb_fit_img = QPushButton('Norm Image')
        self.pb_fit_img.setFont(self.font2)
        self.pb_fit_img.clicked.connect(self.fit_edge_img)
        self.pb_fit_img.setFixedWidth(90)

        self.chkbox_norm_pre_edge_only = QCheckBox('pre-edge only')
        self.chkbox_norm_pre_edge_only.setFixedWidth(190)
        self.chkbox_norm_pre_edge_only.setFont(self.font2)
        self.chkbox_norm_pre_edge_only.setChecked(False)

        self.norm_group = QButtonGroup()
        self.norm_group.setExclusive(True)
        self.rd_norm1 = QRadioButton('1')
        self.rd_norm1.setFixedWidth(45)

        # self.rd_norm1.toggled.connect(self.select_file)

        self.rd_norm2 = QRadioButton('2')
        self.rd_norm2.setFixedWidth(45)
        # self.rd_norm2.toggled.connect(self.select_file)

        self.norm_group.addButton(self.rd_norm1)
        self.norm_group.addButton(self.rd_norm2)
        self.norm_group = QButtonGroup()
        self.norm_group.setExclusive(True)
        self.rd_norm1.setChecked(True)

        lb_reg_img = QLabel()
        lb_reg_img.setFont(self.font2)
        lb_reg_img.setText('max:')
        lb_reg_img.setFixedWidth(30)

        self.tx_reg_max = QLineEdit()
        self.tx_reg_max.setFont(self.font2)
        self.tx_reg_max.setText('1.65')
        self.tx_reg_max.setValidator(QDoubleValidator())
        self.tx_reg_max.setFixedWidth(60)

        lb_reg_width = QLabel()
        lb_reg_width.setText(' Width:')
        lb_reg_width.setFont(self.font2)
        lb_reg_width.setFixedWidth(50)

        self.tx_reg_width = QLineEdit()
        self.tx_reg_width.setFont(self.font2)
        self.tx_reg_width.setValidator(QDoubleValidator())
        self.tx_reg_width.setText('0.05')
        self.tx_reg_width.setFixedWidth(60)

        self.pb_reg_img = QPushButton('Regulation')
        self.pb_reg_img.setFont(self.font2)
        self.pb_reg_img.clicked.connect(self.regular_edge_img)
        self.pb_reg_img.setFixedWidth(90)

        # load and norm existing spectrum
        lb_alt = QLabel()
        lb_alt.setFont(self.font1)
        lb_alt.setText('External spectrum')
        lb_alt.setFixedWidth(150)

        self.pb_load_exist_spec = QPushButton('Load external Spec.')
        self.pb_load_exist_spec.setFont(self.font2)
        self.pb_load_exist_spec.clicked.connect(self.load_external_spec)
        self.pb_load_exist_spec.setFixedWidth(200)

        self.pb_norm_exist_spec = QPushButton('Norm external Spec.')
        self.pb_norm_exist_spec.setFont(self.font2)
        self.pb_norm_exist_spec.clicked.connect(self.norm_external_spec)
        self.pb_norm_exist_spec.setFixedWidth(200)

        self.pb_save_exist_spec = QPushButton('Save external Spec.')
        self.pb_save_exist_spec.setFont(self.font2)
        self.pb_save_exist_spec.clicked.connect(self.save_external_spec)
        self.pb_save_exist_spec.setFixedWidth(200)

        vbox_exist_spec = QVBoxLayout()
        vbox_exist_spec.addWidget(lb_alt)
        vbox_exist_spec.addWidget(self.pb_load_exist_spec)
        vbox_exist_spec.addWidget(self.pb_norm_exist_spec)
        vbox_exist_spec.addWidget(self.pb_save_exist_spec)
        vbox_exist_spec.setAlignment(QtCore.Qt.AlignTop)


        hbox_fit_pre = QHBoxLayout()
        hbox_fit_pre.addWidget(lb_fit_pre_s)
        hbox_fit_pre.addWidget(self.tx_fit_pre_s)
        hbox_fit_pre.addWidget(lb_fit_pre_e)
        hbox_fit_pre.addWidget(self.tx_fit_pre_e)
        hbox_fit_pre.setAlignment(QtCore.Qt.AlignLeft)

        hbox_fit_post = QHBoxLayout()
        hbox_fit_post.addWidget(lb_fit_post_s)
        hbox_fit_post.addWidget(self.tx_fit_post_s)
        hbox_fit_post.addWidget(lb_fit_post_e)
        hbox_fit_post.addWidget(self.tx_fit_post_e)
        hbox_fit_post.setAlignment(QtCore.Qt.AlignLeft)

        hbox_fit_pb = QHBoxLayout()
        hbox_fit_pb.addWidget(lb_fit_roi)
        hbox_fit_pb.addWidget(self.pb_fit_roi)
        hbox_fit_pb.addWidget(self.pb_save_fit_roi)
        hbox_fit_pb.setAlignment(QtCore.Qt.AlignLeft)

        hbox_fit_pb_img = QHBoxLayout()
        # hbox_fit_pb_img.addWidget(lb_fit_img)
        hbox_fit_pb_img.addWidget(self.pb_fit_img)
        hbox_fit_pb_img.addWidget(self.rd_norm1)
        hbox_fit_pb_img.addWidget(self.rd_norm2)
        hbox_fit_pb_img.addWidget(self.chkbox_norm_pre_edge_only)
        hbox_fit_pb_img.setAlignment(QtCore.Qt.AlignLeft)

        hbox_reg_pb_img = QHBoxLayout()
        hbox_reg_pb_img.addWidget(self.pb_reg_img)
        hbox_reg_pb_img.addWidget(lb_reg_img)
        hbox_reg_pb_img.addWidget(self.tx_reg_max)
        hbox_reg_pb_img.addWidget(lb_reg_width)
        hbox_reg_pb_img.addWidget(self.tx_reg_width)
        hbox_reg_pb_img.setAlignment(QtCore.Qt.AlignLeft)

        hbox_reg_pb_img2 = QHBoxLayout()
        hbox_reg_pb_img2.addWidget(lb_empty2)
        hbox_reg_pb_img2.addWidget(self.pb_reg_img)
        hbox_reg_pb_img2.setAlignment(QtCore.Qt.AlignLeft)

        vbox_fit = QVBoxLayout()
        vbox_fit.addWidget(lb_fit_edge)
        vbox_fit.addLayout(hbox_fit_pre)
        vbox_fit.addLayout(hbox_fit_post)
        vbox_fit.addLayout(hbox_fit_pb)
        vbox_fit.addLayout(hbox_fit_pb_img)
        vbox_fit.addLayout(hbox_reg_pb_img)
        # vbox_fit.addLayout(hbox_reg_pb_img2)
        vbox_fit.addWidget(lb_empty)
        vbox_fit.addLayout(vbox_exist_spec)
        vbox_fit.setAlignment(QtCore.Qt.AlignLeft)
        return vbox_fit

    def layout_fit2d(self):
        lb_empty = QLabel()
        lb_fit2d = QLabel()
        lb_fit2d.setFont(self.font1)
        lb_fit2d.setText('Fit 2D XANES')
        lb_fit2d.setFixedWidth(150)

        self.lb_ref_info = QLabel()
        self.lb_ref_info.setFont(self.font2)
        self.lb_ref_info.setStyleSheet('color: rgb(200, 50, 50);')
        self.lb_ref_info.setText('Reference spectrum: ')
        self.lb_ref_info.setFixedWidth(300)
        
        self.pb_ld_ref = QPushButton('Load Ref.')
        self.pb_ld_ref.setFont(self.font2)
        self.pb_ld_ref.clicked.connect(self.load_xanes_ref)
        self.pb_ld_ref.setEnabled(True)
        self.pb_ld_ref.setFixedWidth(105)

        self.pb_plt_ref = QPushButton('Plot Ref.')
        self.pb_plt_ref.setFont(self.font2)
        self.pb_plt_ref.clicked.connect(self.plot_xanes_ref)
        self.pb_plt_ref.setEnabled(True)
        self.pb_plt_ref.setFixedWidth(105)

        lb_elem = QLabel()
        lb_elem.setFont(self.font2)
        lb_elem.setText(' Elem.: ')
        lb_elem.setFixedWidth(40)

        self.tx_elem = QLineEdit(self)
        self.tx_elem.setFont(self.font2)
        self.tx_elem.setFixedWidth(60)

        hbox_ref = QHBoxLayout()
        hbox_ref.addWidget(self.pb_ld_ref)
        hbox_ref.addWidget(lb_elem)
        hbox_ref.addWidget(self.tx_elem)
        hbox_ref.addWidget(self.pb_plt_ref)
        hbox_ref.setAlignment(QtCore.Qt.AlignTop)

        #######################
        self.pb_fit2d = QPushButton('Fit 2D')
        self.pb_fit2d.setFont(self.font2)
        self.pb_fit2d.clicked.connect(self.fit_2d_xanes)
        self.pb_fit2d.setEnabled(True)
        self.pb_fit2d.setFixedWidth(105)

        self.pb_reset_ref = QPushButton('Reset Ref.')
        self.pb_reset_ref.setFont(self.font2)
        self.pb_reset_ref.clicked.connect(self.reset_xanes_ref)
        self.pb_reset_ref.setEnabled(True)
        self.pb_reset_ref.setFixedWidth(105)

        self.pb_reset_fit = QPushButton('Reset All')
        self.pb_reset_fit.setFont(self.font2)
        self.pb_reset_fit.clicked.connect(self.reset_xanes_fit)
        self.pb_reset_fit.setEnabled(True)
        self.pb_reset_fit.setFixedWidth(105)

        hbox_reset_ref = QHBoxLayout()
        # hbox_fit2d.addWidget(self.pb_fit2d)
        hbox_reset_ref.addWidget(self.pb_reset_ref)
        hbox_reset_ref.addWidget(self.pb_reset_fit)
        hbox_reset_ref.setAlignment(QtCore.Qt.AlignTop)

        #############################################
        lb_fit2d_s = QLabel()
        lb_fit2d_s.setText('Fit energy range  start:')
        lb_fit2d_s.setFont(self.font2)
        lb_fit2d_s.setFixedWidth(150)

        lb_fit2d_e = QLabel()
        lb_fit2d_e.setText(' end:')
        lb_fit2d_e.setFont(self.font2)
        lb_fit2d_e.setFixedWidth(40)

        self.tx_fit2d_s = QLineEdit()
        self.tx_fit2d_s.setFont(self.font2)
        self.tx_fit2d_s.setFixedWidth(60)

        self.tx_fit2d_e = QLineEdit()
        self.tx_fit2d_e.setFont(self.font2)
        self.tx_fit2d_e.setFixedWidth(60)

        hbox_fit2d_range = QHBoxLayout()
        hbox_fit2d_range.addWidget(lb_fit2d_s)
        hbox_fit2d_range.addWidget(self.tx_fit2d_s)
        hbox_fit2d_range.addWidget(lb_fit2d_e)
        hbox_fit2d_range.addWidget(self.tx_fit2d_e)
        hbox_fit2d_range.setAlignment(QtCore.Qt.AlignLeft)

        ######################################
        
        self.pb_fit2d_iter = QPushButton('Fit 2D (iter)')
        self.pb_fit2d_iter.setFont(self.font2)
        self.pb_fit2d_iter.clicked.connect(self.fit_2d_xanes_iter)
        self.pb_fit2d_iter.setEnabled(True)
        self.pb_fit2d_iter.setFixedWidth(105)

        # lb_iter_rate = QLabel()
        # lb_iter_rate.setFont(self.font2)
        # lb_iter_rate.setText(' Rate:')
        # lb_iter_rate.setFixedWidth(50)

        lb_iter_num = QLabel()
        lb_iter_num.setFont(self.font2)
        lb_iter_num.setText(' # iter.')
        lb_iter_num.setFixedWidth(50)

        # self.tx_iter_rate = QLineEdit(self)
        # self.tx_iter_rate.setFont(self.font2)
        # self.tx_iter_rate.setText('0.005')
        # self.tx_iter_rate.setFixedWidth(20)

        self.tx_iter_num = QLineEdit(self)
        self.tx_iter_num.setFont(self.font2)
        self.tx_iter_num.setText('5')
        self.tx_iter_num.setValidator(QIntValidator())
        self.tx_iter_num.setFixedWidth(50)

        hbox_fit_method_iter = QHBoxLayout()
        #hbox_fit_method_iter.addWidget(self.pb_fit2d)
        hbox_fit_method_iter.addWidget(self.pb_fit2d_iter)
        hbox_fit_method_iter.addWidget(lb_iter_num)
        hbox_fit_method_iter.addWidget(self.tx_iter_num)
        hbox_fit_method_iter.setAlignment(QtCore.Qt.AlignTop)

        ##########################

        lb_iter = QLabel()
        lb_iter.setText('Iterative fitting parameters:')
        lb_iter.setFont(self.font1)
        lb_iter.setFixedWidth(200)

        self.chkbox_fit = QCheckBox('Update existing fitting')
        self.chkbox_fit.setFixedWidth(190)
        self.chkbox_fit.setFont(self.font2)

        self.chkbox_bound = QCheckBox('Bounds to [0, 1]')
        self.chkbox_bound.setFixedWidth(190)
        self.chkbox_bound.setFont(self.font2)

        lb_method = QLabel()
        lb_method.setFont(self.font1)
        lb_method.setText('Algorithm:')
        lb_method.setFixedWidth(140)

        lb_method1 = QLabel()
        lb_method1.setFont(self.font2)
        lb_method1.setText('updating rate:')
        lb_method1.setFixedWidth(120)

        self.tx_method1 = QLineEdit(self)
        self.tx_method1.setFont(self.font2)
        self.tx_method1.setText('0.005') # conjugate gradient updating rate
        self.tx_method1.setFixedWidth(60)

        lb_method2 = QLabel()
        lb_method2.setFont(self.font2)
        lb_method2.setText('L1 Norm lamda:')
        lb_method2.setFixedWidth(120)

        self.tx_method2 = QLineEdit(self)
        self.tx_method2.setFont(self.font2)
        self.tx_method2.setText('0.01') # lasso lambda
        self.tx_method2.setFixedWidth(60)

        lb_method3 = QLabel()
        lb_method3.setFont(self.font2)
        lb_method3.setText('rho:')
        lb_method3.setFixedWidth(120)

        self.tx_method3 = QLineEdit(self)
        self.tx_method3.setFont(self.font2)
        self.tx_method3.setText('0.01')  # ADMM rho
        self.tx_method3.setFixedWidth(60)

        self.method_group = QButtonGroup()
        self.method_group.setExclusive(True)
        self.rd_method1 = QRadioButton('Conj. Grad.')
        self.rd_method1.setFont(self.font2)
        self.rd_method1.setFixedWidth(140)
        self.rd_method1.toggled.connect(self.select_fitting_method)
        self.rd_method1.setChecked(True)

        self.rd_method2 = QRadioButton('Coord. Desc.')
        self.rd_method2.setFixedWidth(140)
        self.rd_method2.setFont(self.font2)
        self.rd_method2.toggled.connect(self.select_fitting_method)
        self.rd_method2.setChecked(False)

        self.rd_method3 = QRadioButton('ADMM')
        self.rd_method3.setFixedWidth(140)
        self.rd_method3.setFont(self.font2)
        self.rd_method3.toggled.connect(self.select_fitting_method)
        self.rd_method3.setChecked(False)

        self.method_group.addButton(self.rd_method1)
        self.method_group.addButton(self.rd_method2)
        self.method_group.addButton(self.rd_method3)

        hbox_method1 = QHBoxLayout()
        hbox_method1.addWidget(self.rd_method1)
        hbox_method1.addWidget(lb_method1)
        hbox_method1.addWidget(self.tx_method1)
        hbox_method1.setAlignment(QtCore.Qt.AlignTop)

        hbox_method2 = QHBoxLayout()
        hbox_method2.addWidget(self.rd_method2)
        hbox_method2.addWidget(lb_method2)
        hbox_method2.addWidget(self.tx_method2)
        hbox_method2.setAlignment(QtCore.Qt.AlignTop)

        hbox_method3 = QHBoxLayout()
        hbox_method3.addWidget(self.rd_method3)
        hbox_method3.addWidget(lb_method3)
        hbox_method3.addWidget(self.tx_method3)
        hbox_method3.setAlignment(QtCore.Qt.AlignTop)

        hbox_method_head = QHBoxLayout()
        hbox_method_head.addWidget(self.chkbox_fit)
        hbox_method_head.addWidget(self.chkbox_bound)
        hbox_method_head.setAlignment(QtCore.Qt.AlignTop)

        vbox_algorithm = QVBoxLayout()
        # vbox_algorithm.addLayout(hbox_method_head)
        vbox_algorithm.addLayout(hbox_method1)
        vbox_algorithm.addLayout(hbox_method2)
        vbox_algorithm.addLayout(hbox_method3)
        vbox_algorithm.setAlignment(QtCore.Qt.AlignLeft)

        vbox_method = QVBoxLayout()
        # vbox_method.addLayout(hbox_fit_method)
        vbox_method.addWidget(self.pb_fit2d)
        vbox_method.addWidget(lb_iter)

        vbox_method.addWidget(lb_method)
        vbox_method.addLayout(vbox_algorithm)
        vbox_method.addLayout(hbox_method_head)
        vbox_method.addLayout(hbox_fit_method_iter)
        vbox_method.setAlignment(QtCore.Qt.AlignLeft)

        ##########################

        self.pb_plot_roi = QPushButton('Plot ROI fit.')
        self.pb_plot_roi.setFont(self.font2)
        self.pb_plot_roi.clicked.connect(lambda return_flag: self.plot_roi_fit(1))
        self.pb_plot_roi.setEnabled(True)
        self.pb_plot_roi.setFixedWidth(105)

        lb_fit_roi = QLabel()
        lb_fit_roi.setFont(self.font2)
        lb_fit_roi.setText(' ROI #: ')
        lb_fit_roi.setFixedWidth(50)

        self.tx_fit_roi = QLineEdit(self)
        self.tx_fit_roi.setFont(self.font2)
        self.tx_fit_roi.setText('-1')
        # self.tx_fit_roi.setValidator(QIntValidator())
        self.tx_fit_roi.setFixedWidth(50)
        
        self.pb_export_roi_fit = QPushButton('Export ROI fit')
        self.pb_export_roi_fit.setFont(self.font2)
        self.pb_export_roi_fit.clicked.connect(self.export_roi_fit)
        self.pb_export_roi_fit.setEnabled(False)
        self.pb_export_roi_fit.setFixedWidth(105)

        self.chkbox_overlay_ref = QCheckBox('Overlay Ref.')
        self.chkbox_overlay_ref.setFont(self.font2)
        self.chkbox_overlay_ref.setChecked(True)
        self.chkbox_overlay_ref.setFixedWidth(105)


        hbox_plot = QHBoxLayout()
        hbox_plot.addWidget(self.pb_plot_roi)
        hbox_plot.addWidget(lb_fit_roi)
        hbox_plot.addWidget(self.tx_fit_roi)
        hbox_plot.addWidget(self.chkbox_overlay_ref)
        hbox_plot.setAlignment(QtCore.Qt.AlignTop)

        ############################################

        self.pb_save = QPushButton('Save 2D Fit')
        self.pb_save.setFont(self.font2)
        self.pb_save.clicked.connect(self.save_2Dfit)
        self.pb_save.setEnabled(False)
        self.pb_save.setFixedWidth(105)

        hbox_save = QHBoxLayout()
        hbox_save.addWidget(self.pb_export_roi_fit)
        hbox_save.addWidget(self.pb_save)
        hbox_save.setAlignment(QtCore.Qt.AlignTop)

        ##########################
        
        vbox_assemble = QVBoxLayout()
        vbox_assemble.addWidget(lb_fit2d)
        vbox_assemble.addWidget(self.lb_ref_info)
        vbox_assemble.addLayout(hbox_fit2d_range)
        vbox_assemble.addLayout(hbox_ref)
        vbox_assemble.addLayout(hbox_reset_ref)
        # vbox_assemble.addLayout(hbox_iter)
        vbox_assemble.addLayout(vbox_method)

        vbox_assemble.addLayout(hbox_plot)
        vbox_assemble.addLayout(hbox_save)
        vbox_assemble.addWidget(lb_empty)
        vbox_assemble.setAlignment(QtCore.Qt.AlignLeft)
        return vbox_assemble

    def layout_img_tools(self):
        lb_empty = QLabel()
        lb_img = QLabel()
        lb_img.setFont(self.font1)
        lb_img.setText('Threshold Mask')
        lb_img.setFixedWidth(150)

        # median filter
        self.pb_filt = QPushButton('Median Filter')
        self.pb_filt.setFont(self.font2)
        self.pb_filt.clicked.connect(self.xanes_img_smooth)
        self.pb_filt.setFixedWidth(105)

        lb_filt = QLabel()
        lb_filt.setFont(self.font2)
        lb_filt.setText(' Kernal: ')
        lb_filt.setFixedWidth(50)

        self.tx_filt = QLineEdit(self)
        self.tx_filt.setFont(self.font2)
        self.tx_filt.setText('1')
        self.tx_filt.setValidator(QIntValidator())
        self.tx_filt.setFixedWidth(50)

        hbox_filt = QHBoxLayout()
        hbox_filt.addWidget(self.pb_filt)
        hbox_filt.addWidget(lb_filt)
        hbox_filt.addWidget(self.tx_filt)
        hbox_filt.setAlignment(QtCore.Qt.AlignLeft)

        # mask1
        self.pb_mask1 = QPushButton('Gen. Mask1')
        self.pb_mask1.setFont(self.font2)
        self.pb_mask1.clicked.connect(self.generate_mask1)
        self.pb_mask1.setEnabled(True)
        self.pb_mask1.setFixedWidth(105)

        self.tx_mask1 = QLineEdit(self)
        self.tx_mask1.setFont(self.font2)
        self.tx_mask1.setText('<0.1')
        self.tx_mask1.setFixedWidth(50)

        lb_mask1 = QLabel()
        lb_mask1.setFont(self.font2)
        lb_mask1.setText('Thresh: ')
        lb_mask1.setFixedWidth(50)

        self.pb_mask1_rm = QPushButton('Rmv. Mask1')
        self.pb_mask1_rm.setFont(self.font2)
        self.pb_mask1_rm.clicked.connect(self.rm_mask1)
        self.pb_mask1_rm.setEnabled(True)
        self.pb_mask1_rm.setFixedWidth(105)

        hbox_mask1 = QHBoxLayout()
        hbox_mask1.addWidget(self.pb_mask1)
        hbox_mask1.addWidget(lb_mask1)
        hbox_mask1.addWidget(self.tx_mask1)
        hbox_mask1.addWidget(self.pb_mask1_rm)
        hbox_mask1.setAlignment(QtCore.Qt.AlignLeft)

        # mask2
        self.pb_mask2 = QPushButton('Gen. Mask2')
        self.pb_mask2.setFont(self.font2)
        self.pb_mask2.clicked.connect(self.generate_mask2)
        self.pb_mask2.setEnabled(True)
        self.pb_mask2.setFixedWidth(105)

        self.tx_mask2 = QLineEdit(self)
        self.tx_mask2.setFont(self.font2)
        self.tx_mask2.setText('>0.5')
        self.tx_mask2.setFixedWidth(50)

        lb_mask2 = QLabel()
        lb_mask2.setFont(self.font2)
        lb_mask2.setText('Thresh: ')
        lb_mask2.setFixedWidth(50)

        self.pb_mask2_rm = QPushButton('Rmv. Mask2')
        self.pb_mask2_rm.setFont(self.font2)
        self.pb_mask2_rm.clicked.connect(self.rm_mask2)
        self.pb_mask2_rm.setEnabled(True)
        self.pb_mask2_rm.setFixedWidth(105)

        hbox_mask2 = QHBoxLayout()
        hbox_mask2.addWidget(self.pb_mask2)
        hbox_mask2.addWidget(lb_mask2)
        hbox_mask2.addWidget(self.tx_mask2)
        hbox_mask2.addWidget(self.pb_mask2_rm)
        hbox_mask2.setAlignment(QtCore.Qt.AlignTop)

        # circular mask
        self.pb_mask3 = QPushButton('Circ. Mask')
        self.pb_mask3.setFont(self.font2)
        self.pb_mask3.clicked.connect(self.generate_mask3)
        self.pb_mask3.setEnabled(True)
        self.pb_mask3.setFixedWidth(105)

        self.tx_mask3 = QLineEdit(self)
        self.tx_mask3.setFont(self.font2)
        self.tx_mask3.setText('1.0')
        self.tx_mask3.setFixedWidth(50)

        lb_mask3 = QLabel()
        lb_mask3.setFont(self.font2)
        lb_mask3.setText('Ratio: ')
        lb_mask3.setFixedWidth(50)

        self.pb_mask3_rm = QPushButton('Rmv. Mask3')
        self.pb_mask3_rm.setFont(self.font2)
        self.pb_mask3_rm.clicked.connect(self.rm_mask3)
        self.pb_mask3_rm.setEnabled(True)
        self.pb_mask3_rm.setFixedWidth(105)

        hbox_mask3 = QHBoxLayout()
        hbox_mask3.addWidget(self.pb_mask3)
        hbox_mask3.addWidget(lb_mask3)
        hbox_mask3.addWidget(self.tx_mask3)
        hbox_mask3.addWidget(self.pb_mask3_rm)
        hbox_mask3.setAlignment(QtCore.Qt.AlignTop)

        # smart mask
        lb_smart_mask = QLabel()
        lb_smart_mask.setFont(self.font1)
        lb_smart_mask.setText('Clustering Mask')
        lb_smart_mask.setFixedWidth(120)

        self.pb_smart_mask = QPushButton('Gen. Mask')
        self.pb_smart_mask.setFont(self.font2)
        self.pb_smart_mask.clicked.connect(self.generate_smart_mask)
        self.pb_smart_mask.setEnabled(True)
        self.pb_smart_mask.setFixedWidth(105)

        self.pb_rem_smart_mask = QPushButton('Rmv. mask')
        self.pb_rem_smart_mask.setFont(self.font2)
        self.pb_rem_smart_mask.clicked.connect(self.rm_smart_mask)
        self.pb_rem_smart_mask.setEnabled(True)
        self.pb_rem_smart_mask.setFixedWidth(105)

        self.pb_add_smart_mask_roi = QPushButton('Add to ROI')
        self.pb_add_smart_mask_roi.setFont(self.font2)
        self.pb_add_smart_mask_roi.clicked.connect(self.add_smart_mask_toi_roi)
        self.pb_add_smart_mask_roi.setEnabled(True)
        self.pb_add_smart_mask_roi.setFixedWidth(105)

        self.pb_apply_smart_mask = QPushButton('Apply mask')
        self.pb_apply_smart_mask.setFont(self.font2)
        self.pb_apply_smart_mask.clicked.connect(self.apply_smart_mask)
        self.pb_apply_smart_mask.setEnabled(True)
        self.pb_apply_smart_mask.setFixedWidth(105)

        self.chkbox_smask = QCheckBox('Use stack')
        self.chkbox_smask.setFont(self.font2)
        self.chkbox_smask.stateChanged.connect(self.smart_mask_use_img_stack)
        self.chkbox_smask.setFixedWidth(105)

        lb_smart_mask_comp = QLabel()
        lb_smart_mask_comp.setFont(self.font2)
        lb_smart_mask_comp.setText('comp #')
        lb_smart_mask_comp.setFixedWidth(50)

        self.tx_smart_mask_comp  = QLineEdit(self)
        self.tx_smart_mask_comp.setFont(self.font2)
        self.tx_smart_mask_comp.setText('2')
        self.tx_smart_mask_comp.setFixedWidth(50)
        self.tx_smart_mask_comp.setValidator(QIntValidator())

        lb_smart_mask_start = QLabel()
        lb_smart_mask_start.setFont(self.font2)
        lb_smart_mask_start.setText('start: ')
        lb_smart_mask_start.setFixedWidth(50)

        self.tx_smart_mask_start = QLineEdit(self)
        self.tx_smart_mask_start.setFont(self.font2)
        self.tx_smart_mask_start.setFixedWidth(50)
        self.tx_smart_mask_start.setValidator(QIntValidator())

        lb_smart_mask_end = QLabel()
        lb_smart_mask_end.setFont(self.font2)
        lb_smart_mask_end.setText('  end: ')
        lb_smart_mask_end.setFixedWidth(50)

        self.tx_smart_mask_end = QLineEdit(self)
        self.tx_smart_mask_end.setFont(self.font2)
        self.tx_smart_mask_end.setFixedWidth(50)
        self.tx_smart_mask_end.setValidator(QIntValidator())

        self.pb_update_img_label = QPushButton('Update Label')
        self.pb_update_img_label.setFont(self.font2)
        self.pb_update_img_label.clicked.connect(self.smart_mask_update_label)
        self.pb_update_img_label.setEnabled(True)
        self.pb_update_img_label.setFixedWidth(105)

        lb_update_img_label = QLabel()
        lb_update_img_label.setFont(self.font2)
        lb_update_img_label.setText('  Label value:')
        lb_update_img_label.setFixedWidth(105)

        self.tx_update_img_label = QLineEdit(self)
        self.tx_update_img_label.setFont(self.font2)
        self.tx_update_img_label.setFixedWidth(105)

        lb_empty = QLabel()

        hbox_smart_mask1 = QHBoxLayout()
        # hbox_smart_mask1.addWidget(self.pb_smart_mask)
        hbox_smart_mask1.addWidget(self.chkbox_smask)
        hbox_smart_mask1.addWidget(lb_smart_mask_start)
        hbox_smart_mask1.addWidget(self.tx_smart_mask_start)
        hbox_smart_mask1.addWidget(lb_smart_mask_end)
        hbox_smart_mask1.addWidget(self.tx_smart_mask_end)
        hbox_smart_mask1.setAlignment(QtCore.Qt.AlignLeft)

        hbox_smart_mask2 = QHBoxLayout()
        hbox_smart_mask2.addWidget(self.pb_smart_mask)
        hbox_smart_mask2.addWidget(lb_smart_mask_comp)
        hbox_smart_mask2.addWidget(self.tx_smart_mask_comp)
        hbox_smart_mask2.setAlignment(QtCore.Qt.AlignLeft)

        hbox_smart_mask3 = QHBoxLayout()
        hbox_smart_mask3.addWidget(self.pb_update_img_label)
        hbox_smart_mask3.addWidget(lb_update_img_label)
        hbox_smart_mask3.addWidget(self.tx_update_img_label)
        hbox_smart_mask3.setAlignment(QtCore.Qt.AlignLeft)

        hbox_smart_mask4 = QHBoxLayout()
        hbox_smart_mask4.addWidget(self.pb_add_smart_mask_roi)
        hbox_smart_mask4.addWidget(self.pb_apply_smart_mask)
        hbox_smart_mask4.addWidget(self.pb_rem_smart_mask)
        hbox_smart_mask4.setAlignment(QtCore.Qt.AlignLeft)

        vbox_smart_mask = QVBoxLayout()
        vbox_smart_mask.addWidget(lb_empty)
        vbox_smart_mask.addWidget(lb_smart_mask)
        vbox_smart_mask.addLayout(hbox_smart_mask1)
        vbox_smart_mask.addLayout(hbox_smart_mask2)
        vbox_smart_mask.addLayout(hbox_smart_mask3)
        vbox_smart_mask.addLayout(hbox_smart_mask4)
        vbox_smart_mask.addWidget(lb_empty)
        vbox_smart_mask.setAlignment(QtCore.Qt.AlignTop)

        # noise removal
        lb_other = QLabel()
        lb_other.setFont(self.font1)
        lb_other.setText('Other Tools')
        lb_other.setFixedWidth(150)

        self.pb_rm_noise = QPushButton('Rmv. Noise')
        self.pb_rm_noise.setFont(self.font2)
        self.pb_rm_noise.clicked.connect(self.noise_removal)
        self.pb_rm_noise.setEnabled(True)
        self.pb_rm_noise.setFixedWidth(105)

        self.tx_rm_noise_level = QLineEdit(self)
        self.tx_rm_noise_level.setFont(self.font2)
        self.tx_rm_noise_level.setText('0.002')
        self.tx_rm_noise_level.setFixedWidth(50)
        self.tx_rm_noise_level.setValidator(QDoubleValidator())

        lb_rm_noise_level = QLabel()
        lb_rm_noise_level.setFont(self.font2)
        lb_rm_noise_level.setText('Thresh: ')
        lb_rm_noise_level.setFixedWidth(50)

        self.tx_rm_noise_size = QLineEdit(self)
        self.tx_rm_noise_size.setFont(self.font2)
        self.tx_rm_noise_size.setText('5')
        self.tx_rm_noise_size.setValidator(QIntValidator())
        self.tx_rm_noise_size.setFixedWidth(50)

        lb_rm_noise_size = QLabel()
        lb_rm_noise_size.setFont(self.font2)
        lb_rm_noise_size.setText('Filt_sz: ')
        lb_rm_noise_size.setFixedWidth(50)

        hbox_rm_noise = QHBoxLayout()
        hbox_rm_noise.addWidget(self.pb_rm_noise)
        hbox_rm_noise.addWidget(lb_rm_noise_level)
        hbox_rm_noise.addWidget(self.tx_rm_noise_level)
        hbox_rm_noise.addWidget(lb_rm_noise_size)
        hbox_rm_noise.addWidget(self.tx_rm_noise_size)
        hbox_rm_noise.setAlignment(QtCore.Qt.AlignLeft)

        # dilation
        lb_dilation = QLabel()
        lb_dilation.setFont(self.font2)
        lb_dilation.setText(' Mask Dilation: ')
        lb_dilation.setFixedWidth(105)

        lb_dilation_iter = QLabel()
        lb_dilation_iter.setFont(self.font2)
        lb_dilation_iter.setText('Iter(s): ')
        lb_dilation_iter.setFixedWidth(50)

        self.tx_dilation_iter = QLineEdit(self)
        self.tx_dilation_iter.setFont(self.font2)
        self.tx_dilation_iter.setText('2')
        self.tx_dilation_iter.setValidator(QIntValidator())
        self.tx_dilation_iter.setFixedWidth(50)

        self.pb_dilation = QPushButton('Dilation')
        self.pb_dilation.setFont(self.font2)
        self.pb_dilation.clicked.connect(self.mask_dilation)
        self.pb_dilation.setEnabled(True)
        self.pb_dilation.setFixedWidth(105)

        hbox_dilation = QHBoxLayout()
        hbox_dilation.addWidget(lb_dilation)
        hbox_dilation.addWidget(lb_dilation_iter)
        hbox_dilation.addWidget(self.tx_dilation_iter)
        hbox_dilation.addWidget(self.pb_dilation)
        hbox_dilation.setAlignment(QtCore.Qt.AlignTop)

        # erosion
        lb_erosion = QLabel()
        lb_erosion.setFont(self.font2)
        lb_erosion.setText(' Mask Erosion: ')
        lb_erosion.setFixedWidth(105)

        lb_erosion_iter = QLabel()
        lb_erosion_iter.setFont(self.font2)
        lb_erosion_iter.setText('Iter(s): ')
        lb_erosion_iter.setFixedWidth(50)

        self.tx_erosion_iter = QLineEdit(self)
        self.tx_erosion_iter.setFont(self.font2)
        self.tx_erosion_iter.setText('2')
        self.tx_erosion_iter.setValidator(QIntValidator())
        self.tx_erosion_iter.setFixedWidth(50)

        self.pb_erosion = QPushButton('Erosion')
        self.pb_erosion.setFont(self.font2)
        self.pb_erosion.clicked.connect(self.mask_erosion)
        self.pb_erosion.setEnabled(True)
        self.pb_erosion.setFixedWidth(105)

        hbox_erosion = QHBoxLayout()
        hbox_erosion.addWidget(lb_erosion)
        hbox_erosion.addWidget(lb_erosion_iter)
        hbox_erosion.addWidget(self.tx_erosion_iter)
        hbox_erosion.addWidget(self.pb_erosion)
        hbox_erosion.setAlignment(QtCore.Qt.AlignTop)

        # fill hole
        lb_fhole = QLabel()
        lb_fhole.setFont(self.font2)
        lb_fhole.setText(' Mask fill-hole: ')
        lb_fhole.setFixedWidth(105)

        lb_fhole_iter = QLabel()
        lb_fhole_iter.setFont(self.font2)
        lb_fhole_iter.setText('Iter(s): ')
        lb_fhole_iter.setFixedWidth(50)

        self.tx_fhole_iter_iter = QLineEdit(self)
        self.tx_fhole_iter_iter.setFont(self.font2)
        self.tx_fhole_iter_iter.setText('2')
        self.tx_fhole_iter_iter.setValidator(QIntValidator())
        self.tx_fhole_iter_iter.setFixedWidth(50)

        self.pb_fhole = QPushButton('Fill-hole')
        self.pb_fhole.setFont(self.font2)
        self.pb_fhole.clicked.connect(self.mask_fillhole)
        self.pb_fhole.setEnabled(True)
        self.pb_fhole.setFixedWidth(105)

        hbox_fhole = QHBoxLayout()
        hbox_fhole.addWidget(lb_fhole)
        hbox_fhole.addWidget(lb_fhole_iter)
        hbox_fhole.addWidget(self.tx_fhole_iter_iter)
        hbox_fhole.addWidget(self.pb_fhole)
        hbox_fhole.setAlignment(QtCore.Qt.AlignTop)

        # colormix
        lb_colormix = QLabel()
        lb_colormix.setFont(self.font2)
        lb_colormix.setText(' Color Mix: ')
        lb_colormix.setFixedWidth(105)

        self.pb_colormix = QPushButton('Color Mix')
        self.pb_colormix.setFont(self.font2)
        self.pb_colormix.clicked.connect(self.xanes_colormix)
        self.pb_colormix.setEnabled(True)
        self.pb_colormix.setFixedWidth(105)

        lb_fit_color = QLabel()
        lb_fit_color.setFont(self.font2)
        lb_fit_color.setText('Color: ')
        lb_fit_color.setFixedWidth(50)

        self.tx_fit_color = QLineEdit(self)
        self.tx_fit_color.setFont(self.font2)
        self.tx_fit_color.setText('r, g, b')
        self.tx_fit_color.setFixedWidth(50)

        hbox_colormix = QHBoxLayout()
        hbox_colormix.addWidget(lb_colormix)
        hbox_colormix.addWidget(lb_fit_color)
        hbox_colormix.addWidget(self.tx_fit_color)
        hbox_colormix.addWidget(self.pb_colormix)
        hbox_colormix.setAlignment(QtCore.Qt.AlignTop)

        # color channel
        lb_color_channel = QLabel()
        lb_color_channel.setFont(self.font2)
        lb_color_channel.setText(' Color Channel:')
        lb_color_channel.setFixedWidth(105)

        self.cb_color_channel = QComboBox()
        self.cb_color_channel.setFont(self.font2)
        self.cb_color_channel.setFixedWidth(50)
        '''
        self.sl_color = QSlider(QtCore.Qt.Horizontal)
        self.sl_color.setFocusPolicy(QtCore.Qt.StrongFocus)
        #self.sl_color.setTickPosition(QSlider.TicksBelow)
        self.sl_color.setMinimum(0)
        self.sl_color.setMaximum(100)
        self.sl_color.setValue(50)
        #self.sl_color.setTickInterval(100)
        self.sl_color.valueChanged.connect(self.slider_color_scale)
        self.sl_color.setFixedWidth(115)

        self.lb_colormax = QLabel()
        self.lb_colormax = QLabel()
        self.lb_colormax.setFont(self.font2)
        self.lb_colormax.setText('x 1.00 ')
        self.lb_colormax.setFixedWidth(50)
        '''
        self.tx_color_scale = QLineEdit(self)
        self.tx_color_scale.setFont(self.font2)
        self.tx_color_scale.setText('1.0')
        self.tx_color_scale.setFixedWidth(50)

        self.pb_color_scale = QPushButton('Apply')
        self.pb_color_scale.setFont(self.font2)
        self.pb_color_scale.clicked.connect(self.xanes_colormix)
        self.pb_color_scale.setFixedWidth(55)

        self.pb_color_scale_up = QPushButton('>')
        self.pb_color_scale_up.setFont(self.font2)
        self.pb_color_scale_up.clicked.connect(self.xanes_color_scale_up)
        self.pb_color_scale_up.setFixedWidth(20)

        self.pb_color_scale_down = QPushButton('<')
        self.pb_color_scale_down.setFont(self.font2)
        self.pb_color_scale_down.clicked.connect(self.xanes_color_scale_down)
        self.pb_color_scale_down.setFixedWidth(20)

        hbox_color_channel = QHBoxLayout()
        hbox_color_channel.addWidget(lb_color_channel)
        hbox_color_channel.addWidget(self.cb_color_channel)
        hbox_color_channel.addWidget(self.pb_color_scale_down)
        hbox_color_channel.addWidget(self.tx_color_scale)
        hbox_color_channel.addWidget(self.pb_color_scale_up)
        hbox_color_channel.addWidget(self.pb_color_scale)
        hbox_color_channel.setAlignment(QtCore.Qt.AlignTop)

        # assemble
        vbox_img_assemble = QVBoxLayout()
        vbox_img_assemble.addWidget(lb_img)
        vbox_img_assemble.addLayout(hbox_filt)
        vbox_img_assemble.addLayout(hbox_mask1)
        vbox_img_assemble.addLayout(hbox_mask2)
        vbox_img_assemble.addLayout(hbox_mask3)
        vbox_img_assemble.addLayout(vbox_smart_mask)
        vbox_img_assemble.addWidget(lb_other)
        vbox_img_assemble.addLayout(hbox_rm_noise)
        vbox_img_assemble.addWidget(lb_empty)
        vbox_img_assemble.addLayout(hbox_dilation)
        vbox_img_assemble.addLayout(hbox_erosion)
        vbox_img_assemble.addLayout(hbox_fhole)
        vbox_img_assemble.addLayout(hbox_colormix)
        vbox_img_assemble.addLayout(hbox_color_channel)
        vbox_img_assemble.addWidget(lb_empty)
        vbox_img_assemble.setAlignment(QtCore.Qt.AlignLeft)
        return vbox_img_assemble

    def layout_xanes_prep(self):
        lb_empty = QLabel()
        lb_prep = QLabel()
        lb_prep.setFont(self.font1)
        lb_prep.setText('Preparation')
        lb_prep.setFixedWidth(150)

        self.pb_norm_txm = QPushButton('Norm. TMX (-log)')
        self.pb_norm_txm.setFont(self.font2)
        self.pb_norm_txm.clicked.connect(self.norm_txm)
        self.pb_norm_txm.setFixedWidth(150)

        self.pb_align = QPushButton('Align Img')
        self.pb_align.setFont(self.font2)
        self.pb_align.clicked.connect(self.xanes_align_img)
        self.pb_align.setFixedWidth(150)

        self.pb_align_roi = QPushButton('Align Img (ROI)')
        self.pb_align_roi.setFont(self.font2)
        self.pb_align_roi.clicked.connect(self.xanes_align_img_roi)
        self.pb_align_roi.setFixedWidth(150)

        self.pb_rmbg = QPushButton('Remove Bkg. (ROI)')
        self.pb_rmbg.setFont(self.font2)
        self.pb_rmbg.clicked.connect(self.remove_bkg)
        self.pb_rmbg.setFixedWidth(150)

        self.align_group = QButtonGroup()
        self.align_group.setExclusive(True)
        self.rd_ali1 = QRadioButton('StackReg')
        self.rd_ali1.setFixedWidth(150)
        self.rd_ali1.setChecked(True)
        # self.rd_norm1.toggled.connect(self.select_file)

        self.cb_ali = QComboBox()
        self.cb_ali.setFont(self.font2)
        self.cb_ali.addItem('  translation')
        self.cb_ali.addItem('  rigid')
        self.cb_ali.addItem('  scaled rotation')
        self.cb_ali.setFixedWidth(150)

        self.rd_ali2 = QRadioButton('Cross-Correlation')
        self.rd_ali2.setFixedWidth(160)
        self.rd_ali2.setChecked(False)
        # self.rd_norm2.toggled.connect(self.select_file)

        self.align_group.addButton(self.rd_ali1)
        self.align_group.addButton(self.rd_ali2)
        self.align_group = QButtonGroup()
        self.align_group.setExclusive(True)

        lb_ali_method = QLabel()
        lb_ali_method.setText('Align method:')
        lb_ali_method.setFont(self.font2)

        self.pb_apply_shft = QPushButton('Apply shift')
        self.pb_apply_shft.setFont(self.font2)
        self.pb_apply_shft.clicked.connect(self.apply_shift)
        self.pb_apply_shft.setFixedWidth(150)

        self.pb_save_shft = QPushButton('Save shift list')
        self.pb_save_shft.setFont(self.font2)
        self.pb_save_shft.clicked.connect(self.save_shift)
        self.pb_save_shft.setFixedWidth(150)

        self.pb_load_shft = QPushButton('Load shift list')
        self.pb_load_shft.setFont(self.font2)
        self.pb_load_shft.clicked.connect(self.load_shift)
        self.pb_load_shft.setFixedWidth(150)

        self.lb_shift = QLabel()
        self.lb_shift.setFont(self.font2)
        self.lb_shift.setText('  Shift list: None')
        self.lb_shift.setFixedWidth(150)

        lb_ali_ref = QLabel()
        lb_ali_ref.setFont(self.font2)
        lb_ali_ref.setText('  Ref. Image: ')
        lb_ali_ref.setFixedWidth(90)

        lb_ali_roi = QLabel()
        lb_ali_roi.setFont(self.font2)
        lb_ali_roi.setText('  ROI Index: ')
        lb_ali_roi.setFixedWidth(90)

        self.tx_ali_ref = QLineEdit(self)
        self.tx_ali_ref.setFont(self.font2)
        self.tx_ali_ref.setText('-1')
        self.tx_ali_ref.setValidator(QIntValidator())
        self.tx_ali_ref.setFixedWidth(50)

        self.tx_ali_roi = QLineEdit(self)
        self.tx_ali_roi.setFont(self.font2)
        self.tx_ali_roi.setText('-1')
        self.tx_ali_roi.setValidator(QIntValidator())
        self.tx_ali_roi.setFixedWidth(50)

        hbox_prep = QHBoxLayout()
        hbox_prep.addWidget(self.pb_norm_txm)
        hbox_prep.addWidget(self.pb_rmbg)
        hbox_prep.setAlignment(QtCore.Qt.AlignLeft)

        hbox_ali = QHBoxLayout()
        hbox_ali.addWidget(self.pb_align)
        hbox_ali.addWidget(lb_ali_ref)
        hbox_ali.addWidget(self.tx_ali_ref)
        hbox_ali.setAlignment(QtCore.Qt.AlignLeft)

        hbox_ali_roi = QHBoxLayout()
        hbox_ali_roi.addWidget(self.pb_align_roi)
        hbox_ali_roi.addWidget(lb_ali_roi)
        hbox_ali_roi.addWidget(self.tx_ali_roi)
        hbox_ali_roi.setAlignment(QtCore.Qt.AlignLeft)


        hbox_ali_stackreg = QHBoxLayout()
        hbox_ali_stackreg.addWidget(self.rd_ali1)
        hbox_ali_stackreg.addWidget(self.cb_ali)
        hbox_ali_stackreg.setAlignment(QtCore.Qt.AlignLeft)

        vbox_ali_method = QVBoxLayout()
        # hbox_ali_method.addWidget(lb_ali_method)
        vbox_ali_method.addLayout(hbox_ali_stackreg)
        vbox_ali_method.addWidget(self.rd_ali2)
        vbox_ali_method.setAlignment(QtCore.Qt.AlignTop)

        hbox_shft = QHBoxLayout()
        hbox_shft.addWidget(self.pb_save_shft)
        hbox_shft.addWidget(self.pb_load_shft)
        hbox_shft.setAlignment(QtCore.Qt.AlignLeft)

        hbox_shft1 = QHBoxLayout()
        hbox_shft1.addWidget(self.pb_apply_shft)
        hbox_shft1.addWidget(self.lb_shift)
        hbox_shft1.setAlignment(QtCore.Qt.AlignLeft)

        vbox_prep = QVBoxLayout()
        vbox_prep.addWidget(lb_prep)
        vbox_prep.addLayout(hbox_prep)
        vbox_prep.addLayout(hbox_ali)
        vbox_prep.addLayout(hbox_ali_roi)
        vbox_prep.addLayout(vbox_ali_method)
        vbox_prep.addLayout(hbox_shft)
        vbox_prep.addLayout(hbox_shft1)
        vbox_prep.addWidget(lb_empty)
        return vbox_prep

    def layout_canvas(self):
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        lb_empty = QLabel()
        lb_empty2 = QLabel()
        lb_empty2.setFixedWidth(10)
        self.canvas1 = MyCanvas(obj=self)
        self.toolbar = NavigationToolbar(self.canvas1,self)
        self.sl1 = QScrollBar(QtCore.Qt.Horizontal)
        self.sl1.setMaximum(0)
        self.sl1.setMinimum(0)
        self.sl1.valueChanged.connect(self.sliderval)

        self.cb1 = QComboBox()
        self.cb1.setFont(self.font2)
        self.cb1.addItem('Raw image')
        self.cb1.setFixedWidth(460)
        self.cb1.currentIndexChanged.connect(self.update_canvas_img)

        self.pb_del = QPushButton('Del. single image')
        self.pb_del.setToolTip('Delete single image, it will delete the same slice in other images (e.g., "raw image", "aligned image", "background removed" ')
        self.pb_del.setFont(self.font2)
        self.pb_del.clicked.connect(self.delete_single_img)
        self.pb_del.setEnabled(False)
        self.pb_del.setFixedWidth(150)

        self.pb_save_img_stack = QPushButton('Save image stack')
        self.pb_save_img_stack.setFont(self.font2)
        self.pb_save_img_stack.clicked.connect(self.save_img_stack)
        self.pb_save_img_stack.setFixedWidth(150)

        self.pb_save_img_single = QPushButton('Save current image')
        self.pb_save_img_single.setFont(self.font2)
        self.pb_save_img_single.clicked.connect(self.save_img_single)
        self.pb_save_img_single.setFixedWidth(150)

        hbox_can_l = QHBoxLayout()
        hbox_can_l.addWidget(self.cb1)
        hbox_can_l.addWidget(self.pb_del)
        hbox_can_l.setAlignment(QtCore.Qt.AlignLeft)

        hbox_can_save = QHBoxLayout()
        hbox_can_save.addWidget(self.cb1)
        hbox_can_save.addWidget(self.pb_save_img_single)
        hbox_can_save.addWidget(self.pb_save_img_stack)
        hbox_can_save.setAlignment(QtCore.Qt.AlignLeft)

        self.lb_x_l = QLabel()
        self.lb_x_l.setFont(self.font2)
        self.lb_x_l.setText('x: ')
        self.lb_x_l.setFixedWidth(80)

        self.lb_y_l = QLabel()
        self.lb_y_l.setFont(self.font2)
        self.lb_y_l.setText('y: ')
        self.lb_y_l.setFixedWidth(80)

        self.lb_z_l = QLabel()
        self.lb_z_l.setFont(self.font2)
        self.lb_z_l.setText('intensity: ')
        self.lb_z_l.setFixedWidth(120)

        lb_cmap = QLabel()
        lb_cmap.setFont(self.font2)
        lb_cmap.setText('colormap: ')
        lb_cmap.setFixedWidth(80)

        cmap = ['gray', 'bone', 'viridis', 'terrain', 'gnuplot', 'bwr', 'plasma', 'PuBu', 'summer', 'rainbow', 'jet']
        self.cb_cmap = QComboBox()
        self.cb_cmap.setFont(self.font2)
        for i in cmap:
            self.cb_cmap.addItem(i)
        self.cb_cmap.setCurrentText('viridis')
        self.cb_cmap.currentIndexChanged.connect(self.change_colormap)
        self.cb_cmap.setFixedWidth(80)

        self.pb_adj_cmap = QPushButton('Auto Contrast')
        self.pb_adj_cmap.setFont(self.font2)
        self.pb_adj_cmap.clicked.connect(self.auto_contrast)
        self.pb_adj_cmap.setEnabled(True)
        self.pb_adj_cmap.setFixedWidth(120)

        lb_cmax = QLabel()
        lb_cmax.setFont(self.font2)
        lb_cmax.setText('cmax: ')
        lb_cmax.setFixedWidth(40)
        lb_cmin = QLabel()
        lb_cmin.setFont(self.font2)
        lb_cmin.setText('cmin: ')
        lb_cmin.setFixedWidth(40)

        self.tx_cmax = QLineEdit(self)
        self.tx_cmax.setFont(self.font2)
        self.tx_cmax.setFixedWidth(60)
        self.tx_cmax.setText('1.')
        self.tx_cmax.setValidator(QDoubleValidator())
        self.tx_cmax.setEnabled(True)

        self.tx_cmin = QLineEdit(self)
        self.tx_cmin.setFont(self.font2)
        self.tx_cmin.setFixedWidth(60)
        self.tx_cmin.setText('0.')
        self.tx_cmin.setValidator(QDoubleValidator())
        self.tx_cmin.setEnabled(True)

        self.pb_set_cmap = QPushButton('Set')
        self.pb_set_cmap.setFont(self.font2)
        self.pb_set_cmap.clicked.connect(self.set_contrast)
        self.pb_set_cmap.setEnabled(True)
        self.pb_set_cmap.setFixedWidth(60)

        hbox_chbx_l = QHBoxLayout()
        hbox_chbx_l.addWidget(self.lb_x_l)
        hbox_chbx_l.addWidget(self.lb_y_l)
        hbox_chbx_l.addWidget(self.lb_z_l)
        hbox_chbx_l.addWidget(lb_empty)
        hbox_chbx_l.addWidget(self.pb_save_img_single)
        hbox_chbx_l.addWidget(self.pb_save_img_stack)        
        hbox_chbx_l.setAlignment(QtCore.Qt.AlignLeft)

        hbox_cmap = QHBoxLayout()
        hbox_cmap.addWidget(lb_cmap)
        hbox_cmap.addWidget(self.cb_cmap)
        hbox_cmap.addWidget(self.pb_adj_cmap)
        hbox_cmap.addWidget(lb_cmin)
        hbox_cmap.addWidget(self.tx_cmin)
        hbox_cmap.addWidget(lb_cmax)
        hbox_cmap.addWidget(self.tx_cmax)
        hbox_cmap.addWidget(self.pb_set_cmap)
        hbox_chbx_l.addWidget(lb_empty)
        hbox_cmap.setAlignment(QtCore.Qt.AlignLeft)

        vbox_can1 = QVBoxLayout()
        vbox_can1.addWidget(self.toolbar)
        vbox_can1.addWidget(self.canvas1)
        vbox_can1.addWidget(self.sl1)
        vbox_can1.addLayout(hbox_can_l)
        vbox_can1.addLayout(hbox_chbx_l)
        vbox_can1.addLayout(hbox_cmap)
        vbox_can1.setAlignment(QtCore.Qt.AlignLeft)
        return vbox_can1

    def check_xanes_fit_requirement(self, img_stack):
        n_ref = len(self.spectrum_ref)
        n_eng = len(self.xanes_eng)
        return_flag = 1
        if n_ref < 2:
            self.msg += ';   # of reference spectrum need to be larger than 2, fitting fails ...'
            return_flag = 0
        elif img_stack.shape[0] != n_eng:
            self.msg += ';   # of stack image is not equal to energies, fitting fails ...'
            return_flag = 0
        return return_flag

    def load_xanes_ref(self):
        options = QFileDialog.Option()
        options |= QFileDialog.DontUseNativeDialog
        file_type = 'txt files (*.txt)'
        fn, _ = QFileDialog.getOpenFileName(xanes, "QFileDialog.getOpenFileName()", "", file_type, options=options)
        if fn:
            try:
                print(fn)
                fn_ref = fn.split('/')[-1]
                print(f'selected reference: {fn_ref}')
                self.lb_ref_info.setText(self.lb_ref_info.text() + '\n' + f'ref #{self.num_ref}: ' + fn_ref)
                self.lb_ref_info.setStyleSheet('color: rgb(200, 50, 50);')
                QApplication.processEvents()
                # self.spectrum_ref[f'ref{self.num_ref}'] = np.array(pd.read_csv(fn, sep=' '))
                self.spectrum_ref[f'ref{self.num_ref}'] = np.loadtxt(fn)
                self.num_ref += 1
            except:
                print('un-supported xanes reference format')

    def reset_xanes_ref(self):
        self.num_ref = 0
        self.lb_ref_info.setText('Reference spectrum:')
        self.spectrum_ref = {}
        self.xanes_fit_cost = 0
        self.tx_elem.setText('')
        self.elem_label = []

    def select_fitting_method(self):
        if self.rd_method1.isChecked(): # conjugated gradient
            self.fitting_method = 1
            self.tx_method1.setEnabled(True)
            self.tx_method2.setEnabled(True)
            self.tx_method3.setEnabled(False)
        elif self.rd_method2.isChecked():
            self.fitting_method = 2
            self.tx_method1.setEnabled(False)
            self.tx_method2.setEnabled(True)
            self.tx_method3.setEnabled(False)
        elif self.rd_method3.isChecked():
            self.fitting_method = 3
            self.tx_method1.setEnabled(False)
            self.tx_method2.setEnabled(True)
            self.tx_method3.setEnabled(True)
        else:
            self.fitting_method = 0
            self.tx_method1.setEnabled(False)
            self.tx_method2.setEnabled(False)
            self.tx_method3.setEnabled(False)

    def choose_image_for_fittting(self):
        canvas = self.canvas1
        img_stack = canvas.img_stack
        self.msg = f'Fit image: using "{self.cb1.currentText()}"'
        return_flag = self.check_xanes_fit_requirement(img_stack)

        if not return_flag:
            if self.dataset_used_for_fitting <= 0:
                img_stack = self.img_xanes
                self.msg = 'Fit 2D xanes: using "Image Raw"'
                return_flag = 1
            elif self.dataset_used_for_fitting == 1:
                img_stack = self.img_update
                self.msg = 'Fit 2D xanes: using "Image update"'
                return_flag = 1
            elif self.dataset_used_for_fitting == 2:
                img_stack = self.img_regulation
                self.msg = 'Fit 2D xanes: using "Image regulation"'
                return_flag = 1
        else:
            if "raw" in self.msg.lower():
                self.dataset_used_for_fitting = 0
            elif "update" in self.msg.lower():
                self.dataset_used_for_fitting = 1
            elif "regulation" in self.msg.lower():
                self.dataset_used_for_fitting = 2
            else:
                self.dataset_used_for_fitting = -1
        self.update_msg()
        return return_flag, img_stack

    def fit_2d_xanes(self):
        self.pb_fit2d.setDisabled(True)
        QApplication.processEvents()
        #canvas = self.canvas1
        return_flag, img_stack = self.choose_image_for_fittting()
        img_stack = self.smooth(img_stack * self.mask)
        if return_flag:
            try:
                eng_s = float(self.tx_fit2d_s.text())
                eng_e = float(self.tx_fit2d_e.text())
                fit_eng_s, fit_eng_e = find_nearest(self.xanes_eng, eng_s), find_nearest(self.xanes_eng, eng_e)
                tmp = np.array(self.xanes_eng[fit_eng_s: fit_eng_e] >= self.spectrum_ref['ref0'][0,0]) * np.array(self.xanes_eng[fit_eng_s: fit_eng_e] <= self.spectrum_ref['ref0'][-1,0])
                fit_region = np.arange(fit_eng_s, fit_eng_e)[tmp]
                self.xanes_2d_fit, self.xanes_2d_fit_offset, self.xanes_fit_cost = fit_2D_xanes_non_iter(img_stack[fit_region], self.xanes_eng[fit_region], self.spectrum_ref)
                if self.cb1.findText('XANES Fit (ratio, summed to 1)') < 0:
                    self.cb1.addItem('XANES Fit (ratio, summed to 1)')
                if self.cb1.findText('XANES Fit (Elem. concentration)') < 0:
                    self.cb1.addItem('XANES Fit (Elem. concentration)')
                if self.cb1.findText('XANES Fit (thickness)') < 0:
                    self.img_pre_edge_sub_mean = img_stack[fit_region][-1]
                    self.cb1.addItem('XANES Fit (thickness)')
                if self.cb1.findText('XANES Fit error') < 0:
                    self.cb1.addItem('XANES Fit error')
                if self.cb1.findText('XANES Fit offset') < 0:
                    self.cb1.addItem('XANES Fit offset')
                # save to data_summary:
                # 1: xanes_fit ratio:
                img = self.xanes_2d_fit
                img_sum = np.sum(img, axis=0, keepdims=True)
                img_sum[np.abs(img_sum) < 1e-6] = 1e6
                img = img / img_sum
                img = rm_abnormal(img)
                self.data_summary['XANES Fit (ratio, summed to 1)'] = self.smooth(img) * self.mask
                # 2: xanes_fit concentration:
                img = img * self.img_pre_edge_sub_mean
                img = rm_abnormal(img)
                self.data_summary['XANES Fit (Elem. concentration)'] = self.smooth(img) * self.mask
                # 3: xanes_fit error:
                self.data_summary['XANES Fit error'] = self.xanes_fit_cost.copy()
                # 4: xanes_fit offset:
                self.data_summary['XANES Fit offset'] = self.xanes_2d_fit_offset.copy()
                # finish saving to data_summary

                self.cb1.setCurrentText('XANES Fit (ratio, summed to 1)')
                elem = self.tx_elem.text()
                elem = elem.replace(' ', '')
                elem = elem.replace(';', ',')
                elem = elem.split(',')
                if elem[0] == '':
                    elem = []
                self.elem_label = elem
                self.update_canvas_img()
                self.msg = '2D fitting finished. "XANES Fit" has been added for imshow'
                self.pb_plot_roi.setEnabled(True)
                self.pb_export_roi_fit.setEnabled(True)
                self.pb_colormix.setEnabled(True)
                self.pb_save.setEnabled(True)
                num_ref = len(self.spectrum_ref)
                self.cb_color_channel.clear()
                for i in range(num_ref):
                    if self.cb_color_channel.findText(f'{i}') < 0:
                        self.cb_color_channel.addItem(f'{i}')
            except:
                print('fitting fails ...')
                self.msg = 'fitting fails ..., may need to check energy lists'
        self.update_msg()
        self.pb_fit2d.setEnabled(True)
        QApplication.processEvents()

    def reset_xanes_fit(self):
        self.reset_xanes_ref()
        self.xanes_2d_fit = None
        self.img_pre_edge_sub_mean = np.array([1])
        self.pb_plot_roi.setDisabled(True)

    def fit_2d_xanes_iter(self):
        self.pb_fit2d_iter.setEnabled(False)
        QApplication.processEvents()
        return_flag, img_stack = self.choose_image_for_fittting()
        img_stack = self.smooth(img_stack * self.mask)
        if return_flag:
            if self.chkbox_bound.isChecked():
                bounds = [0, 1]
            else:
                bounds = []
            eng_s = float(self.tx_fit2d_s.text())
            eng_e = float(self.tx_fit2d_e.text())
            fit_eng_s, fit_eng_e = find_nearest(self.xanes_eng, eng_s), find_nearest(self.xanes_eng, eng_e)
            tmp = np.array(self.xanes_eng[fit_eng_s: fit_eng_e] >= self.spectrum_ref['ref0'][0, 0]) * np.array(self.xanes_eng[fit_eng_s: fit_eng_e] <= self.spectrum_ref['ref0'][-1, 0])
            fit_region = np.arange(fit_eng_s, fit_eng_e)[tmp]
            try:
                num_iter = int(self.tx_iter_num.text())
                if self.chkbox_fit.isChecked(): # initializing using existing fitting results
                    coef0 = self.xanes_2d_fit
                    offset = self.xanes_2d_fit_offset
                else:
                    coef0 = None
                    offset = None

                if coef0 is None:
                    self.msg = 'Using random initial guess. It may take few minutes ...'
                else:
                    self.msg = 'Using existing fitting as initial guess'
                self. update_msg()

                self.pb_fit2d_iter.setEnabled(False)

                if self.fitting_method == 1:
                    learning_rate = float(self.tx_method1.text())
                    fit_iter_lambda = float(self.tx_method2.text())
                    self.xanes_2d_fit, self.xanes_2d_fit_offset, self.xanes_fit_cost = \
                        fit_2D_xanes_iter(img_stack[fit_region], self.xanes_eng[fit_region], self.spectrum_ref,
                                          coef0, offset, learning_rate, num_iter, bounds=bounds, fit_iter_lambda=fit_iter_lambda)
                elif self.fitting_method == 2 or self.fitting_method == 3:
                    fit_iter_lambda = float(self.tx_method2.text())
                    rho = float(self.tx_method3.text())
                    self.xanes_2d_fit, self.xanes_2d_fit_offset, self.xanes_fit_cost = \
                        fit_2D_xanes_iter2(img_stack[fit_region], self.xanes_eng[fit_region], self.spectrum_ref,
                                           coef0, offset, fit_iter_lambda, rho, num_iter, bounds=bounds, method=self.fitting_method-1)
                self.pb_fit2d_iter.setEnabled(True)
                QApplication.processEvents()
                if self.cb1.findText('XANES Fit (ratio, summed to 1)') < 0:
                    self.cb1.addItem('XANES Fit (ratio, summed to 1)')
                if self.cb1.findText('XANES Fit (Elem. concentration)') < 0:
                    self.cb1.addItem('XANES Fit (Elem. concentration)')
                if self.cb1.findText('XANES Fit (thickness)') < 0:
                    self.img_pre_edge_sub_mean = img_stack[fit_region][-1]
                    self.cb1.addItem('XANES Fit (thickness)')
                if self.cb1.findText('XANES Fit error') < 0:
                    self.cb1.addItem('XANES Fit error')
                self.cb1.setCurrentText('XANES Fit (ratio, summed to 1)')
                self.update_canvas_img()
                self.msg = 'Iterative fitting finished'
                self.pb_export_roi_fit.setEnabled(True)
                self.pb_colormix.setEnabled(True)
                self.pb_plot_roi.setEnabled(True)
                self.pb_save.setEnabled(True)
                num_ref = len(self.spectrum_ref)
                for i in range(num_ref):
                    if self.cb_color_channel.findText(f'{i}') < 0:
                        self.cb_color_channel.addItem(f'{i}')
            except:
                print('iterative fitting fails ...')
                self.msg = 'iterative fitting fails ...'
            finally:
                self.update_msg()
        self.pb_fit2d_iter.setEnabled(True)

    def plot_xanes_ref(self):
        plt.figure()
        legend = []
        elem = self.tx_elem.text()
        elem = elem.replace(' ','')
        elem = elem.replace(';', ',')
        elem = elem.split(',')
        if elem[0] == '':
            elem = []
        try:
            for i in range(self.num_ref):
                try:
                    plot_label = elem[i]
                except:
                    plot_label = f'ref_{i}'
                self.elem_label.append(plot_label)
                spec = self.spectrum_ref[f'ref{i}']
                line, = plt.plot(spec[:,0], spec[:,1], label=plot_label)
                legend.append(line)
            print(legend)
            plt.legend(handles=legend)
            plt.show()
        except:
            self.msg = 'un-recognized reference spectrum format'
            self.update_msg()

    def bin_image(self):
        try:
            img = self.img_xanes
            s = img.shape
            tmp = self.cb_bin.currentText()
            if tmp == '2 x 2':
                b = 2
            elif tmp == '4 x 4':
                b = 4
            else:
                b = 1
            if s[1]%b or s[2]%b:
                ss = [s[0], s[1]//b*b, s[2]//b*b]
                img = img[:, :ss[1], :ss[2]]
            self.img_xanes = bin_ndarray(img, (s[0], s[1]//b, s[2]//b), 'mean')
            self.msg = 'image shape: {0}'.format(self.img_xanes.shape)
            self.update_msg()
            self.update_canvas_img()
        except:
            self.msg = 'xanes image not exist'
            self.update_msg()

    def scale_image(self):
        try:
            tmp = deepcopy(self.canvas1.img_stack)
            scale = float(self.tx_scale_img.text())
            tmp *= scale
            self.img_update = tmp.copy()
            self.update_canvas_img()
            self.msg = f'scale the current image by {scale}'
            if self.cb1.findText('Image updated') < 0:
                self.cb1.addItem('Image updated')
            self.cb1.setCurrentText('Image updated')
        except:
            self.msg = 'fail to scale image'
        finally:
            self.update_msg()

    def generate_mask1(self):
        try:
            tmp = np.squeeze(self.canvas1.current_img)
            mask = np.ones(tmp.shape)

            tmp1 = self.tx_mask1.text()
            if tmp1[0] == '<':
                thresh = float(tmp1[1:])
                mask[tmp < thresh] = 0
            elif tmp1[0] == '>':
                thresh = float(tmp1[1:])
                mask[tmp > thresh] = 0
            else:
                thresh = float(tmp1)
                mask[tmp > thresh] = 0
            self.canvas1.mask = self.canvas1.mask * mask
            self.mask1 = mask
            self.mask = self.canvas1.mask
            self.update_canvas_img()
            if self.cb1.findText('Mask') < 0:
                self.cb1.addItem('Mask')
            self.pb_mask1.setStyleSheet('color: rgb(200, 50, 50);')
        except:
            self.msg = 'invalid mask '
            self.update_msg()
            self.pb_mask1.setStyleSheet('color: rgb(0,0,0);')

    def generate_mask2(self):
        try:
            tmp = deepcopy(self.canvas1.current_img)
            mask = np.ones(tmp.shape)
            tmp1 = self.tx_mask2.text()
            if tmp1[0] == '<':
                thresh = float(tmp1[1:])
                mask[tmp < thresh] = 0
            elif tmp1[0] == '>':
                thresh = float(tmp1[1:])
                mask[tmp > thresh] = 0
            else:
                thresh = float(tmp1)
                mask[tmp > thresh] = 0
            self.canvas1.mask = self.canvas1.mask * mask
            self.mask2 = mask
            self.mask = self.canvas1.mask
            self.update_canvas_img()
            if self.cb1.findText('Mask') < 0:
                self.cb1.addItem('Mask')
            self.pb_mask2.setStyleSheet('color: rgb(200, 50, 50);')
        except:
            self.msg = 'invalid mask '
            self.update_msg()
            self.pb_mask2.setStyleSheet('color: rgb(0,0,0);')

    def generate_mask3(self):
        try:
            tmp = self.canvas1.current_img
            ratio = float(self.tx_mask3.text())
            s = np.squeeze(tmp).shape
            x = np.arange(s[0])
            y = np.arange(s[1])
            X, Y = np.meshgrid(y, x)
            X = X / s[1]
            Y = Y / s[0]
            mask = np.float32(((X-0.5)**2 + (Y-0.5)**2)<(ratio/2)**2)
            self.canvas1.mask = self.canvas1.mask * mask
            self.mask3 = mask
            self.mask = self.canvas1.mask
            self.update_canvas_img()
            if self.cb1.findText('Mask') < 0:
                self.cb1.addItem('Mask')
            self.pb_mask3.setStyleSheet('color: rgb(200,50,50);')
        except:
            self.msg = 'invalid mask '
            self.update_msg()

    def generate_smart_mask(self):
        try:
            self.pb_smart_mask.setEnabled(False)
            self.pb_smart_mask.setText('Wait ...')
            self.pb_smart_mask.setStyleSheet('color: rgb(200, 200, 200);')
            n = int(self.tx_smart_mask_comp.text())
            self.smart_mask_comp = max(n, 2)
            self.tx_smart_mask_comp.setText(str(self.smart_mask_comp))
            QApplication.processEvents()
            canvas = self.canvas1
            if self.chkbox_smask.isChecked():
                st = int(self.tx_smart_mask_start.text())
                en = int(self.tx_smart_mask_end.text())
                img_stack = canvas.img_stack[st:en] * self.mask

                self.smart_mask, self.img_labels = kmean_mask(img_stack, self.smart_mask_comp)
            else:
                img_stack = np.squeeze(canvas.current_img * self.mask)
                self.smart_mask, self.img_labels, self.img_compress = kmean_mask(img_stack, self.smart_mask_comp)
                if self.cb1.findText('Image compress') < 0:
                    self.cb1.addItem('Image compress')
            if self.cb1.findText('Smart Mask') < 0:
                self.cb1.addItem('Smart Mask')
            if self.cb1.findText('Image Labels') < 0:
                self.cb1.addItem('Image Labels')
            self.cb1.setCurrentText('Smart Mask')
            self.msg = 'Smart Mask generated'
            self.update_canvas_img()
            #self.pb_smart_mask.setStyleSheet('color: rgb(200,50,50);')
        except:
            self.msg = 'invalid mask '
            #self.pb_smart_mask.setStyleSheet('color: rgb(0,0,0);')
        finally:
            self.update_msg()
            self.pb_smart_mask.setStyleSheet('color: rgb(0,0,0);')
            self.pb_smart_mask.setText('Gen. Mask')
            self.pb_smart_mask.setEnabled(True)
            QApplication.processEvents()

    def smart_mask_use_img_stack(self):
        if self.chkbox_smask.isChecked():
            s = self.canvas1.img_stack.shape
            if not len(s) == 3:
                st, en = 0, 0
            else:
                st, en = 0, s[0]
            self.tx_smart_mask_start.setText(str(st))
            self.tx_smart_mask_end.setText(str(en))

    def add_smart_mask_toi_roi(self):
        if self.cb1.currentText() == 'Smart Mask':
            self.smart_mask_current = self.smart_mask[self.sl1.value()]
            roi_name = f'roi_SM_{self.sl1.value()}'
            self.canvas1.roi_add_to_list(roi_name = roi_name)
            self.msg = f'{roi_name} has been added to the ROI list'
            self.update_msg()

    def apply_smart_mask(self):
        if self.cb1.currentText() == 'Smart Mask':
            self.smart_mask_current = self.smart_mask[self.sl1.value()]
            self.canvas1.mask = self.canvas1.mask * self.smart_mask_current
            self.mask = self.canvas1.mask
            if self.cb1.findText('Mask') < 0:
                self.cb1.addItem('Mask')
            self.cb1.setCurrentText('Mask')
            self.update_canvas_img()
            roi_name = f'roi_SM_{self.sl1.value()}'
            self.msg = f'{roi_name} has been set to the "Mask"'

    def smart_mask_update_label(self):
        val = np.int8(self.tx_update_img_label.text().split(','))
        img_label = np.zeros(self.img_labels.shape)
        try:
            assert (len(val) == self.smart_mask_comp), 'number of value no equals to number of smart mask component'
            for i in range(self.smart_mask_comp):
                mask = self.smart_mask[i]
                img_label += mask * val[i]

            self.img_labels = img_label.copy()
            self.update_canvas_img()
            del img_label
        except:
            self.msg = 'fails to update image label'

    def rm_mask1(self):
        try:
            self.canvas1.mask = self.mask2 * self.mask3 * self.smart_mask_current
            self.mask1 = np.array([1])
            self.mask = self.canvas1.mask
            self.update_canvas_img()
            self.pb_mask1.setStyleSheet('color: rgb(0,0,0);')
        except:
            self.msg = 'something wrong in removing mask1'
            self.update_msg()

    def rm_mask2(self):
        try:
            self.canvas1.mask = self.mask1 * self.mask3 * self.smart_mask_current
            self.mask2 = np.array([1])
            self.mask = self.canvas1.mask
            self.update_canvas_img()
            self.pb_mask2.setStyleSheet('color: rgb(0,0,0);')
        except:
            self.msg = 'something wrong in removing mask2'
            self.update_msg()

    def rm_mask3(self):
        try:
            self.canvas1.mask = self.mask1 * self.mask2 * self.smart_mask_current
            self.mask3 = np.array([1])
            self.mask = self.canvas1.mask
            self.update_canvas_img()
            self.pb_mask3.setStyleSheet('color: rgb(0,0,0);')
        except:
            self.msg = 'something wrong in removing mask3'
            self.update_msg()

    def rm_smart_mask(self):
        try:
            self.canvas1.mask = self.mask1 * self.mask2 * self.mask3
            self.smart_mask_current = np.array([1])
            self.mask = self.canvas1.mask
            self.update_canvas_img()
            self.pb_smart_mask.setStyleSheet('color: rgb(0,0,0);')
        except:
            self.msg = 'something wrong in removing smart mask'
            self.update_msg()

    def mask_dilation(self):
        s = int(self.tx_dilation_iter.text())
        struct = ndimage.generate_binary_structure(2, 1)
        struct = ndimage.iterate_structure(struct, s).astype(int)
        if self.cb1.currentText() == 'Smart Mask':
            try:
                img_index = self.canvas1.current_img_index
                img = self.canvas1.img_stack[img_index]
                self.smart_mask[img_index] = ndimage.binary_dilation(img, structure=struct).astype(img.dtype)
                self.mask = self.mask * self.smart_mask[img_index]
                self.canvas1.mask = self.mask
                self.update_canvas_img()
                #self.canvas1.update_img_one(self.smart_mask[img_index], img_index)
            except:
                self.msg = 'fails to perform dilation on "Smart Mask"'
                self.update_msg()
        else:
            try:
                img = np.squeeze(self.canvas1.mask)
                self.mask = ndimage.binary_dilation(img, structure=struct).astype(img.dtype)
                self.canvas1.mask = self.mask
                self.update_canvas_img()
            except:
                self.msg = 'fails to perform dilation on "mask"'
                self.update_msg()

    def mask_erosion(self):
        s = int(self.tx_dilation_iter.text())
        struct = ndimage.generate_binary_structure(2, 1)
        struct = ndimage.iterate_structure(struct, s).astype(int)
        if self.cb1.currentText() == 'Smart Mask':
            try:
                img_index = self.canvas1.current_img_index
                img = self.canvas1.img_stack[img_index]
                self.smart_mask[img_index] = ndimage.binary_erosion(img, structure=struct).astype(img.dtype)
                self.mask = self.mask * self.smart_mask[img_index]
                self.canvas1.mask = self.mask
                self.update_canvas_img()
            except:
                self.msg = 'fails to perform dilation on "Smart Mask"'
                self.update_msg()
        else:
            try:
                img = np.squeeze(self.canvas1.mask)
                self.mask = ndimage.binary_erosion(img, structure=struct).astype(img.dtype)
                self.canvas1.mask = self.mask
                self.update_canvas_img()
            except:
                self.msg = 'fails to perform dilation on "mask"'
                self.update_msg()

    def mask_fillhole(self):
        s = int(self.tx_dilation_iter.text())
        struct = ndimage.generate_binary_structure(2, 1)
        struct = ndimage.iterate_structure(struct, s).astype(int)
        if self.cb1.currentText() == 'Smart Mask':
            try:
                img_index = self.canvas1.current_img_index
                img = self.canvas1.img_stack[img_index]
                self.smart_mask[img_index] = ndimage.binary_fill_holes(img, structure=struct).astype(img.dtype)
                self.mask = self.mask * self.smart_mask[img_index]
                self.canvas1.mask = self.mask
                self.update_canvas_img()
            except:
                self.msg = 'fails to perform fill_holes on "Smart Mask"'
                self.update_msg()
        else:
            try:
                img = np.squeeze(self.canvas1.mask)
                self.mask = ndimage.binary_fill_holes(img, structure=struct).astype(img.dtype)
                self.canvas1.mask = self.mask
                self.update_canvas_img()
            except:
                self.msg = 'fails to perform dilation on "mask"'
                self.update_msg()


    def _select_xanes_fit_img(self):
        if self.dataset_used_for_fitting == 0:
            img = self.img_xanes
        elif self.dataset_used_for_fitting == 1:
            img = self.img_update
        elif self.dataset_used_for_fitting == 2:
            img = self.img_regulation
        else:
            img = []
        return img

    def _roi_fit(self):
        try:
            n = self.tx_fit_roi.text()
            roi_selected = 'roi_' + n
            canvas = self.canvas1
            roi_list = canvas.roi_list
        except:
            print(f'{roi_selected} not exist')
            n = '-1'
            roi_selected = 'roi_-1'
        try:
            eng_s = self.xanes_eng[0]
            eng_e = self.xanes_eng[-1]
            fit_eng_s, fit_eng_e = find_nearest(self.xanes_eng, eng_s), find_nearest(self.xanes_eng, eng_e)
            tmp = np.array(self.xanes_eng[fit_eng_s: fit_eng_e] > self.spectrum_ref['ref0'][0, 0]) * \
                  np.array(self.xanes_eng[fit_eng_s: fit_eng_e] < self.spectrum_ref['ref0'][-1, 0])
            fit_region = np.arange(fit_eng_s, fit_eng_e)[tmp]
            #roi_selected = -1
        except:
            self.msg = 'something wrong (e.g., check ROI, reference spectrum, etc.)'
            self.update_msg()
            x_data = [0]
            y_data = [0]
            y_fit = y_data
            fit_success = 0
            cord = [0, 0, 0, 0]
            fit_coef = [0]
            fit_offset = 0
            try:
                if (type(roi_selected) is str) and ('roi_' in roi_selected) and (int(n) >= 0) \
                        and (not 'SM' in roi_selected):
                    roi_cord = np.int32(np.array(roi_list[roi_selected][:4]))
                    print(f'{roi_cord}')
                    a, b, c, d = roi_cord[0], roi_cord[1], roi_cord[2], roi_cord[3]
                    x1 = min(a, c)
                    x2 = max(a, c)
                    y1 = min(b, d)
                    y2 = max(b, d)
                    cord = [x1, x2, y1, y2]
                    fit_coef = self.xanes_2d_fit[:, y1:y2, x1:x2]
                    try:
                        fit_offset = self.xanes_2d_fit_offset[:, y1:y2, x1:x2]
                        fit_offset = np.mean(np.mean(fit_offset, axis=1), axis=1)
                    except:
                        fit_offset = 0
                    fit_coef = np.mean(np.mean(fit_coef, axis=1), axis=1)
                    fit_success = 0.5
            except:
                pass

            return x_data, y_data, y_fit, fit_coef, fit_offset, cord, fit_success, roi_selected

        img = self._select_xanes_fit_img()
        if len(img):
            img = img[fit_region]
            img = self.smooth(img)

        x_data = self.xanes_eng[fit_region]

        try:
            if (type(roi_selected) is str) and ('roi_' in roi_selected) and (not 'SM' in roi_selected):
                roi_cord = np.int32(np.array(roi_list[roi_selected][:4]))
                print(f'{roi_cord}')
                a, b, c, d = roi_cord[0], roi_cord[1], roi_cord[2], roi_cord[3]
                x1 = min(a, c)
                x2 = max(a, c)
                y1 = min(b, d)
                y2 = max(b, d)
                cord = [x1, x2, y1, y2]
                if len(img):
                    prj = img[:, y1:y2, x1:x2]
                    y_data = np.mean(np.mean(prj, axis=1), axis=1)
                else: # img = []
                    y_data = np.zeros(x_data.shape)
                fit_coef = self.xanes_2d_fit[:, y1:y2, x1:x2]
                try:
                    fit_offset = self.xanes_2d_fit_offset[:, y1:y2, x1:x2]
                    fit_offset = np.mean(np.mean(fit_offset, axis=1), axis=1)
                except:
                    fit_offset = 0
                fit_coef = np.mean(np.mean(fit_coef, axis=1), axis=1)
            else:
                if 'SM' in roi_selected:
                    sm_index = int(roi_selected.split('_')[-1])
                    mask = self.smart_mask[sm_index]
                elif '-1' in roi_selected:
                    mask = self.mask
                else:
                    raise IndexError
                if len(img):
                    prj = img * mask
                    prj_mean = np.zeros([prj.shape[0], 1, 1])
                    prj_mean[:, 0, 0] = np.sum(np.sum(prj, axis=1), axis=1) / np.sum(mask)
                    fit_coef, fit_offset, fit_cost = fit_2D_xanes_non_iter(prj_mean, x_data, self.spectrum_ref)
                    fit_coef = np.squeeze(fit_coef)
                    fit_offset = np.squeeze(fit_offset)
                    y_data = np.sum(np.sum(prj, axis=1), axis=1) / np.sum(mask)
                else: # img = []
                    fit_coef = self.xanes_2d_fit * mask
                    fit_offset = self.xanes_2d_fit_offset * mask
                    fit_coef = np.sum(np.sum(fit_coef, axis=1), axis=1) / np.sum(mask)
                    fit_offset = np.sum(fit_offset) / np.sum(mask)
                    y_data = np.zeros(x_data.shape)
                cord = [0, 0, 0, 0]

            y_fit = 0
            for i in range(self.num_ref):
                ref = self.spectrum_ref[f'ref{i}']
                tmp = interp1d(ref[:,0], ref[:,1], kind='cubic')
                ref_interp = tmp(x_data)
                y_fit += fit_coef[i] * ref_interp
            y_fit += fit_offset
            fit_success = 1
        except:
            self.msg = 'something wrong (e.g., check ROI)'
            self.update_msg()
            y_data = np.zeros(x_data.shape)
            y_fit = y_data
            fit_success = 0
            cord = [0,0,0,0]
            fit_coef = [0]
            fit_offset = 0
        return x_data, y_data, y_fit, fit_coef, fit_offset, cord, fit_success, roi_selected

    '''
    def _roi_fit_mean(self):
        eng_s = self.xanes_eng[0]
        eng_e = self.xanes_eng[-1]
        fit_eng_s, fit_eng_e = find_nearest(self.xanes_eng, eng_s), find_nearest(self.xanes_eng, eng_e)
        tmp = np.array(self.xanes_eng[fit_eng_s: fit_eng_e] > self.spectrum_ref['ref0'][0, 0]) * np.array(self.xanes_eng[fit_eng_s: fit_eng_e] < self.spectrum_ref['ref0'][-1, 0])
        fit_region = np.arange(fit_eng_s, fit_eng_e)[tmp]
        roi_selected = 1
        canvas = self.canvas1
        roi_list = canvas.roi_list
        if not self.img_update.shape == self.img_regulation.shape:
            img = deepcopy(self.img_update[fit_region])
        else:
            img = deepcopy(self.img_regulation[fit_region])
        x_data = self.xanes_eng[fit_region]
        try:
            n = int(self.tx_fit_roi.text())
            roi_selected = 'roi_' + str(n)
        except:
            print('index should be integer')
            n = 0
        n_roi = self.lst_roi.count()
        if n > n_roi or n < 0:
            self.msg = 'roi not exist, will show the fitting average'
            roi_selected = -1
            self.update_msg()
        if roi_selected:
            print(f'{roi_selected}')
            try:
                roi_cord = np.int32(np.array(roi_list[roi_selected][:4]))
                print(f'{roi_cord}')
                a, b, c, d = roi_cord[0], roi_cord[1], roi_cord[2], roi_cord[3]
                x1 = min(a, c)
                x2 = max(a, c)
                y1 = min(b, d)
                y2 = max(b, d)
                cord = [x1, x2, y1, y2]
                try:
                    prj = img[:, y1:y2, x1:x2]
                    s = prj.shape
                    y_data = np.mean(np.mean(prj, axis=1), axis=1) 
                except:
                    y_data = []
                #prj_mean = np.zeros([prj.shape[0], 1, 1])
                #prj_mean[:,0,0] = np.mean(np.mean(prj, axis=1), axis=1)
                #fit_coef, fit_cost = fit_2D_xanes_non_iter(prj_mean, self.xanes_eng[fit_eng_s:fit_eng_e],self.spectrum_ref)
                #fit_coef = np.squeeze(fit_coef)
                #fit_cost = np.squeeze(fit_cost)
                
                fit_coef = self.xanes_2d_fit[:, y1:y2, x1:x2]
                fit_coef = np.mean(np.mean(fit_coef, axis=1), axis=1)
                try:
                    fit_offset = self.xanes_2d_fit_offset[:, y1:y2, x1:x2]
                    fit_offset = np.mean(np.mean(fit_offset, axis=1), axis=1)
                except:
                    fit_offset = 0
                               
                y_fit = 0
                for i in range(self.num_ref):
                    ref = self.spectrum_ref[f'ref{i}']
                    tmp = interp1d(ref[:,0], ref[:,1], kind='cubic')
                    ref_interp = tmp(x_data).reshape(y_data.shape)
                    y_fit += fit_coef[i] * ref_interp
                y_fit += fit_offset
                fit_success = 1
            except:
                self.msg = 'something wrong, no fitting performed'
                self.update_msg()
                y_data = np.zeros(x_data.shape)
                y_fit = y_data
                fit_success = 0
                cord = [0,0,0,0]
                fit_coef = [0]
                fit_offset = 0
        else:
            y_data = np.zeros(x_data.shape)
            y_fit = y_data
            fit_success = 0
            cord = [0,0,0,0]
            fit_coef = [0]
            fit_offset = [0]
        return x_data, y_data, y_fit, fit_coef, fit_offset, cord, fit_success, roi_selected
    '''

    def plot_roi_fit(self, return_flag):
        x_data, y_data, y_fit, fit_coef, fit_offset, cord, fit_success, roi_selected = self._roi_fit()
        elem = self.tx_elem.text()
        elem = elem.replace(' ', '')
        elem = elem.replace(';', ',')
        elem = elem.split(',')
        if elem[0] == '':
            elem = []
        if fit_success > 0:
            fit_coef_sum = np.sum(fit_coef)
            title = f'{roi_selected}:  '
            for i in range(len(fit_coef)):
                try:
                    plot_label = elem[i]
                except:
                    plot_label = f'ref#{i}'
                self.elem_label.append(plot_label)
                title += plot_label + f': {fit_coef[i] / fit_coef_sum:.3f}, '
            title += f'offset: {float(fit_offset):.3f}'
            plt.figure()
            legend = []
            if fit_success == 1:
                line_raw, = plt.plot(x_data, y_data, 'b.', label='Experiment data')
                legend.append(line_raw)
                line_fit, = plt.plot(x_data, y_fit, color='r', label='Fitted')
                legend.append(line_fit)
                if self.chkbox_overlay_ref.isChecked():
                    t_color = ['g', 'orange', 'm', 'c', 'y']
                    line_ref = {}
                    try:
                        scale = (np.max(y_fit) - fit_offset) / np.max(self.spectrum_ref['ref0'][:,1])
                        for i in range(self.num_ref):
                            ref_name = f'ref{i}'
                            scale_i = fit_coef[i]/fit_coef_sum
                            x_ref = self.spectrum_ref[ref_name][:,0]
                            y_ref = self.spectrum_ref[ref_name][:,1]
                            line_ref[ref_name], = plt.plot(x_ref, y_ref*scale*scale_i+fit_offset,
                                                           alpha=0.6, color=t_color[i%5], label=ref_name)
                            plt.xlim([x_data[0]-0.02, x_data[-1]+0.02])
                            legend.append(line_ref[ref_name])
                    except:
                        pass
            plt.legend(handles=legend)
            plt.title(title)
            plt.show()
        if return_flag:
            return x_data, y_data, y_fit, fit_coef, cord, fit_success

    def export_roi_fit(self):
        x_data, y_data, y_fit, fit_coef, cord, fit_success = self.plot_roi_fit(return_flag=1)
        dir_path = self.fpath + '/ROI_fit'
        '''
        try:
            os.makedirs(dir_path)
            make_directory_success = True
        except:
            if os.path.exists(dir_path):
                make_directory_success = True
            else:
                print('access directory: ' + dir_path + ' failed')
                make_directory_success = False
        '''
        if fit_success:
        #if make_directory_success and fit_success:
            n = self.tx_fit_roi.text()
            label_raw = 'roi_' + n
            label_fit = label_raw + '_fit'
            fn_spec = dir_path + '/' + 'spectrum_' + label_fit + '.txt'
            #fn_cord = dir_path + '/' + 'coordinates_' + label_fit + '.txt'
            roi_dict_spec = {'X_eng': pd.Series(self.xanes_eng)}
            #roi_dict_cord = {}
            if fit_success:
                roi_dict_spec[label_raw] = pd.Series(y_data)
                roi_dict_spec[label_fit] = pd.Series(y_fit)
                roi_dict_spec[label_raw + '_fit_coef'] = pd.Series(fit_coef)
                #roi_dict_cord[label_raw] = pd.Series([cord[0], cord[1], cord[2], cord[3]], index=['x1', 'y1', 'x2', 'y2'])
            df_spec = pd.DataFrame(roi_dict_spec)
            #df_cord = pd.DataFrame(roi_dict_cord)

            options = QFileDialog.Option()
            options |= QFileDialog.DontUseNativeDialog
            file_type = 'txt files (*.txt)'
            fn, _ = QFileDialog.getSaveFileName(self, 'Save File', "", file_type, options=options)
            if fn[-4:] == '.txt':
                fn_spec = fn
            else:
                fn_spec = fn + '.txt'
            with open(fn_spec, 'w') as f:
                df_spec.to_csv(f, float_format='%.4f', sep=' ', index=False)
            #with open(fn_cord, 'w') as f:
            #    df_cord.to_csv(f, float_format='%.4f', sep=' ')
            self.msg = 'Fitted ROI spectrum file saved:    ' + fn_spec
            self.update_msg()
        else:
            self.msg = 'export fails'
            self.update_msg()

    def select_file(self):
        self.tx_hdf_xanes.setEnabled(True)
        self.tx_hdf_eng.setEnabled(True)

    def noise_removal(self):
        try:
            canvas = self.canvas1
            img = canvas.current_img.copy()
            noise_level = float(self.tx_rm_noise_level.text())
            filter_size = int(self.tx_rm_noise_size.text())
            self.img_rm_noise = rm_noise(img, noise_level, filter_size)
            if self.cb1.findText('Noise removal') < 0:
                self.cb1.addItem('Noise removal')
            self.cb1.setCurrentText('Noise removal')
            self.update_canvas_img()
            self.msg = 'Noise removed'
        except:
            self.msg = 'fails to remove noise'
        finally:
            self.update_msg()

    def convert_percentage_image(self):
        try:
            p_min = float(self.tx_cvt_min.text())
            p_max = float(self.tx_cvt_max.text())
            img = self.xanes_peak_fit.copy()
            self.peak_percentage = rm_abnormal((img - p_min) / (p_max - p_min))
            if self.cb1.findText('Peak percentage') < 0:
                self.cb1.addItem('Peak percentage')
            self.cb1.setCurrentText('Peak percentage')
        except:
            self.msg = 'fails to convert'
            self.update_msg()

    def fit_edge_curve(self):
        from scipy.interpolate import UnivariateSpline
        try:
            x, y = self.external_spec[:,0], self.external_spec[:,1]
            if self.tx_edge_s.text():
                xs = float(self.tx_edge_s.text())
            else:
                xs = x[0]
                self.tx_edge_s.setText(f'{xs}')
            if self.tx_edge_e.text():
                xe = float(self.tx_edge_e.text())
            else:
                xe = x[-1]
                self.tx_edge_e.setText(f'{xe}')
            if self.tx_edge_order.text():
                k = float(self.tx_edge_order.text())
            else:
                k = 2
            k = np.min([k, 3])
            k = np.max([k, 2])
            self.tx_edge_order.setText(f'{k}')
            try:
                x0 = float(self.tx_edge_pos.text())
            except:
                x0 = xs
            xs_id = find_nearest(x, xs)
            xe_id = find_nearest(x, xe)
            x0_id = find_nearest(x, x0)
            x = x[xs_id: xe_id]
            y = y[xs_id: xe_id]
            w = float(self.tx_pre_edge_wt.text())
            wt = np.ones(len(x))
            wt[:x0_id - xs_id] = w
            edge_smooth = float(self.tx_edge_smooth.text())
            s = UnivariateSpline(x, y, k=k, s=edge_smooth, w=wt)
            xx = np.linspace(x[0], x[-1], 1001)
            y_eval = s(xx)
            pos = np.argmax(np.abs(np.diff(y_eval)))
            if self.rd_peak_max.isChecked():
                factor = 1
            else:
                factor = -1
            pos_max = np.argmax(y_eval*factor)
            plt.figure()
            plt.plot(x, y, '.', label='experiment data')
            plt.plot(xx, y_eval, 'r', label='fitting')
            plt.plot(xx[pos], y_eval[pos], 'rx',markersize=10, label=f'x = {xx[pos]:2.6f}')
            plt.plot(xx[pos_max], y_eval[pos_max], 'g+', markersize=16, label=f'x = {xx[pos_max]:2.6f}')
            plt.legend()
            plt.title('Curve fitting')
            plt.show()
        except:
            self.msg = 'Fails to fit curve'
            self.update_msg()

    def find_edge_peak_single(self):
        from scipy.interpolate import UnivariateSpline
        try:
            x, y, _, roi_selected = self.extract_roi_spectrum_data(use_current_image=1)
            xs = float(self.tx_edge_s.text())
            xe = float(self.tx_edge_e.text())
            k = float(self.tx_edge_order.text())
            k = np.min([k, 3])
            k = np.max([k, 2])
            try:
                x0 = float(self.tx_edge_pos.text())
            except:
                x0 = xs
            xs_id = find_nearest(self.xanes_eng, xs)
            xe_id = find_nearest(self.xanes_eng, xe)
            x0_id = find_nearest(self.xanes_eng, x0)
            x = x[xs_id: xe_id]
            y = y[xs_id: xe_id]
            w = float(self.tx_pre_edge_wt.text())
            wt = np.ones(len(x))
            wt[:x0_id - xs_id] = w
            edge_smooth = float(self.tx_edge_smooth.text())
            s = UnivariateSpline(x, y, k=k, s=edge_smooth, w=wt)
            xx = np.linspace(x[0], x[-1], 1001)
            y_eval = s(xx)
            pos = np.argmax(np.abs(np.diff(y_eval)))
            if self.rd_peak_max.isChecked():
                factor = 1
            else:
                factor = -1
            pos_max = np.argmax(y_eval*factor)

            plt.figure()
            plt.plot(x, y, '.', label='experiment data')
            plt.plot(xx, y_eval, 'r', label='fitting')
            plt.plot(xx[pos], y_eval[pos], 'rx',markersize=10, label=f'x = {xx[pos]:2.6f}')
            plt.plot(xx[pos_max], y_eval[pos_max], 'g+', markersize=16, label=f'x = {xx[pos_max]:2.6f}')
            plt.legend()
            plt.title(roi_selected)
            plt.show()
        except:
            self.msg = 'Fails to find edge for ROI'
            self.update_msg()

    def find_edge_peak_image(self):
        try:
            self.pb_find_edge_img.setEnabled(False)
            self.msg = 'Fitting image, it may take few miniuts, please wait ...'
            self.update_msg()
            QApplication.processEvents()
            fit_order = int(self.tx_edge_order.text())
            xs = float(self.tx_edge_s.text())
            xe = float(self.tx_edge_e.text())
            try:
                x0 = float(self.tx_edge_pos.text())
            except:
                x0 = xs
            xs_id = find_nearest(self.xanes_eng, xs)
            xe_id = find_nearest(self.xanes_eng, xe)
            x0_id = find_nearest(self.xanes_eng, x0)
            eng = self.xanes_eng
            x = eng[xs_id:xe_id]
            w = float(self.tx_pre_edge_wt.text())
            xx = np.linspace(x[0], x[-1], 1001)
            wt = np.ones(len(x))
            wt[:x0_id-xs_id] = w
            edge_smooth = float(self.tx_edge_smooth.text())
            #return_flag, img = self.choose_image_for_fittting()
            img = self.canvas1.img_stack.copy()
            img = self.smooth(img * self.mask)
            s = img.shape
            self.xanes_edge_fit = np.zeros([1, s[1], s[2]])
            self.xanes_peak_fit = np.zeros([1, s[1], s[2]])
            self.xanes_peak_fit_height = np.zeros([1, s[1], s[2]])
            self.spl = {}
            if self.rd_peak_max.isChecked():
                factor = 1
            else:
                factor = -1
            time_s = time.time()
            for i in range(s[1]):
                if not i % 10:
                    print(f'row # {i:4d}: {time.time() - time_s:3.2f} sec')
                for j in range(s[2]):
                    y = img[xs_id:xe_id, i, j]
                    spl = UnivariateSpline(x, y,k=fit_order,s=edge_smooth, w=wt)
                    tmp_edge = np.argmax(np.abs(np.diff(spl(xx))))
                    tmp_peak = np.argmax(spl(xx) * factor)
                    self.xanes_edge_fit[0, i,j] = xx[tmp_edge]
                    self.xanes_peak_fit[0, i,j] = xx[tmp_peak]
                    self.xanes_peak_fit_height[0, i, j] = spl(xx[tmp_peak])
                    #tmp = np.argmax(np.abs(np.diff(spl(xx))))
                    self.spl[f'{i},{j}'] = spl
            if self.chkbox_edge.isChecked():
                if self.cb1.findText('XANES Edge Fit') < 0:
                    self.cb1.addItem('XANES Edge Fit')
                self.cb1.setCurrentText('XANES Edge Fit')
            if self.chkbox_peak.isChecked():
                if self.cb1.findText('XANES Peak Fit') < 0:
                    self.cb1.addItem('XANES Peak Fit')
                if self.cb1.findText('XANES Peak Fit Height') < 0:
                    self.cb1.addItem('XANES Peak Fit Height')
                self.cb1.setCurrentText('XANES Edge Fit Height')
            self.msg = 'Fit image finished'
        except:
            self.msg = 'Fails in fitting image'
        finally:
            self.pb_find_edge_img.setEnabled(True)
            QApplication.processEvents()
            self.update_msg()

    def extract_roi_spectrum_data(self, use_current_image = 0):
        try:
            canvas = self.canvas1
            x, y_data = [], []
            x = self.xanes_eng

            if use_current_image:
                img = canvas.img_stack
            else:
                if self.dataset_used_for_fitting == 0:
                    img = deepcopy(self.img_xanes)
                elif self.dataset_used_for_fitting == 1:
                    img = deepcopy(self.img_update)
                elif self.dataset_used_for_fitting == 2:
                    img = deepcopy(self.img_regulation)
                else:
                    img = canvas.img_stack

            img = self.smooth(img * self.mask)

            roi_list = canvas.roi_list
            roi_selected = f'roi_{int(self.tx_overlay_roi.text())}'
            roi_cord = np.int32(np.array(roi_list[roi_selected][:4]))
            print(f'{roi_cord}')
            a, b, c, d = roi_cord[0], roi_cord[1], roi_cord[2], roi_cord[3]
            x1 = min(a, c)
            x2 = max(a, c)
            y1 = min(b, d)
            y2 = max(b, d)
        except:
            x1, x2 = 0, img.shape[2]
            y1, y2 = 0, img.shape[1]
            roi_selected = 'image average'
            x = []
        if len(self.mask) == 1:
            y_data = np.mean(np.mean(img[:, y1:y2, x1:x2], axis=1), axis=1)
        else:
            n = np.sum(self.mask[y1:y2, x1:x2])
            y_data = np.sum(np.sum(img[:, y1:y2, x1:x2], axis=1), axis=1) / n
        if len(x) == 0:
            x = np.arange(len(y_data))
        cord = [y1, y2, x1, x2]
        return x, y_data, cord, roi_selected

    '''
    def plot_edge_estimation_obsolete(self):
        try:
            x0 = float(self.tx_edge_pos.text())
            scale = float(self.tx_edge_scale.text())
            coef = float(self.tx_edge_coef.text())
            try:
                xs = float(self.tx_edge_s.text())
                xe = float(self.tx_edge_e.text())
                xs_id = find_nearest(self.xanes_eng, xs)
                xe_id = find_nearest(self.xanes_eng, xe)
                x = np.linspace(self.xanes_eng[xs_id], self.xanes_eng[xe_id], 101)
            except:
                x = np.linspace(-3, 3, 101)

            y_est = erf_fun(x, x0, scale, coef)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x, y_est, label='estimation')
            ax.legend()
            plt.show()

            if self.chkbox_overlay_roi.isChecked():
                x_data, y_data, _, roi_selected= self.extract_roi_spectrum_data()
                ax.plot(x_data, y_data, 'r+', label='spectrum')
                ax.legend()
                plt.title(roi_selected)
                plt.show()
        except:
            self.msg = 'fails to draw'
            self.update_msg()
    '''

    def plot_fit_edge_peak_roi(self):
        try:
            xs = float(self.tx_edge_s.text())
            xe = float(self.tx_edge_e.text())
            x0 = float(self.tx_edge_pos.text())

            xs_id = find_nearest(self.xanes_eng, xs)
            xe_id = find_nearest(self.xanes_eng, xe)
            x0_id = find_nearest(self.xanes_eng, x0)

            w = float(self.tx_pre_edge_wt.text())

            x, y, cord, roi_selected = self.extract_roi_spectrum_data()
            x = x[xs_id:xe_id]
            y = y[xs_id:xe_id]
            xx = np.linspace(x[0], x[-1], 1001)
            wt = np.ones(len(x))
            wt[:x0_id - xs_id] = w

            row = np.arange(cord[0], cord[1])
            col = np.arange(cord[2], cord[3])
            y_eval = 0
            n = 0
            for i in row:
                for j in col:
                    y_eval += self.spl[f'{int(i)},{int(j)}'](xx)
                    n += 1
            y_eval /= n
            plt.figure()
            plt.plot(x, y, '.', label='experiment data')
            plt.plot(xx, y_eval, 'r', label='fitting')
            plt.legend()
            plt.title(roi_selected)
            plt.show()
        except:
            pass

    def convert_rgb_img(self, img, color_vec):
        s = img.shape
        img_color = np.zeros([s[1], s[2], 3])
        cR, cG, cB = 0, 0, 0
        for i in range(s[0]):
            cR += img[i] * color_vec[i][0]
            cG += img[i] * color_vec[i][1]
            cB += img[i] * color_vec[i][2]
        img_color[:, :, 0] = cR
        img_color[:, :, 1] = cG
        img_color[:, :, 2] = cB
        return img_color

    def xanes_apply_color_scale(self):
        self.xanes_colormix()

    def xanes_color_scale_up(self):
        current_scale = float(self.tx_color_scale.text())
        if current_scale >= 1:
            self.tx_color_scale.setText(f'{current_scale+1:2.1f}')
        else:
            self.tx_color_scale.setText(f'{current_scale+0.1:2.1f}')
        self.xanes_colormix()

    def xanes_color_scale_down(self):
        current_scale = float(self.tx_color_scale.text())
        if current_scale >= 2:
            self.tx_color_scale.setText(f'{current_scale - 1:2.1f}')
        elif current_scale > 0.1:
            self.tx_color_scale.setText(f'{current_scale - 0.1:2.1f}')
        self.xanes_colormix()

    def xanes_colormix(self):
        try:
            self.msg = ''
            self.update_msg()
            canvas = self.canvas1
            color = self.tx_fit_color.text()
            color = color.replace(' ','')
            color = color.replace(';', ',')
            color = color.split(',')
            if color[0] == '':
                color = ['r', 'g', 'b', 'c', 'p', 'y']
            color_vec = self.convert_rgb_vector(color)
            self.color_vec = color_vec
            if len(self.img_colormix_raw) == 0:
                try:
                    img = canvas.img_stack * canvas.mask
                    self.img_colormix_raw = deepcopy(img)
                except:
                    img = []

                if not len(img) == len(self.spectrum_ref):
                    '''
                    self.msg = 'invalid image stack for colormix, will using "XANES fit (concentration)" to demonstrate'
                    img = self.xanes_2d_fit * self.mask1 * self.mask2 * self.img_pre_edge_sub_mean
                    self.update_msg()
                    self.img_colormix_raw = deepcopy(img)
                    '''
                    self.cb_color_channel.clear()
                    for i in range(len(img)):
                        self.cb_color_channel.addItem(str(i))
            else:
                img = deepcopy(self.img_colormix_raw)
            '''
            # set color scale for selected color channel
            scale = self.get_slider_color_scale_value()
            selected_channel = int(self.cb_color_channel.currentText())
            img[selected_channel] *= scale
            '''
            # set color scale for selected color channel
            scale = float(self.tx_color_scale.text())
            selected_channel = int(self.cb_color_channel.currentText())
            img[selected_channel] *= scale
            img_color = self.convert_rgb_img(img, color_vec)
            self.img_colormix = deepcopy(img_color)
            print('plot the colormix ')

            if self.cb1.findText('Color mix') < 0:
                self.cb1.addItem('Color mix')
                self.cb1.setCurrentText('Color mix')
            else:
                self.cb1.setCurrentText('Color mix')
                self.update_canvas_img()
        except:
            pass

    def convert_rgb_vector(self, color):
        n = len(color)
        vec = np.zeros([n, 3])
        for i in range(n):
            if color[i] == 'r':
                vec[i] = [1, 0, 0]
            if color[i] == 'g':
                vec[i] = [0, 1, 0]
            if color[i] == 'b':
                vec[i] = [0, 0, 1]
            if color[i] == 'c':
                vec[i] = [0, 1, 1]
            if color[i] == 'p':
                vec[i] = [1, 0, 1]
            if color[i] == 'y':
                vec[i] = [1, 1, 0]
        return vec 

    def save_2Dfit(self):
        pre_s = float(self.tx_fit_pre_s.text())
        pre_e = float(self.tx_fit_pre_e.text())
        post_s = float(self.tx_fit_post_s.text())
        post_e = float(self.tx_fit_post_e.text())
        label = self.elem_label
        mask = self.canvas1.mask
        s = mask.shape
        if len(s) == 2:
            mask = mask.reshape(1, s[0], s[1])
        img = self.xanes_2d_fit  #/ np.sum(self.xanes_2d_fit, axis=0, keepdims=True)
        img_sum = np.sum(img, axis=0, keepdims=True)
        img_sum[np.abs(img_sum) < 1e-6] = 1e6
        img = img / img_sum
        img = rm_abnormal(img)
        if not label:
            label = [f'ref{i}' for i in range(self.num_ref)]
        try:
            options = QFileDialog.Option()
            options |= QFileDialog.DontUseNativeDialog
            file_type = 'hdf files (*.h5)'
            fn, _ = QFileDialog.getSaveFileName(self, 'Save File', "", file_type, options=options)
            if fn[-3:] != '.h5':
                fn += '.h5'
            with h5py.File(fn, 'w') as hf:
                hf.create_dataset('X_eng', data=self.xanes_eng)
                hf.create_dataset('pre_edge', data=[pre_s, pre_e])
                hf.create_dataset('post_edge', data=[post_s, post_e])
                hf.create_dataset('unit', data='keV')
                for key, val in self.data_summary.items():
                    try:
                        attr_name = key.replace(' ', '_')
                        hf.create_dataset(attr_name, data = np.array(val, dtype=np.float32))
                    except:
                        pass
                for i in range(self.num_ref):
                    hf.create_dataset(f'ref{i}', data=self.spectrum_ref[f'ref{i}'])
            '''
            with h5py.File(self.fn_raw_image, 'a') as hf:
                #hf.create_dataset('X_eng', data=self.xanes_eng)
                if 'pre_edge' in hf.keys():
                    hf.__delitem__('pre_edge')
                hf.create_dataset('pre_edge', data=[pre_s, pre_e])
                if 'post_edge' in hf.keys():
                    hf.__delitem__('post_edge')
                hf.create_dataset('post_edge', data=[post_s, post_e])
                if 'unit' in hf.keys():
                    hf.__delitem__('unit')
                hf.create_dataset('unit', data='keV')
                if 'fitted' in hf.keys():
                    hf.__delitem__('fitted')
                hf.create_dataset('fitted', data=1)
                for key, val in self.data_summary.items():
                    try:
                        attr_name = key.replace(' ', '_')
                        if attr_name in hf.keys():
                            hf.__delitem__(attr_name)
                        hf.create_dataset(attr_name, data = np.array(val, dtype=np.float32))
                    except:
                        pass
                for i in range(self.num_ref):
                    if f'ref{i}' in hf.keys():
                        hf.__delitem__(f'ref{i}')
                    hf.create_dataset(f'ref{i}', data=self.spectrum_ref[f'ref{i}'])
            '''
            msg = f'xanes_fit has been saved to file: "{fn}" and append to ".../{self.fn_raw_image.split("/")[-1]}"'
            msg = textwrap.fill(msg, 100)
            print(msg)
            self.msg = msg
        except:
            self.msg = 'file saving fails ...'
        finally:
            self.update_msg()

    def remove_bkg(self):
        '''
        Treat if it is fluorescent image.
        calculate the mean value of 5% of the lowest pixel value, and substract this value from the whole image
        '''
        self.pb_rmbg.setText('Remove backgroud ...')
        self.pb_rmbg.setEnabled(False)
        QApplication.processEvents()

        canvas = self.canvas1
        img_stack = deepcopy(canvas.img_stack)

        roi_list = canvas.roi_list
        try:
            roi_spec = 0
            num = len(self.lst_roi.selectedItems())
            for item in self.lst_roi.selectedItems():
                print(item.text())
                roi_cord = np.int32(roi_list[item.text()][:4])
                a, b, c, d = roi_cord[0], roi_cord[1], roi_cord[2], roi_cord[3]
                x1 = min(a, c)
                x2 = max(a, c)
                y1 = min(b, d)
                y2 = max(b, d)
                roi_spec += np.mean(np.mean(img_stack[:, y1:y2, x1:x2, ], axis=1), axis=1)
            roi_spec = roi_spec / num
            for i in range(img_stack.shape[0]):
                img_stack[i] -= roi_spec[i]
            self.img_update = deepcopy(img_stack)
            del img_stack
            self.msg = 'Background removed '
        except:
            self.msg = 'fails in remove background using ROI, check ROI selection'
        finally:
            self.update_msg()
            self.pb_rmbg.setEnabled(True)
            self.pb_rmbg.setText('Remove Bkg. (ROI) ')
            QApplication.processEvents()
            if self.cb1.findText('Image updated') < 0:
                self.cb1.addItem('Image updated')
            self.cb1.setCurrentText('Image updated')
            self.update_canvas_img()

    def xanes_img_smooth(self):
        self.pb_filt.setEnabled(False)
        self.pb_filt.setText('Smoothing ...')
        QApplication.processEvents()
        self.smooth_param['kernal_size'] = int(self.tx_filt.text())
        self.smooth_param['flag'] = 1
        self.update_canvas_img()
        self.pb_filt.setEnabled(True)
        self.pb_filt.setText('Median filter')
        self.msg = 'Image smoothed'
        self.update_msg()
        QApplication.processEvents()

    def smooth(self, img, axis=0):
        img_stack = deepcopy(img)
        if self.smooth_param['flag'] == 1:
            try:
                kernal_size = self.smooth_param['kernal_size']
                if kernal_size > 1:
                    img_stack = img_smooth(img_stack, kernal_size, axis=axis)
            except:
                self.msg = 'image smoothing fails...'
                self. update_msg()
            finally:
                self.smooth_param['flag'] = 0
        return img_stack

    def get_roi_mask(self, roi_list, roi_item, roi_shape):
        if 'SM' in roi_item.text():
            sm_index = int(roi_item.text().split('_')[-1])
            mask_roi = self.smart_mask[sm_index]
            mask_type = 'SM'
        else:
            if len(roi_shape) == 3:
                mask_roi = np.zeros([roi_shape[1], roi_shape[2]])
            else:
                mask_roi = np.zeros([roi_shape[0], roi_shape[1]])
            roi_cord = np.int32(np.array(roi_list[roi_item.text()][:4]))
            a, b, c, d = roi_cord[0], roi_cord[1], roi_cord[2], roi_cord[3]
            x1 = min(a, c)
            x2 = max(a, c)
            y1 = min(b, d)
            y2 = max(b, d)
            mask_roi[y1:y2, x1:x2] = 1
            mask_type = 'ROI'
        return mask_roi, mask_type

    def plot_spectrum(self):
        canvas = self.canvas1
        img_stack = deepcopy(canvas.img_stack)
        roi_shape = img_stack.shape
        plt.figure()
        roi_color = canvas.roi_color
        roi_list = canvas.roi_list
        x = self.xanes_eng
        if len(x) != len(img_stack):
            x = np.arange(len(img_stack))
        legend = []
        try:
            if len(self.lst_roi.selectedItems()):
                for item in self.lst_roi.selectedItems():
                    print(item.text())
                    plot_label = item.text()
                    mask_roi, mask_type = self.get_roi_mask(roi_list, item, roi_shape)
                    roi_spec = np.sum(np.sum(img_stack * mask_roi, axis=1), axis=1) / np.sum(mask_roi)
                    plot_color = roi_color[item.text()]
                    line, = plt.plot(x, roi_spec, marker='.', color=plot_color, label=plot_label)
                    legend.append(line)
            else:
                mask = self.mask
                if np.squeeze(mask).shape != img_stack[0].shape:
                    mask = np.ones(img_stack[0].shape)
                roi_spec = np.sum(np.sum(img_stack * mask, axis=1), axis=1) / np.sum(mask)
                line, = plt.plot(x, roi_spec, marker='.', color='r', label='image average')
                legend.append(line)
            print(legend)
            plt.legend(handles=legend)
            plt.show()
        except:
            self.msg = 'no spectrum available for current image stack ...'
            self.update_msg()

    def evaluate_glitch(self):
            canvas = self.canvas1
            img_stack = deepcopy(canvas.img_stack)
            roi_list = canvas.roi_list
            x = self.xanes_eng
            item = self.lst_roi.selectedItems()[0]
            roi_cord = np.int32(np.array(roi_list[item.text()][:4]))
            a, b, c, d = roi_cord[0], roi_cord[1], roi_cord[2], roi_cord[3]
            x1 = min(a, c)
            x2 = max(a, c)
            y1 = min(b, d)
            y2 = max(b, d)
            roi_spec = np.mean(np.mean(img_stack[:, y1:y2, x1:x2], axis=1), axis=1)
            roi_spec_median = medfilt(roi_spec, 5)
            self.roi_spec_dif = roi_spec/roi_spec_median
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(x[:-1], self.roi_spec_dif[:-1], 'r.')
            plt.title('Identify the threshold for removing glitch')
            plt.subplots_adjust(hspace = 0.5)
            plt.subplot(2, 1, 2)
            plt.plot(x, roi_spec, 'b.-')
            plt.plot(x[:-1], roi_spec_median[:-1], 'r.-')
            plt.title(f'median value: {np.median(self.roi_spec_dif)}')
            plt.show()
            self.msg = 'no spectrum available for current image stack ...'
            self.update_msg()

    def remove_glitch(self):
        tmp = self.tx_glitch_thresh.text()           
        glitch_thresh =  self.tx_glitch_thresh.text()
        if len(self.roi_spec_dif) != len(self.xanes_eng):
            self.msg = 'data size disagree...'
            self.update_msg()
        else:
            if tmp[0] == '<':
                mask = self.roi_spec_dif > np.float(glitch_thresh[1:])
            elif tmp[0] == '>':
                mask = self.roi_spec_dif < np.float(glitch_thresh[1:])
            else:
                mask = self.roi_spec_dif < np.float(glitch_thresh)
            msg = QMessageBox()
            msg.setText('This will delete the image in \"raw data\", and \"image_update\" ')
            msg.setWindowTitle('Delete multiple image')
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            reply = msg.exec_()
            if reply == QMessageBox.Ok:
                try:
                    self.img_xanes = self.img_xanes[mask]
                    print('glitch removed in img_xanes ')
                    self.img_update = self.img_update[mask]
                    print('glitch removed in img_update ')
                    self.xanes_eng = self.xanes_eng[mask]
                    st = '{0:3.1f}, {1:3.1f}, ..., {2:3.1f}  (totally, {3} angles)'.format(self.xanes_eng[0],self.xanes_eng[1],self.xanes_eng[-1],len(self.xanes_eng))
                    self.lb_eng1.setText(st) 
                    print('glitch removed in xanes_eng ')
                    self.msg = 'glitch removed !'
                except:
                    print('cannot delete xanes_eng')
                    self.msg = 'removing glitch failed ...'
                finally:
                    self.update_msg()
            self.update_canvas_img()
            QApplication.processEvents()

    def show_roi(self):
        plt.figure()
        canvas = self.canvas1
        canvas.show_roi_flag = True
        current_img = canvas.current_img
        current_colormap = canvas.colormap
        # cmin, cmax = canvas.cmin, canvas.cmax
        self.update_canvas_img()
        # canvas.update_img_stack()
        mask = canvas.mask
        s = current_img.shape
        type_index = self.cb1.currentText()
        cmax = np.float(self.tx_cmax.text())
        cmin = np.float(self.tx_cmin.text())
        if type_index == 'Color mix':
            current_img = self.img_colormix
            for i in range(current_img.shape[2]):
                current_img[:, :, i] *= canvas.rgb_mask
            current_img = (current_img - cmin) / (cmax - cmin)
            plt.imshow(current_img)
        else:
            plt.imshow(current_img * mask, cmap=current_colormap, vmin=cmin, vmax=cmax)

        if len(mask.shape) > 1:
            print(f'canvas_mask.shape = {canvas.mask.shape}')
            canvas.special_info = None
        plt.axis('equal')
        plt.axis('off')
        plt.colorbar()
        roi_color = canvas.roi_color
        roi_list = canvas.roi_list
        for item in self.lst_roi.selectedItems():
            mask_roi, mask_type = self.get_roi_mask(roi_list, item, s)
            if mask_type == 'ROI':
                print(item.text())
                plot_color = roi_color[item.text()]
                roi_cord = np.int32(np.array(roi_list[item.text()][:4]))
                plot_label = item.text()
                a, b, c, d = roi_cord[0], roi_cord[1], roi_cord[2], roi_cord[3]
                x1 = min(a, c)
                x2 = max(a, c)
                y1 = min(b, d)
                y2 = max(b, d)
                x = [x1, x2, x2, x1, x1]
                y = [y1, y1, y2, y2, y1]
                plt.plot(x, y, '-', color=plot_color, linewidth=1.0, label=plot_label)
                roi_name = '#' + plot_label.split('_')[-1]
                plt.annotate(roi_name, xy=(x1, y1 - 40),
                             bbox={'facecolor': plot_color, 'alpha': 0.5, 'pad': 2},
                             fontsize=10)
            else: # mask_type == 'SM':
                plt.figure()
                plt.imshow(mask_roi)
        plt.show()

    def hide_roi(self):
        canvas = self.canvas1
        canvas.show_roi_flag = False
        self.update_canvas_img()

    def export_spectrum(self):
        self.show_roi()
        # self.tx_file_index = int(self.tx_file_index.text())
        '''
        try:
            os.makedirs(self.fpath + '/ROI')
            make_director_success = True
        except:
            if os.path.exists(self.fpath + '/ROI'):
                make_director_success = True
            else:
                print(self.fpath + '/ROI failed')
                make_director_success = False
                self.msg = 'Access to directory: "' + self.path + '/ROI' + '" fails'
                self.update_msg()
        '''
        make_director_success = 1 # temporay to enable saving anyway
        if make_director_success:
            canvas = self.canvas1
            img_stack = deepcopy(canvas.img_stack)
            s = img_stack.shape
            x = self.xanes_eng
            roi_list = canvas.roi_list
            roi_dict_spec = {'X_eng': pd.Series(x)}
            # roi_dict_cord = {}
            if len(roi_list):
                for item in self.lst_roi.selectedItems():
                    plot_label = item.text()
                    mask_roi, mask_type = self.get_roi_mask(roi_list, item, s)
                    roi_spec = np.sum(np.sum(img_stack * mask_roi, axis=1), axis=1) / np.sum(mask_roi)
                    roi_spec = np.around(roi_spec, 9)
                    roi_dict_spec[plot_label] = pd.Series(roi_spec)
            else:
                mask = self.mask
                if np.squeeze(mask).shape != img_stack[0].shape:
                    mask = np.ones(img_stack[0].shape)
                roi_spec = np.sum(np.sum(img_stack * mask, axis=1), axis=1) / np.sum(mask)
                roi_dict_spec = {'-1':pd.Series(roi_spec)}

            df_spec = pd.DataFrame(roi_dict_spec)
            options = QFileDialog.Option()
            options |= QFileDialog.DontUseNativeDialog
            file_type = 'txt files (*.txt)'
            fn, _ = QFileDialog.getSaveFileName(self, 'Save Spectrum', "", file_type, options=options)
            if fn[-4:] == '.txt':
                fn = fn[:-4]
            fn_spec = f'{fn}_spec.txt'
            with open(fn_spec, 'w') as f:
                df_spec.to_csv(f, float_format='%.9f', sep=' ', index=False)
            self.roi_file_id += 1
            self.tx_file_index.setText(str(self.roi_file_id))
            print(fn_spec + '  saved')
            self.msg = 'ROI spectrum file saved:   ' + fn_spec
            self.update_msg()

    def reset_roi(self):
        canvas = self.canvas1
        self.lst_roi.clear()
        canvas.current_roi = [0, 0, 0, 0, '0']
        canvas.show_roi_flag = False
        canvas.current_color = 'red'
        canvas.roi_list = {}
        canvas.roi_count = 0
        canvas.roi_color = {}
        self.update_canvas_img()
        s = canvas.current_img.shape
        self.tx_roi_x1.setText('0.0')
        self.tx_roi_x2.setText('{:3.1f}'.format(s[1]))
        self.tx_roi_y1.setText('{:3.1f}'.format(0))
        self.tx_roi_y2.setText('{:3.1f}'.format(s[0]))

    def draw_roi(self):
        self.pb_roi_draw.setEnabled(False)
        QApplication.processEvents()
        canvas = self.canvas1
        canvas.draw_roi()

    def load_external_spec(self):
        options = QFileDialog.Option()
        options |= QFileDialog.DontUseNativeDialog
        file_type = 'txt files (*.txt)'
        fn, _ = QFileDialog.getOpenFileName(xanes, "QFileDialog.getOpenFileName()", "", file_type, options=options)
        if fn:
            try:
                print(fn)
                fn_spec = fn.split('/')[-1]
                self.external_spec = np.loadtxt(fn)
                self.msg = f'loading existing spectrum: {fn_spec}'
                self.update_msg()
                QApplication.processEvents()
                plt.figure()
                plt.plot(self.external_spec[:,0], self.external_spec[:,1])
                plt.title('External xanes spectrum')
                plt.show()
            except:
                print('un-supported spectrum format')

    def norm_external_spec(self):
        try:
            pre_s = float(self.tx_fit_pre_s.text())
            pre_e = float(self.tx_fit_pre_e.text())
            post_s = float(self.tx_fit_post_s.text())
            post_e = float(self.tx_fit_post_e.text())
            x_eng = self.external_spec[:,0]
            self.external_spec_fit = self.external_spec.copy()
            self.external_spec_fit[:, 1], y_pre_fit, y_post_fit = normalize_1D_xanes(self.external_spec[:,1], x_eng, [pre_s, pre_e],
                                                                       [post_s, post_e])
            plt.figure()  # generate figure for each roi
            plt.subplot(211)
            plt.subplots_adjust(hspace=0.5)
            plt.plot(x_eng, self.external_spec[:,1], '.', color='gray')
            plt.plot(x_eng, y_pre_fit, 'b', linewidth=1)
            plt.plot(x_eng, y_post_fit + y_pre_fit, 'r', linewidth=1)
            plt.title('pre-post edge fitting')
            plt.subplot(212)  # plot normalized spectrum
            plt.plot(x_eng, self.external_spec_fit[:,1])
            plt.title('normalized spectrum')
            plt.show()
        except:
            self.msg = 'faild to fit external spectrum'
            self.update_msg()

    def save_external_spec(self):
        try:
            options = QFileDialog.Option()
            options |= QFileDialog.DontUseNativeDialog
            file_type = 'txt files (*.txt)'
            fn, _ = QFileDialog.getSaveFileName(self, 'Save File', "", file_type, options=options)
            if fn:
                if not fn[-4:] == 'txt':
                    fn += '.txt'
                np.savetxt(fn, np.array(self.external_spec_fit), '%2.5f')
                print(fn + '  saved')
                self.msg = f'{fn} is saved'
        except:
            self.msg = 'fails to save normed external spectrum'
        finally:
            self.update_msg()

    def fit_edge(self):
        try:
            pre_s = float(self.tx_fit_pre_s.text())
            pre_e = float(self.tx_fit_pre_e.text())
            post_s = float(self.tx_fit_post_s.text())
            post_e = float(self.tx_fit_post_e.text())

            canvas = self.canvas1
            img_stack = deepcopy(canvas.img_stack)
            roi_list = canvas.roi_list
            x_eng = deepcopy(self.xanes_eng)        
            num_roi_sel = len(self.lst_roi.selectedItems())
            roi_spec_fit = np.zeros([len(x_eng), num_roi_sel+1])
            roi_spec_fit[:,0] = x_eng
            n = 0
            for item in self.lst_roi.selectedItems():
                n = n + 1
                plt.figure()  # generate figure for each roi
                plt.subplot(211)
                roi_cord = np.int32(np.array(roi_list[item.text()][:4]))
                plot_label = item.text()
                a, b, c, d = roi_cord[0], roi_cord[1], roi_cord[2], roi_cord[3]
                x1 = min(a, c)
                x2 = max(a, c)
                y1 = min(b, d)
                y2 = max(b, d)
                roi_spec = np.mean(np.mean(img_stack[:, y1:y2, x1:x2, ], axis=1), axis=1)
                roi_spec_fit[:,n], y_pre_fit, y_post_fit = normalize_1D_xanes(roi_spec, x_eng, [pre_s, pre_e], [post_s, post_e])
                plt.subplots_adjust(hspace=0.5)
                plt.plot(x_eng, roi_spec, '.', color='gray')
                plt.plot(x_eng, y_pre_fit, 'b', linewidth=1)
                plt.plot(x_eng, y_post_fit + y_pre_fit, 'r', linewidth=1)
                plt.title('pre-post edge fitting for ' + plot_label)
                plt.subplot(212)  # plot normalized spectrum
                plt.plot(x_eng, roi_spec_fit[:, n])
                plt.title('normalized spectrum for ' + plot_label)
                plt.show()    
            self.roi_spec = roi_spec_fit
        except:
            self.msg = 'Fitting error ...'
            self.update_msg()

    def save_normed_roi(self):
        try:
            os.makedirs(self.fpath + '/ROI/fitted_roi')
        except:
            print(self.fpath + '/ROI failed')
            pass
        try:
            fn_spec = 'fitted_spectrum_roi_from_' + self.cb1.currentText() + '_' + self.tx_file_index.text() + '.txt'
            fn_spec = self.fpath + '/ROI/fitted_roi/' + fn_spec
            # fn_cord = 'fitted_coordinates_roi_from_' + self.cb1.currentText() + '_' + self.tx_file_index.text() + '.txt'
            # fn_cord = self.fpath + '/ROI/fitted_roi/' + fn_cord
            canvas = self.canvas1            
            roi_spec = deepcopy(self.roi_spec)
            roi_spec = np.around(roi_spec, 4)
            roi_list = canvas.roi_list
            roi_dict_spec = {'X_eng': pd.Series(roi_spec[:, 0])}
            roi_dict_cord = {}
            n = 0
            for item in self.lst_roi.selectedItems():
                n = n + 1
                plot_label = item.text()
                roi_cord = np.int32(np.array(roi_list[item.text()][:4]))
                a, b, c, d = roi_cord[0], roi_cord[1], roi_cord[2], roi_cord[3]
                x1 = min(a, c)
                x2 = max(a, c)
                y1 = min(b, d)
                y2 = max(b, d)
                area = (x2 - x1) * (y2 - y1)            
                roi_dict_spec[plot_label] = pd.Series(roi_spec[:, n])
                roi_dict_cord[plot_label] = pd.Series([x1, y1, x2, y2, area], index=['x1', 'y1', 'x2', 'y2', 'area'])
            df_spec = pd.DataFrame(roi_dict_spec)
            df_cord = pd.DataFrame(roi_dict_cord)

            options = QFileDialog.Option()
            options |= QFileDialog.DontUseNativeDialog
            file_type = 'txt files (*.txt)'
            fn, _ = QFileDialog.getSaveFileName(self, 'Save File', "", file_type, options=options)
            if fn:
                if not fn[-4:] == '.txt':
                    fn += '.txt'
                np.savetxt(fn, np.array(df_spec), '%2.5f')
                # fn_cord_another = fn+f'_fitted_coordinates_roi_from_{self.cb1.currentText()}_{self.tx_file_index.text()}.txt'
                with open(fn_spec, 'w') as f:
                    df_cord.to_csv(f, sep=' ')
                print(fn_spec + '  saved')
                self.msg = 'Fitted ROI spectrum saved:   ' + fn
        except:
            self.msg = 'Save fitted roi spectrum fails ...'
        finally:
            self.update_msg()

    def fit_edge_img(self):
        pre_s = float(self.tx_fit_pre_s.text())
        pre_e = float(self.tx_fit_pre_e.text())
        post_s = float(self.tx_fit_post_s.text())
        post_e = float(self.tx_fit_post_e.text())
        canvas = self.canvas1
        try:
            self.pb_fit_img.setText('wait ...')
            self.pb_fit_img.setEnabled(False)
            QApplication.processEvents()
            img_norm = deepcopy(canvas.img_stack) * self.mask
            x_eng = deepcopy(self.xanes_eng)
            pre_edge_only_flag = 1 if self.chkbox_norm_pre_edge_only.isChecked() else 0
            if self.rd_norm1.isChecked():
                img_norm, self.img_pre_edge_sub_mean = normalize_2D_xanes2(img_norm, x_eng, [pre_s, pre_e], [post_s, post_e], pre_edge_only_flag)
                self.msg = '2D Spectra image normalized (using method1)'
            elif self.rd_norm2.isChecked():
                img_norm, tmp = normalize_2D_xanes_old(img_norm, x_eng, [pre_s, pre_e], [post_s, post_e], pre_edge_only_flag)
                # if len(self.img_pre_edge_sub_mean.shape) == 1:
                #    self.img_pre_edge_sub_mean = tmp
                self.img_pre_edge_sub_mean = tmp
                self.msg = '2D Spectra image normalized (using method2)'
            self.data_summary['XANES Fit (thickness)'] = self.smooth(self.img_pre_edge_sub_mean)
            self.img_update = deepcopy(img_norm)
            self.cb1.setCurrentText('Image updated')
            self.update_canvas_img()
            self.pb_fit_img.setText('Norm Img.')
            self.pb_fit_img.setEnabled(True)
            if self.cb1.findText('Image updated') < 0:
                self.cb1.addItem('Image updated')
            if self.cb1.findText('XANES Fit (thickness)') < 0:
                self.cb1.addItem('XANES Fit (thickness)')
            self.cb1.setCurrentText('Image updated')
            QApplication.processEvents()
            del img_norm
        except:
            self.msg = 'fails to normalize 2D spectra image'
        finally:
            self.pb_fit_img.setText('Norm Image')
            self.pb_fit_img.setEnabled(True)
            QApplication.processEvents()
            self.update_msg()

    def regular_edge_img(self):
        try:
            canvas = self.canvas1
            img_norm = deepcopy(canvas.img_stack)
            pre_s = float(self.tx_fit_pre_s.text())
            pre_e = float(self.tx_fit_pre_e.text())
            post_s = float(self.tx_fit_post_s.text())
            post_e = float(self.tx_fit_post_e.text())
            x_eng = deepcopy(self.xanes_eng)
            regular_max = float(self.tx_reg_max.text())
            regular_width = float(self.tx_reg_width.text())
            self.pb_reg_img.setText('wait ...')
            self.pb_reg_img.setEnabled(False)
            QApplication.processEvents()
            self.img_regulation = normalize_2D_xanes_regulation(img_norm, x_eng, pre_edge=[pre_s, pre_e], post_edge=[post_s, post_e], designed_max=regular_max, gamma=regular_width)
            self.msg = 'Image regulation finished'
            if self.cb1.findText('Image regulation') < 0:
                self.cb1.addItem('Image regulation')
            self.cb1.setCurrentText('Image regulation')
            QApplication.processEvents()
        except:
            self.msg = 'fails to regularize 2D spectra image'
        finally:
            self.pb_reg_img.setText('Regulation')
            self.pb_reg_img.setEnabled(True)
            QApplication.processEvents()
            self.update_msg()

    def save_img_stack(self):
        try:
            self.pb_save_img_stack.setEnabled(False)
            QApplication.processEvents()
            canvas = self.canvas1
            if self.cb1.currentText() == 'Color mix':
                self.save_img_single()
            else:
                img_stack = (canvas.img_stack* self.mask1 * self.mask2).astype(np.float32)
                options = QFileDialog.Option()
                options |= QFileDialog.DontUseNativeDialog
                file_type = 'tif files (*.tiff)'
                fn, _ = QFileDialog.getSaveFileName(self, 'Save File', "", file_type, options=options)
                if fn[-5:] != '.tiff' or fn[-4:]!='.tif':
                    fn += '.tiff'
                io.imsave(fn, img_stack)
                print(f'current image stack has been saved to file: {fn}')
                self.msg = f'image stack saved to: {fn}'
        except:
            self.msg = 'file saving fails ...'
        finally:            
            self.update_msg()
            self.pb_save_img_stack.setEnabled(True)
            QApplication.processEvents()

    def save_img_single(self):
        try:
            self.pb_save_img_single.setEnabled(False)
            QApplication.processEvents()
            canvas = self.canvas1
            cmax = np.float(self.tx_cmax.text())
            cmin = np.float(self.tx_cmin.text())
            if self.cb1.currentText() == 'Color mix':
                img = self.img_colormix
                for i in range(img.shape[2]):
                    img[:, :, i] = img[:, :, i] * canvas.rgb_mask
                plt.figure()
                img = (img - cmin) / (cmax - cmin)
                plt.imshow(img, clim=[cmin, cmax])
                plt.show()
            else:
                img_stack = canvas.img_stack[canvas.current_img_index] * canvas.mask
                img_stack = np.array(img_stack, dtype = np.float32)
                options = QFileDialog.Option()
                options |= QFileDialog.DontUseNativeDialog
                file_type = 'tif files (*.tiff)'
                fn, _ = QFileDialog.getSaveFileName(self, 'Save File', "", file_type, options=options)
                if not(fn[-5:] == '.tiff' or fn[-4:] =='.tif'):
                    fn += '.tiff'
                io.imsave(fn, img_stack)
                print(f'current image has been saved to file: {fn}')
                self.msg = f'current image saved to: {fn}'
                plt.figure()
                plt.imshow(img_stack, clim=[cmin, cmax], cmap=canvas.colormap)
                plt.axis('off')
                plt.show()
        except:
            self.msg = 'file saving fails ...'
        finally:
            self.update_msg()
            self.pb_save_img_single.setEnabled(True)
            QApplication.processEvents()

    def delete_single_img(self):
        canvas = self.canvas1
        if canvas.img_stack.shape[0] < 2:
            self.msg = 'cannot delete image in single-image-stack'
            self.update_msg()
        else:
            msg = QMessageBox()
            msg.setText('This will delete the image in \"raw data\", and \"image_update\" ')
            msg.setWindowTitle('Delete single image')
            msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            reply = msg.exec_()
            if reply == QMessageBox.Ok:
                #img_type = self.cb1.currentText()
                current_slice = self.sl1.value()
                try:
                    self.img_xanes = np.delete(self.img_xanes, current_slice, axis=0)
                except:
                    print('cannot delete img_xanes')
                try:
                    self.img_update = np.delete(self.img_update, current_slice, axis=0)
                except:
                    print('cannot delete img_update')
                try:
                    self.xanes_eng = np.delete(self.xanes_eng, current_slice, axis=0)
                    st = '{0:3.1f}, {1:3.1f}, ..., {2:3.1f}  (totally, {3} energies)'.format(self.xanes_eng[0],self.xanes_eng[1],self.xanes_eng[-1],len(self.xanes_eng))
                    self.lb_eng1.setText(st)  # update angle information showing on the label
                except:
                    print('cannot delete energy')
                self.msg = 'image #{} has been deleted'.format(current_slice)
                self.update_msg()
                self.update_canvas_img()

    def load_energy(self):
        options = QFileDialog.Option()
        options |= QFileDialog.DontUseNativeDialog
        file_txt_flag = 0
        if len(self.tx_hdf_eng.text()):
            dataset_eng = self.tx_hdf_eng.text()
        else:
            dataset_eng = 'X_eng'
        if self.rd_hdf_eng.isChecked():
            file_type = 'hdf files (*.h5)'
            file_txt_flag = 0
        else:                                           # self.rd_txt_eng.isChecked() == True:
            file_type = 'txt files (*.txt)'
            file_txt_flag = 1
        fn, _ = QFileDialog.getOpenFileName(xanes, "QFileDialog.getOpenFileName()", "", file_type, options=options)
        if fn:
            try:
                print(fn)
                if file_txt_flag:
                    self.xanes_eng = np.array(np.loadtxt(fn))
                else:
                    f = h5py.File(fn, 'r')
                    self.xanes_eng = np.array(f[dataset_eng])
                    f.close()
                st = '{0:2.4f}, {1:2.4f}, ..., {2:2.4f} keV   totally, {3} energies'.format(self.xanes_eng[0], self.xanes_eng[1], self.xanes_eng[-1], len(self.xanes_eng))
                self.tx_fit_pre_s.setText('{:2.4f}'.format(min(self.xanes_eng)))
                self.tx_fit_pre_e.setText('{:2.4f}'.format(min(self.xanes_eng) + 0.010))
                self.tx_fit_post_e.setText('{:2.4f}'.format(max(self.xanes_eng)))
                self.tx_fit_post_s.setText('{:2.4f}'.format(max(self.xanes_eng) - 0.010))
                self.lb_eng1.setText(st)
                self.tx_fit2d_s.setText('{:2.4f}'.format(self.xanes_eng[0]))
                self.tx_fit2d_e.setText('{:2.4f}'.format(self.xanes_eng[-1]))
            except:
                self.xanes_eng = np.array([0])
                self.lb_eng1.setText('No energy data ...')
                self.lb_eng2.setVisible(True)
                self.tx_eng.setVisible(True)
                self.pb_eng.setVisible(True)
                self.msg = self.msg + ';  Energy list not exist'
            self.update_msg()
            self.update_canvas_img()

    def load_fitted_file_sub(self, fn):
        self.dataset_used_for_fitting = -1
        self.xanes_2d_fit_offset = 0
        self.num_ref = 0
        thickness_flag = 0
        concentration_flag = 0
        f = h5py.File(fn, 'r')
        keys = list(f.keys())
        # edge
        post_edge = np.array(f['post_edge'])
        pre_edge = np.array(f['pre_edge'])
        self.tx_fit_pre_s.setText(str(pre_edge[0]))
        self.tx_fit_pre_e.setText(str(pre_edge[1]))
        self.tx_fit_post_s.setText(str(post_edge[0]))
        self.tx_fit_post_e.setText(str(post_edge[1]))
        self.msg = f"reading: {', '.join(t for t in keys)}"
        for k in keys:
            if k.lower() == 'mask':
                self.mask = np.squeeze(np.array(f[k]))
                self.canvas1.mask = self.mask
                self.mask1 = self.mask
                self.pb_mask1.setStyleSheet('color: rgb(200, 50, 50);')
                self.cb1.addItem('Mask')
                continue
            if 'smart' in k.lower():
                self.smart_mask = np.array(f[k])
                self.cb1.addItem('Smart Mask')
                continue
            if 'eng' in k.lower():
                self.xanes_eng = np.array(f[k])
                st = '{0:2.4f}, {1:2.4f}, ..., {2:2.4f} keV   totally, {3} energies'.format(self.xanes_eng[0], self.xanes_eng[1], self.xanes_eng[-1], len(self.xanes_eng))
                self.lb_eng1.setText(st)
            if 'ratio' in k.lower() and 'sum' in k.lower():
                self.xanes_2d_fit = np.array(f[k])
                self.cb1.addItem('XANES Fit (ratio, summed to 1)')
                continue
            if 'thickness' in k.lower():
                self.img_pre_edge_sub_mean = np.array(f[k])
                self.cb1.addItem('XANES Fit (thickness)')
                concentration_flag = 1
                thickness_flag = 1
                continue
            if 'error' in k.lower():
                self.xanes_fit_cost = np.array(f[k])
                self.cb1.addItem('XANES Fit error')
                continue
            if 'ref' in k.lower():
                self.num_ref += 1
                continue
            if 'offset' in k.lower():
                self.xanes_2d_fit_offset = np.array(f[k])
                self.cb1.addItem('XANES Fit offset')
                continue
        if thickness_flag == 0:
            for k in keys:
                if 'concentration' in k.lower():
                    self.img_pre_edge_sub_mean = self.xanes_2d_fit / np.array(f[k])
                    concentration_flag = 1
                    break
        if concentration_flag:
            self.cb1.addItem('XANES Fit (Elem. concentration)')
        for i in range(self.num_ref):
            try:
                ref_name = f'ref{i}'
                self.spectrum_ref[ref_name] = np.float32(np.array(f[ref_name]))
                self.lb_ref_info.setText(self.lb_ref_info.text() + '\n' + ref_name + '  loaded')
                self.lb_ref_info.setStyleSheet('color: rgb(200, 50, 50);')
                self.cb_color_channel.addItem(str(i))
                QApplication.processEvents()
            except:
                self.num_ref -= 1
        f.close()
        return 1


    def load_fitted_file(self):
        self.default_layout()
        options = QFileDialog.Option()
        options |= QFileDialog.DontUseNativeDialog
        file_type = 'hdf files (*.h5)'
        fn, _ = QFileDialog.getOpenFileName(xanes, "QFileDialog.getOpenFileName()", "", file_type, options=options)
        if fn:
            load_successful = self.load_fitted_file_sub(fn)
        self.lb_ip.setText('File loaded:   {}'.format(fn))
        self.pb_plot_roi.setEnabled(True)
        self.pb_export_roi_fit.setEnabled(True)



    def open_imagej(self):
        try:
            os.system('imagej &')
        except:
            self.msg = 'can not find/open imagej'
            self.update_msg()

    def close_all_figures(self):
        plt.close('all')

    def load_image(self):
        self.default_layout()
        self.pb_ld.setEnabled(False)
        if len(self.tx_hdf_xanes.text()):
            dataset_xanes = self.tx_hdf_xanes.text()
        else:
            dataset_xanes = 'img_xanes'

        if len(self.tx_hdf_eng.text()):
            dataset_eng = self.tx_hdf_eng.text()
        else:
            dataset_eng = 'X_eng'
        options = QFileDialog.Option()
        options |= QFileDialog.DontUseNativeDialog
        if self.rd_hdf.isChecked():
            file_type = 'hdf files (*.h5)'
            flag_read_from_file = 1
            flag_read_from_database =0
        elif self.rd_tif.isChecked():
            print('read tiff or tif file')
            file_type = 'tif files (*.tif, *.tiff)'
            flag_read_from_file = 1
            flag_read_from_database = 0
        else:
            flag_read_from_database = 1
            flag_read_from_file = 0
        if flag_read_from_file:
            fn, _ = QFileDialog.getOpenFileName(xanes, "QFileDialog.getOpenFileName()", "", file_type, options=options)
            if fn:
                print(fn)
                self.fn_raw_image = fn
                fn_relative = fn.split('/')[-1]
                self.fpath = fn[:-len(fn_relative)-1]
                print(f'current path: {self.fpath}')

                self.lb_ip.setStyleSheet('color: rgb(200, 50, 50);')
                if self.rd_hdf.isChecked():  # read hdf file
                    f = h5py.File(fn, 'r')
                    # read energy
                    try:
                        self.xanes_eng = np.array(f[dataset_eng])  # Generally, it is in unit of keV
                        # if min(self.xanes_eng) < 4000:  # make sure it is in unit of KeV
                        #     self.xanes_eng = self.xanes_eng * 1000
                        st = '{0:2.4f}, {1:2.4f}, ..., {2:2.4f} keV (totally, {3} energies)'.format(self.xanes_eng[0], self.xanes_eng[1], self.xanes_eng[-1], len(self.xanes_eng))
                        self.tx_fit_pre_s.setText('{:2.4f}'.format(min(self.xanes_eng) - 0.001))
                        self.tx_fit_pre_e.setText('{:2.4f}'.format(min(self.xanes_eng) + 0.010))
                        self.tx_fit_post_e.setText('{:2.4f}'.format(max(self.xanes_eng) + 0.001))
                        self.tx_fit_post_s.setText('{:2.4f}'.format(max(self.xanes_eng) - 0.010))
                        self.lb_eng1.setText(st)
                        self.tx_fit2d_s.setText('{:2.4f}'.format(self.xanes_eng[0]))
                        self.tx_fit2d_e.setText('{:2.4f}'.format(self.xanes_eng[-1]))
                    except:
                        self.xanes_eng = np.array([0])
                        self.lb_eng1.setText('No energy data ...')
                        self.lb_eng2.setVisible(True)
                        self.tx_eng.setVisible(True)
                        self.pb_eng.setVisible(True)
                        self.msg = self.msg + ';  Energy list not exist'
                        self.update_msg()
                    # read xanes-scan image
                    try:
                        self.img_xanes = rm_abnormal(np.array(f[dataset_xanes]))
                        s = self.img_xanes.shape
                        if len(s) == 2:
                            self.img_xanes = np.expand_dims(self.img_xanes, axis=0)
                        self.img_update = deepcopy(self.img_xanes)
                        print('Image size: ' + str(self.img_xanes.shape))
                        self.pb_norm_txm.setEnabled(True)
                        self.pb_align.setEnabled(True)
                        self.pb_del.setEnabled(True)  # delete single image
                        self.pb_filt.setEnabled(True)  # delete single image
                        self.pb_rmbg.setEnabled(True)
                        self.pb_align_roi.setEnabled(True)
                        self.msg = 'image shape: {0}'.format(self.img_xanes.shape)
                        self.lb_ip.setText('File loaded:   {}'.format(fn))
                        if self.cb1.findText('Raw image') < 0:
                            self.cb1.addItem('Raw image')
                        self.cb1.setCurrentText('Raw image')
                        print(f'num of eng: {len(self.xanes_eng)}   image_shape: {self.img_xanes.shape}')
                        if (len(self.xanes_eng) != self.img_xanes.shape[0]):
                            self.msg = 'number of energy does not match number of images, try manual input ...'
                    except:
                        self.img_xanes = np.zeros([1, 100, 100])
                        print('xanes image not exist')
                        self.lb_ip.setText('File loading fails ...')
                        self.msg = 'xanes image not exist'
                    load_fitted = 0
                    '''
                    if 'fitted' in list(f.keys()):
                        try:
                            load_fitted = self.load_fitted_file_sub(fn)
                            if load_fitted:
                                self.msg = textwrap.fill(self.msg, 100) + '\nfitted data is loaded'
                        except:
                            pass
                    '''
                    self.update_canvas_img()
                    self.update_msg()
                    f.close()
                else:  # read tiff file
                    try:
                        self.img_xanes = rm_abnormal(np.array(io.imread(fn)))
                        s = self.img_xanes.shape
                        if len(s) == 2:
                            self.img_xanes = np.expand_dims(self.img_xanes, axis=0)
                        self.msg = 'image shape: {0}'.format(self.img_xanes.shape)
                        self.update_msg()
                        self.pb_norm_txm.setEnabled(True)  # remove background
                        self.lb_ip.setText('File loaded:   {}'.format(fn))
                        self.pb_del.setEnabled(True)  # delete single image
                        if self.cb1.findText('Raw image') < 0:
                            self.cb1.addItem('Raw image')
                        self.cb1.setCurrentText('Raw image')
                        QApplication.processEvents()
                    except:
                        self.img_xanes = np.zeros([1, 100, 100])
                        print('image not exist')
                    finally:
                        self.img_update = deepcopy(self.img_xanes)
                        self.update_canvas_img()
                        self.xanes_eng = np.array([])
                        self.lb_eng1.setText('No energy data ...')
                        self.lb_eng2.setVisible(True)
                        self.tx_eng.setVisible(True)
                        self.pb_eng.setVisible(True)
        elif flag_read_from_database:
            print('read_from_databroker, not implemented yet')
        else:
            print('Something happened in loading file ... :(')
        self.pb_ld.setEnabled(True)
        QApplication.processEvents()

    def update_msg(self):
        self.lb_msg.setFont(self.font1)
        self.lb_msg.setText('Message: ' + self.msg)
        self.lb_msg.setStyleSheet('color: rgb(200, 50, 50);')

    def manu_energy_input(self):
        energy_list_old = deepcopy(self.xanes_eng)
        com = self.tx_eng.text()
        try:
            exec(com)
            if len(self.xanes_eng) != self.img_xanes.shape[0] or self.img_xanes.shape[0] <=3:
                self.msg = 'invalid command of number of energy does not match number of images'
                self.xanes_eng = deepcopy(energy_list_old)
            else:
                st = '{0:2.4f}, {1:2.4f}, ..., {2:2.4f}    totally, {3} energies'.format(self.xanes_eng[0], self.xanes_eng[1], self.xanes_eng[-1], len(self.xanes_eng))
                self.lb_eng1.setText(st)
                self.msg = 'command executed'
        except:
            self.msg = 'un-recognized python command '
        finally:
            self.update_msg()

    def sliderval(self):
        canvas = self.canvas1
        img_index = self.sl1.value()
        canvas.current_img_index = img_index
        canvas.current_img = canvas.img_stack[img_index]
        img = canvas.img_stack[img_index]
        canvas.update_img_one(img, img_index=img_index)

    def get_slider_color_scale_value(self):
        t = self.sl_color.value()
        t = np.power(10, t / 50. - 1)
        return t

    def slider_color_scale(self):
        scale = self.get_slider_color_scale_value()
        self.lb_colormax.setText(f'x {scale:1.2f}')
        selected_channel = int(self.cb_color_channel.currentText())
        if self.cb1.currentText() == 'Color mix':
            img = self.img_colormix_raw
            img[selected_channel] *= scale
            self.img_colormix = self.convert_rgb_img(img, self.color_vec)
            self.update_canvas_img()
        else:
            self.msg = 'invalid colormix'
            self.update_msg()

    def norm_txm(self):
        self.pb_norm_txm.setText('wait ...')
        QApplication.processEvents()
        canvas = self.canvas1
        tmp = canvas.img_stack
        tmp = rm_abnormal(tmp)
        tmp = -np.log(tmp)
        tmp[np.isinf(tmp)] = 0
        tmp[np.isnan(tmp)] = 0
        tmp[tmp<0] = 0
        self.img_update = deepcopy(tmp)
        self.pb_norm_txm.setText('Norm. TMX (-log)')
        del tmp
        QApplication.processEvents()
        if self.cb1.findText('Image updated') < 0:
            self.cb1.addItem('Image updated')
        print('img = -log(img) \n"img_updated" has been created or updated')
        self.msg = 'img = -log(img)'
        self.update_msg()
        if not self.cb1.currentText() == 'Image updated':
            self.cb1.setCurrentText('Image updated')
        else:
            self.update_canvas_img()

    def xanes_align_img(self):
        self.pb_align.setText('Aligning ...')
        QApplication.processEvents()
        self.pb_align.setEnabled(False)
        canvas = self.canvas1
        prj = deepcopy(canvas.img_stack) * self.mask1 * self.mask2
        img_ali = deepcopy(prj)
        n = prj.shape[0]
        self.shift_list = []
        try:
            ref_index = int(self.tx_ali_ref.text())
            if ref_index >= prj.shape[0]:   # sequential align next image with its previous image
                self.shift_list.append([0, 0])
                for i in range(1, n):
                    print('Aligning image slice ' + str(i))

                    if self.rd_ali1.isChecked():
                        method = self.cb_ali.currentText().strip()
                        img_ali[i], rsft, csft, _ = align_img_stackreg(prj[i - 1], prj[i], method=method)
                    elif self.rd_ali2.isChecked():
                        _, rsft, csft = align_img(prj[i - 1], prj[i])
                        img_ali[i] = shift(canvas.img_stack[i], [rsft, csft], mode='constant', cval=0)
                    self.shift_list.append([rsft, csft])
                    self.msg = f'Aligned image slice {i}, row_shift: {rsft:3.2f}, col_shift: {csft:3.2f}'
                    self.update_msg()
                    QApplication.processEvents()
            else:                                            # align all images with image_stack[ref_index]
                for i in range(0, n):
                    print('Aligning image slice ' + str(i))
                    self.msg = 'Aligning image slice ' + str(i)
                    self.update_msg()
                    if self.rd_ali1.isChecked():
                        method = self.cb_ali.currentText().strip()
                        img_ali[i], rsft, csft, _ = align_img_stackreg(prj[ref_index], prj[i], method=method)
                    elif self.rd_ali2.isChecked():
                        _, rsft, csft = align_img(prj[ref_index], prj[i])
                        img_ali[i] = shift(canvas.img_stack[i], [rsft, csft], mode='constant', cval=0)
                    self.msg = f'Aligned image slice {i}, row_shift: {rsft:3.2f}, col_shift: {csft:3.2f}'
                    self.update_msg()
                    self.shift_list.append([rsft, csft])
                    QApplication.processEvents()
            self.img_update = deepcopy(img_ali)
            if self.cb1.findText('Image updated') < 0:
                self.cb1.addItem('Image updated')
            if not self.cb1.currentText() == 'Image updated':
                self.cb1.setCurrentText('Image updated')
            else:
                self.update_canvas_img()
            self.pb_align.setText('Align Img')
            self.pb_align.setEnabled(True)
            print('Image aligned.\n Item "Aligned Image" has been added.')
            self.msg = 'Image aligning finished'
        except:
            self.msg = 'image stack has only one image slice, aligning aborted... '
        finally:
            self.update_msg()
            del prj, img_ali

    def xanes_align_img_roi(self):
        self.pb_align_roi.setText('Aligning ...')
        QApplication.processEvents()
        self.pb_align_roi.setEnabled(False)
        canvas = self.canvas1
        roi_list = canvas.roi_list
        img_ali = deepcopy(canvas.img_stack)
        self.shift_list = []
        try:
            n = int(self.tx_ali_roi.text())
            roi_selected = 'roi_' + str(n)
        except:
            print('index should be integer')
            n = 0
            roi_selected = 'None'
        n_roi = self.lst_roi.count()
        if n > n_roi or n < 0:
            n = 0
            roi_selected = 'None'
        if not roi_selected == 'None':
            print(f'{roi_selected}')
            try:
                roi_cord = np.int32(np.array(roi_list[roi_selected][:4]))
                print(f'{roi_cord}')
                a, b, c, d = roi_cord[0], roi_cord[1], roi_cord[2], roi_cord[3]
                x1 = min(a, c)
                x2 = max(a, c)
                y1 = min(b, d)
                y2 = max(b, d)
                prj = (img_ali * self.mask1 * self.mask2)[:, y1: y2, x1: x2]
                s = prj.shape
                ref_index = int(self.tx_ali_ref.text())
                if ref_index >= s[0]:  # sequantial align next image with its previous image
                    print(f'sequency : {ref_index}')
                    self.shift_list.append([0, 0])
                    for i in range(1, s[0]):
                        print('Aligning image slice ' + str(i))
                        if self.rd_ali1.isChecked():
                            method = self.cb_ali.currentText().strip()
                            rsft, csft, sr = align_img_stackreg(prj[i - 1], prj[i], align_flag=0, method=method)
                            img_ali[i] = sr.transform(img_ali[i])
                        elif self.rd_ali2.isChecked():
                            rsft, csft = align_img(prj[i - 1], prj[i], align_flag=0)
                            img_ali[i] = shift(img_ali[i], [rsft, csft], mode='constant', cval=0)
                        self.msg = f'Aligned image slice {i}, row_shift: {rsft:3.2f}, col_shift: {csft:3.2f}'
                        self.update_msg()
                        self.shift_list.append([rsft, csft])
                        QApplication.processEvents()
                else:  # align all images with image_stack[ref_index]
                    for i in range(0, s[0]):
                        print('Aligning image slice ' + str(i))
                        if self.rd_ali1.isChecked():
                            method = self.cb_ali.currentText().strip()
                            rsft, csft, sr = align_img_stackreg(prj[ref_index], prj[i], align_flag=0, method=method)
                            img_ali[i] = sr.transform(img_ali[i])
                        elif self.rd_ali2.isChecked():
                            rsft, csft = align_img(prj[ref_index], prj[i], align_flag=0)
                            img_ali[i] = shift(img_ali[i], [rsft, csft], mode='constant', cval=0)
                        self.shift_list.append([rsft, csft])
                        self.msg = f'Aligned image slice {i}, row_shift: {rsft:3.2f}, col_shift: {csft:3.2f}'
                        self.update_msg()
                        QApplication.processEvents()
                self.img_update = deepcopy(img_ali)
                if self.cb1.findText('Image updated') < 0:
                    self.cb1.addItem('Image updated')
                    self.cb1.setCurrentText('Image updated')
                print('Image aligned.\n Item "Aligned Image" has been added.')
                self.msg = 'Image aligning finished'
            except:
                self.msg = 'image stack has only one image slice, aligning aborted... '
            finally:
                self.pb_align_roi.setText('Align Img (ROI)')
                self.pb_align_roi.setEnabled(True)
                self.update_msg()      
        else:
            self.pb_align_roi.setText('Align Img (ROI)')
            self.pb_align_roi.setEnabled(True)
            self.msg = 'Invalid roi index for aligning, nothing applied'
            self.update_msg()

    def apply_shift(self):
        canvas =self.canvas1
        prj = deepcopy(canvas.img_stack)
        img_ali = np.zeros(prj.shape)
        n = prj.shape[0]
        n1 = len(self.shift_list)
        if n!=n1:
            self.msg = 'number of shifts does not match number of images, aligning will not perform'
        else:
            for i in range(min(n, n1)):
                rsft, csft = self.shift_list[i]
                print(f'Aligned image slice {i}, row_shift: {rsft}, col_shift: {csft}')
                img_ali[i] = shift(prj[i], [rsft, csft], mode='constant', cval=0)
                self.msg = f'Aligned image slice {i}, row_shift: {rsft}, col_shift: {csft}'
                self.update_msg()
            self.img_update = deepcopy(img_ali)
            if self.cb1.findText('Image updated') < 0:
                self.cb1.addItem('Image updated')
            self.cb1.setCurrentText('Image updated')
            self.update_canvas_img()
            self.msg = 'Applied shift to current image stack, Image upated'
        self.update_msg()

    def load_shift(self):
        options = QFileDialog.Option()
        options |= QFileDialog.DontUseNativeDialog
        file_type = 'txt files (*.txt)'
        fn, _ = QFileDialog.getOpenFileName(xanes, "QFileDialog.getOpenFileName()", "", file_type, options=options)
        if fn:
            try:
                print(fn)
                fn_shift = fn.split('/')[-1]
                print(f'selected shift list: {fn_shift}')
                self.msg = f'selected shift list: {fn_shift}'
                self.lb_shift.setText('  '+ fn_shift)
                QApplication.processEvents()
                self.shift_list = np.loadtxt(fn)
            except:
                print('un-recognized shift list')
            finally:
                self.update_msg()

    def save_shift(self):
        num = len(self.shift_list)
        if num == 0:
            self.msg = 'shift list not exist'
        elif num != self.img_xanes.shape[0]:
            self.msg = 'number of shifts not match number of images'
        else:
            try:
                options = QFileDialog.Option()
                options |= QFileDialog.DontUseNativeDialog
                file_type = 'txt files (*.txt)'
                fn, _ = QFileDialog.getSaveFileName(self, 'Save Spectrum', "", file_type, options=options)
                if fn[-4:] != '.txt':
                    fn += '.txt'
                np.savetxt(fn, self.shift_list, '%3.2f')
                self.msg = fn + ' saved.'
            except:
                self.msg = f'fails to save {fn}'
        self.update_msg()

    def update_canvas_img(self):
        canvas = self.canvas1
        slide = self.sl1
        type_index = self.cb1.currentText()
        QApplication.processEvents()
        canvas.draw_line = False
        self.pb_adj_cmap.setEnabled(True)
        self.pb_set_cmap.setEnabled(True)
        self.pb_del.setEnabled(True)
        if len(canvas.mask.shape) > 1:
            print(f'canvas_mask.shape = {canvas.mask.shape}')
            sh = canvas.img_stack.shape
            canvas.special_info = None
            canvas.title = []
            canvas.update_img_stack()
            slide.setMaximum(max(sh[0] - 1, 0))
        try:
            if type_index == 'Raw image':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                self.pb_roi_draw.setEnabled(True)
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                sh = self.img_xanes.shape
                canvas.img_stack = self.smooth(self.img_xanes)
                canvas.special_info = None
                canvas.current_img_index = self.sl1.value()
                canvas.title = [f'#{i:3d},   {self.xanes_eng[i]:2.4f} keV' for i in range(len(self.xanes_eng))]
                canvas.update_img_stack()
                slide.setMaximum(max(sh[0] - 1, 0))
                self.current_image = self.img_xanes[0]
            elif type_index == 'Image updated':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                img = self.img_update
                self.pb_roi_draw.setEnabled(True)
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                sh = img.shape
                canvas.img_stack = self.smooth(img)
                canvas.special_info = None
                canvas.current_img_index = self.sl1.value()
                canvas.title = [f'#{i:3d},   {self.xanes_eng[i]:2.4f} keV' for i in range(len(self.xanes_eng))]
                canvas.update_img_stack()
                slide.setMaximum(max(sh[0] - 1, 0))
                self.current_image = img[0]
                #self.data_summary[type_index] = self.canvas1.img_stack
            elif type_index == 'XANES Fit (thickness)': # will be saved
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                img = self.img_pre_edge_sub_mean
                self.pb_roi_draw.setEnabled(True)
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                sh = img.shape
                canvas.img_stack = self.smooth(img)
                canvas.special_info = None
                canvas.current_img_index = 0
                canvas.title = self.elem_label
                canvas.update_img_stack()
                slide.setMaximum(0)
                self.current_image = img[0]
                self.data_summary[type_index] = self.canvas1.img_stack
            elif type_index == 'XANES Fit (ratio, summed to 1)': # will be saved
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                img = self.xanes_2d_fit
                img_sum = np.sum(img, axis=0, keepdims=True)
                img_sum[np.abs(img_sum) < 1e-6] = 1e6
                img = img / img_sum
                img = rm_abnormal(img)
                self.pb_roi_draw.setEnabled(True)
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                sh = img.shape
                canvas.img_stack = self.smooth(img)
                canvas.special_info = None
                # canvas.current_img_index = 0
                canvas.current_img_index = self.sl1.value()
                canvas.title = self.elem_label
                canvas.update_img_stack()
                slide.setMaximum(max(sh[0] - 1, 0))
                self.current_image = img[0]
                self.data_summary[type_index] = self.canvas1.img_stack
            elif type_index == 'XANES Fit (Elem. concentration)':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                img = self.xanes_2d_fit
                img_sum = np.sum(img, axis=0, keepdims=True)
                img_sum[np.abs(img_sum) < 1e-6] = 1e6
                img = img / img_sum
                img = img * self.img_pre_edge_sub_mean
                img = rm_abnormal(img)
                self.pb_roi_draw.setEnabled(True)
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                sh = img.shape
                canvas.img_stack = self.smooth(img)
                canvas.special_info = None
                # canvas.current_img_index = 0
                canvas.current_img_index = self.sl1.value()
                canvas.title = self.elem_label
                canvas.update_img_stack()
                slide.setMaximum(max(sh[0] - 1, 0))
                self.current_image = img[0]
                self.data_summary[type_index] = self.canvas1.img_stack
            elif type_index == 'XANES Fit error':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                img = self.xanes_fit_cost
                self.pb_roi_draw.setEnabled(True)
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                canvas.img_stack = self.smooth(img)
                canvas.special_info = None
                # canvas.current_img_index = 0
                canvas.current_img_index = self.sl1.value()
                canvas.title = []
                canvas.update_img_stack()
                slide.setMaximum(0)
                self.current_image = img[0]
                self.data_summary[type_index] = self.canvas1.img_stack
            elif type_index == 'XANES Fit offset':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                img = self.xanes_2d_fit_offset
                self.pb_roi_draw.setEnabled(True)
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                canvas.img_stack = self.smooth(img)
                canvas.special_info = None
                # canvas.current_img_index = 0
                canvas.current_img_index = self.sl1.value()
                canvas.title = []
                canvas.update_img_stack()
                slide.setMaximum(0)
                self.current_image = img[0]
                self.data_summary[type_index] = self.canvas1.img_stack
            elif type_index == 'Image regulation':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                img = self.img_regulation
                self.pb_roi_draw.setEnabled(True)
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                sh = img.shape
                canvas.img_stack = self.smooth(img)
                canvas.special_info = None
                # canvas.current_img_index = 0
                canvas.current_img_index = self.sl1.value()
                canvas.title = [f'#{i:3d},   {self.xanes_eng[i]:2.4f} keV' for i in range(len(self.xanes_eng))]
                canvas.update_img_stack()
                slide.setMaximum(max(sh[0] - 1, 0))
                self.current_image = img[0]
            elif type_index == 'Mask':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                img = self.mask
                if len(img.shape) == 2:
                    img = img.reshape([1, img.shape[0], img.shape[1]])
                self.pb_roi_draw.setEnabled(True)
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                canvas.img_stack = self.smooth(img)
                canvas.special_info = None
                canvas.current_img_index = 0
                canvas.title = []
                canvas.update_img_stack()
                slide.setMaximum(0)
                self.current_image = img[0]
                self.data_summary[type_index] = self.canvas1.img_stack
            elif type_index == 'Smart Mask':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                img = self.smart_mask
                if len(img.shape) == 2:
                    img = img.reshape([1, img.shape[0], img.shape[1]])
                self.pb_roi_draw.setEnabled(True)
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                canvas.img_stack = self.smooth(img)
                canvas.special_info = None
                canvas.current_img_index = self.sl1.value()
                if canvas.current_img_index >= img.shape[0]:
                    canvas.current_img_index = 0
                canvas.title = []
                canvas.update_img_stack()
                slide.setMaximum(img.shape[0]-1)
                self.current_image = img[0]
                self.data_summary[type_index] = self.canvas1.img_stack
            elif type_index == 'Image compress':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                img = self.img_compress
                self.pb_roi_draw.setEnabled(True)
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                canvas.img_stack = self.smooth(img)
                canvas.special_info = None
                # canvas.current_img_index = 0
                canvas.current_img_index = self.sl1.value()
                canvas.title = []
                canvas.update_img_stack()
                slide.setMaximum(0)
                self.data_summary[type_index] = self.canvas1.img_stackv
            elif type_index == 'Image Labels':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                img = self.img_labels
                self.pb_roi_draw.setEnabled(True)
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                canvas.img_stack = self.smooth(img)
                canvas.special_info = None
                # canvas.current_img_index = 0
                canvas.current_img_index = self.sl1.value()
                canvas.title = []
                canvas.update_img_stack()
                slide.setMaximum(0)
                self.data_summary[type_index] = self.canvas1.img_stack
            elif type_index == 'Noise removal':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                img = self.img_rm_noise
                self.pb_roi_draw.setEnabled(True)
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                canvas.img_stack = self.smooth(img)
                canvas.special_info = None
                # canvas.current_img_index = 0
                canvas.current_img_index = self.sl1.value()
                canvas.title = []
                canvas.update_img_stack()
                slide.setMaximum(0)
                self.data_summary[type_index] = self.canvas1.img_stack
            elif type_index == 'XANES Edge Fit':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                img = self.xanes_edge_fit
                self.pb_roi_draw.setEnabled(True)
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                canvas.img_stack = self.smooth(img)
                canvas.special_info = None
                canvas.current_img_index = 0
                # canvas.current_img_index = self.sl1.value()
                canvas.title = []
                canvas.update_img_stack()
                slide.setMaximum(0)
                self.current_image = img[0]
                self.data_summary[type_index] = self.canvas1.img_stack
            elif type_index == 'XANES Peak Fit':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                img = self.xanes_peak_fit
                self.pb_roi_draw.setEnabled(True)
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                canvas.img_stack = self.smooth(img)
                canvas.special_info = None
                canvas.current_img_index = 0
                try:
                    canvas.cmin = float(self.tx_edge_s.text())
                    self.tx_cmin.setText(f'{canvas.cmin}')
                    canvas.cmax = float(self.tx_edge_e.text())
                    self.tx_cmax.setText(f'{canvas.cmax}')
                except:
                    pass
                # canvas.current_img_index = self.sl1.value()
                canvas.title = []
                canvas.update_img_stack()
                slide.setMaximum(0)
                self.current_image = img[0]
                self.data_summary[type_index] = self.canvas1.img_stack
            elif type_index == 'XANES Peak Fit Height':
                self.img_colormix_raw  = np.array([])
                canvas.rgb_flag = 0
                img = self.xanes_peak_fit_height
                self.pb_roi_draw.setEnabled(True)
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                canvas.img_stack = self.smooth(img)
                canvas.special_info = None
                canvas.current_img_index = 0
                # canvas.current_img_index = self.sl1.value()
                canvas.title = []
                canvas.update_img_stack()
                slide.setMaximum(0)
                self.current_image = img[0]
                self.data_summary[type_index] = self.canvas1.img_stack
            elif type_index == 'Peak percentage':
                self.img_colormix_raw = np.array([])
                canvas.rgb_flag = 0
                img = self.peak_percentage
                self.pb_roi_draw.setEnabled(True)
                canvas.x, canvas.y = [], []
                canvas.axes.clear()  # this is important, to clear the current image before another imshow()
                canvas.img_stack = self.smooth(img)
                canvas.special_info = None
                canvas.current_img_index = 0
                canvas.cmin = 0
                canvas.cmax = 1
                self.tx_cmin.setText('0')
                self.tx_cmax.setText('1')
                # canvas.current_img_index = self.sl1.value()
                canvas.title = []
                canvas.update_img_stack()
                slide.setMaximum(0)
                self.current_image = img[0]
                self.data_summary[type_index] = self.canvas1.img_stack
            elif type_index == 'Color mix':
                img = self.img_colormix
                img = self.smooth(img, axis=2)
                self.img_colormix = deepcopy(img)
                canvas.rgb_flag = 1
                cmax = np.float(self.tx_cmax.text())
                cmin = np.float(self.tx_cmin.text())
                # img = (img - cmin) / (cmax - cmin)
                canvas.img_stack = img
                canvas.cmin = cmin
                canvas.cmax = cmax
                canvas.current_img_index = 0
                canvas.title = 'RGB colormix'
                slide.setMaximum(0)
                # canvas.update_img_stack()
                canvas.set_contrast(cmin, cmax)
        except:
            self.msg = f'fails to update {type_index}'
            self.update_msg()
        QApplication.processEvents()

    def update_roi_list(self, mode='add', item_name=''):
        # de-select all the existing selection
        if mode == 'add':
            for i in range(self.lst_roi.count()):
                item = self.lst_roi.item(i)
                item.setSelected(False)
            item = QListWidgetItem(item_name)
            self.lst_roi.addItem(item)
            self.lst_roi.setCurrentItem(item)
        elif mode == 'del_all':
            self.lst_roi.clear()
        elif mode == 'del':
            for selectItem in self.lst_roi.selectedItems():
                self.lst_roi.removeItemWidget(selectItem)

    def change_colormap(self):
        canvas = self.canvas1
        cmap = self.cb_cmap.currentText()
        canvas.colormap = cmap
        canvas.colorbar_on_flag = True
        canvas.update_img_one(canvas.current_img, canvas.current_img_index)

    def auto_contrast(self):
        canvas = self.canvas1
        cmin, cmax = canvas.auto_contrast()
        self.tx_cmax.setText('{:6.3f}'.format(cmax))
        self.tx_cmin.setText('{:6.3f}'.format(cmin))

    def set_contrast(self):
        canvas = self.canvas1
        cmax = np.float(self.tx_cmax.text())
        cmin = np.float(self.tx_cmin.text())
        canvas.set_contrast(cmin, cmax)


class MyCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3, dpi=120, obj=[]):
        self.obj = obj
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.axes.axis('off')
        self.cmax = 1
        self.cmin = 0
        self.rgb_flag = 0
        self.img_stack = np.zeros([1, 100, 100])
        self.current_img = self.img_stack[0]
        self.current_img_index = 0
        self.mask = np.array([1])
        self.rgb_mask = np.array([1])
        self.colorbar_on_flag = True
        self.colormap = 'viridis'
        self.title = []
        self.draw_line = False
        self.overlay_flag = True
        self.x, self.y, = [], []
        self.plot_label = ''
        self.legend_flag = False
        self.roi_list = {}
        self.roi_color = {}
        self.roi_count = 0
        self.show_roi_flag = False
        self.current_roi = [0, 0, 0, 0, '0'] # x1, y1, x2, y1, roi_name
        self.color_list = ['red', 'brown', 'orange', 'olive', 'green', 'cyan', 'blue', 'pink', 'purple', 'gray']
        self.current_color = 'red'
        self.special_info = None
        FigureCanvas.__init__(self, self.fig)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.setParent(parent)
        self.mpl_connect('motion_notify_event', self.mouse_moved)

    def mouse_moved(self, mouse_event):
        if mouse_event.inaxes:
            x, y = mouse_event.xdata, mouse_event.ydata
            self.obj.lb_x_l.setText('x: {:3.2f}'.format(x))
            self.obj.lb_y_l.setText('y: {:3.2f}'.format(y))
            row = int(np.max([np.min([self.current_img.shape[0], y]), 0]))
            col = int(np.max([np.min([self.current_img.shape[1], x]), 0]))
            try:
                z = self.current_img[row][col]
                self.obj.lb_z_l.setText('intensity: {:3.4f}'.format(z))
            except:
                self.obj.lb_z_l.setText('')

    def update_img_stack(self):
        self.axes = self.fig.add_subplot(111)
        if self.rgb_flag:  # RGB image
            return self.update_img_one(self.img_stack)
        elif self.img_stack.shape[0] == 0:
            img_blank = np.zeros([100, 100])
            return self.update_img_one(img_blank, img_index=0)
        else:
            s = self.img_stack.shape
            if len(s) == 2:
                self.img_stack = self.img_stack.reshape(1, s[0], s[1])
            if self.current_img_index >= len(self.img_stack):
                self.current_img_index = 0
            return self.update_img_one(self.img_stack[self.current_img_index], img_index=self.current_img_index)

    def update_img_one(self, img=np.array([]), img_index=0):
        self.axes.clear()
        try:
            if self.rgb_flag:
                self.rgb_mask = self.mask
                if len(self.mask.shape) == 3:
                    self.rgb_mask = self.mask[0]
                for i in range(img.shape[2]):
                    img[:,:,i] *= self.rgb_mask
                self.im = self.axes.imshow(img)
                self.draw()
            else:
                if len(img) == []: img = self.current_img
                self.current_img = img
                self.current_img_index = img_index
                self.im = self.axes.imshow(img*self.mask, cmap=self.colormap, vmin=self.cmin, vmax=self.cmax)
                self.axes.axis('on')
                self.axes.set_aspect('equal', 'box')
                if len(self.title) == self.img_stack.shape[0]:
                    self.axes.set_title('current image: ' + self.title[img_index])
                else:
                    self.axes.set_title('current image: ' + str(img_index))
                self.axes.title.set_fontsize(10)
                if self.colorbar_on_flag:
                    self.add_colorbar()
                    self.colorbar_on_flag = False
                self.draw()
            if self.show_roi_flag:
                for i in range(len(self.roi_list)):
                    self.roi_display(self.roi_list[f'roi_{i}'])
        except:
            print('Error in updating image')

    def add_line(self):
        if self.draw_line:
            if self.overlay_flag:
                self.axes.plot(self.x, self.y, '-', color=self.current_color, linewidth=1.0, label=self.plot_label)
            else:
                self.rm_colorbar()
                line, = self.axes.plot(self.x, self.y, '.-', color=self.current_color, linewidth=1.0, label=self.plot_label)
                if self.legend_flag:
                    self.axes.legend(handles=[line])
                self.axes.axis('on')
                self.axes.set_aspect('auto')
                self.draw()

    def draw_roi(self):
        self.cidpress = self.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.mpl_connect('button_release_event', self.on_release)
        self.show_roi_flag = True

    def on_press(self, event):
        x1, y1 = event.xdata, event.ydata
        self.current_roi[0] = x1
        self.current_roi[1] = y1

    def on_release(self, event):
        x2, y2 = event.xdata, event.ydata
        self.current_roi[2] = x2
        self.current_roi[3] = y2
        self.current_roi[4] = str(self.roi_count)
        self.roi_add_to_list()
        self.roi_display(self.current_roi)
        self.roi_disconnect()

    def roi_disconnect(self):
        self.mpl_disconnect(self.cidpress)
        self.mpl_disconnect(self.cidrelease)

    def roi_display(self, selected_roi):
        x1, y1 = selected_roi[0], selected_roi[1]
        x2, y2 = selected_roi[2], selected_roi[3]
        roi_index = selected_roi[4]
        self.x = [x1, x2, x2, x1, x1]
        self.y = [y1, y1, y2, y2, y1]
        self.draw_line = True
        self.add_line()
        self.draw_line = False
        roi_name = f'#{roi_index}'
        s = self.current_img.shape
        self.axes.annotate(roi_name, xy=(x1, y1 - s[0]//40),
                           bbox={'facecolor': self.current_color, 'alpha': 0.5, 'pad': 2},
                           fontsize=10)
        self.draw()
        # self.roi_count += 1
        self.obj.tx_roi_x1.setText('{:4.1f}'.format(x1))
        self.obj.tx_roi_y1.setText('{:4.1f}'.format(y1))
        self.obj.tx_roi_x2.setText('{:4.1f}'.format(x2))
        self.obj.tx_roi_y2.setText('{:4.1f}'.format(y2))
        self.obj.pb_roi_draw.setEnabled(True)
        QApplication.processEvents()

    def roi_add_to_list(self, roi_name=''):
        if not len(roi_name):
            roi_name = 'roi_' + str(self.roi_count)
        self.roi_list[roi_name] = deepcopy(self.current_roi)
        self.current_color = self.color_list[self.roi_count % 10]
        self.roi_color[roi_name] = self.current_color
        self.roi_count += 1
        self.obj.update_roi_list(mode='add', item_name=roi_name)

    def set_contrast(self, cmin, cmax):
        self.cmax = cmax
        self.cmin = cmin
        self.colorbar_on_flag = True
        if self.rgb_flag:
            img = (self.img_stack - cmin) / (cmax - cmin)
            mask = deepcopy(self.mask)
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask,axis=2)
                mask = np.repeat(mask, repeats=3, axis=2)
            self.update_img_one(img * mask)
        else:
            self.update_img_one(self.current_img*self.mask, self.current_img_index)

    def auto_contrast(self):
        img = self.current_img*self.mask
        self.cmax = np.max(img)
        self.cmin = np.min(img)
        self.colorbar_on_flag = True
        self.update_img_one(self.current_img*self.mask, self.current_img_index)
        return self.cmin, self.cmax

    def rm_colorbar(self):
        try:
            self.cb.remove()
            self.draw()
        except:
            pass

    def add_colorbar(self):
        if self.colorbar_on_flag:
            try:
                self.cb.remove()
                self.draw()
            except:
                pass
            self.divider = make_axes_locatable(self.axes)
            self.cax = self.divider.append_axes('right', size='3%', pad=0.06)
            self.cb = self.fig.colorbar(self.im, cax=self.cax, orientation='vertical')
            self.cb.ax.tick_params(labelsize=10)
            self.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    xanes = App()
    xanes.show()
    sys.exit(app.exec_())
