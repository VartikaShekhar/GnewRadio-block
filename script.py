#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# python reading_MEP.py --data-dir mep12 --channel chA --start-sec 400
# GNU Radio Python Flow Graph
# Title: Not titled yet
# Author: ar
# GNU Radio version: 3.10.9.2

from PyQt5 import Qt
from PyQt5 import QtCore
from PyQt5.QtWidgets import QStyle, QStyleOptionSlider
from gnuradio import qtgui
from gnuradio import blocks
from gnuradio import gr
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
import sip
import gnuradio.lora_sdr as lora_sdr
import argparse
import reading_MEP_epy_block_0 as epy_block_0  # embedded python block
import os

mep8_file = "/media/research1/T9/vla data/mep08_b081_250710/sr10MHz"
mep10_file = "/media/research1/T9/vla data/mep10_b08a_250710/sr10MHz"
mep12_file = "/media/research1/T9/vla data/mep12_b0f9_250710/sr10MHz/sr10MHz"


class ClickableSlider(Qt.QSlider):
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)

            if self.orientation() == QtCore.Qt.Horizontal:
                slider_length = self.style().pixelMetric(QStyle.PM_SliderLength, opt, self)
                slider_space = self.style().pixelMetric(QStyle.PM_SliderSpaceAvailable, opt, self)

                value = QStyle.sliderValueFromPosition(
                    self.minimum(),
                    self.maximum(),
                    event.pos().x() - slider_length // 2,
                    slider_space,
                    opt.upsideDown,
                )
                self.setValue(value)
                self.sliderReleased.emit()
                event.accept()
                return

        super().mousePressEvent(event)


class reading_MEP(gr.top_block, Qt.QWidget):
    def __init__(self, data_dir, channel, start_sec):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)
        Qt.QWidget.__init__(self)

        self.setWindowTitle("Not titled yet")
        qtgui.util.check_set_qss()

        try:
            self.setWindowIcon(Qt.QIcon.fromTheme("gnuradio-grc"))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)

        # ---- scroll layout boilerplate ----
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)

        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)

        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "reading_MEP")
        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 10e6
        self.decimation = decimation = 20
        self.new_samp_rate = new_samp_rate = int(samp_rate / decimation)

        self.mep8 = mep8 = data_dir
        self.drf_range = float(start_sec)

        # slider resolution: 0.01 s
        self.slider_scale = 100

        ##################################################
        # Blocks
        ##################################################
        self.epy_block_0 = epy_block_0.digital_rf_relseek_source(
            data_dir=mep8,
            channel=channel,
            seek_seconds=self.drf_range,
            subchannel=0,
        )

        self.duration_s = self.epy_block_0.get_duration_s()
        self.drf_range = max(0.0, min(self.drf_range, self.duration_s))

        ##################################################
        # Seek controls UI
        ##################################################

        # playback label
        self.playback_label = Qt.QLabel()
        self.playback_label.setAlignment(QtCore.Qt.AlignCenter)
        self.top_layout.addWidget(self.playback_label)

        # slider row
        self.slider_row = Qt.QVBoxLayout()

        self.slider_title = Qt.QLabel("Seek position")
        self.slider_row.addWidget(self.slider_title)

        self.seek_slider = ClickableSlider(QtCore.Qt.Horizontal)
        self.seek_slider.setMinimum(0)
        self.seek_slider.setMaximum(max(1, int(self.duration_s * self.slider_scale)))
        self.seek_slider.setValue(int(self.drf_range * self.slider_scale))
        self.slider_row.addWidget(self.seek_slider)

        self.top_layout.addLayout(self.slider_row)

        # manual seek row
        self.seek_row = Qt.QHBoxLayout()

        self.seek_label = Qt.QLabel("Go to second:")
        self.seek_row.addWidget(self.seek_label)

        self.seek_edit = Qt.QLineEdit()
        self.seek_edit.setPlaceholderText("Enter seconds, e.g. 400.5")
        self.seek_edit.setText(f"{self.drf_range:.2f}")
        self.seek_row.addWidget(self.seek_edit)

        self.seek_button = Qt.QPushButton("Go")
        self.seek_row.addWidget(self.seek_button)

        self.top_layout.addLayout(self.seek_row)

        # prevent recursive slider updates
        self._updating_slider = False

        # connect UI
        self.seek_slider.sliderReleased.connect(self.on_slider_released)
        self.seek_button.clicked.connect(self.on_seek_button_clicked)
        self.seek_edit.returnPressed.connect(self.on_seek_button_clicked)
        self.seek_edit.editingFinished.connect(self.on_seek_button_clicked)

        ##################################################
        # QT GUI sinks
        ##################################################
        self.qtgui_sink_x_0_0_0_0 = qtgui.sink_c(
            8192,
            window.WIN_BLACKMAN_hARRIS,
            0,
            samp_rate,
            "Raw (10 MHz)",
            True,
            True,
            True,
            True,
            None,
        )
        self.qtgui_sink_x_0_0_0_0.set_update_time(0.1)
        self.qtgui_sink_x_0_0_0_0.enable_rf_freq(False)
        self._qtgui_sink_x_0_0_0_0_win = sip.wrapinstance(
            self.qtgui_sink_x_0_0_0_0.qwidget(), Qt.QWidget
        )
        self.top_layout.addWidget(self._qtgui_sink_x_0_0_0_0_win)

        self.qtgui_sink_x_0_0_0 = qtgui.sink_c(
            8192,
            window.WIN_BLACKMAN_hARRIS,
            0,
            new_samp_rate,
            "Filtered/Decimated",
            True,
            True,
            True,
            True,
            None,
        )
        self.qtgui_sink_x_0_0_0.set_update_time(0.1)
        self.qtgui_sink_x_0_0_0.enable_rf_freq(False)
        self._qtgui_sink_x_0_0_0_win = sip.wrapinstance(
            self.qtgui_sink_x_0_0_0.qwidget(), Qt.QWidget
        )
        self.top_layout.addWidget(self._qtgui_sink_x_0_0_0_win)

        ##################################################
        # DSP chain
        ##################################################
        self.blocks_throttle2_0 = blocks.throttle(
            gr.sizeof_gr_complex * 1, samp_rate, True, 0
        )

        self.blocks_conjugate_cc_0 = blocks.conjugate_cc()

        self.freq_xlating_fir_filter_xxx_0 = filter.freq_xlating_fir_filter_ccc(
            decimation,
            firdes.low_pass(1.0, samp_rate, 150e3, 40e3, window.WIN_HAMMING),
            (-2.525e6 + 11e3 + 0.2e3 + 0.11e3 + 0.03e3),
            samp_rate,
        )
        self.freq_xlating_fir_filter_xxx_0.set_min_output_buffer(16392)

        self.lora_rx_1 = lora_sdr.lora_sdr_lora_rx(
            bw=125000,
            cr=1,
            has_crc=False,
            impl_head=False,
            pay_len=255,
            samp_rate=new_samp_rate,
            sf=12,
            sync_word=[0x12],
            soft_decoding=True,
            ldro_mode=2,
            print_rx=[True, True],
        )

        self.blocks_message_debug_1 = blocks.message_debug(True, gr.log_levels.debug)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.epy_block_0, 0), (self.blocks_throttle2_0, 0))
        self.connect((self.blocks_throttle2_0, 0), (self.blocks_conjugate_cc_0, 0))

        self.connect((self.blocks_conjugate_cc_0, 0), (self.qtgui_sink_x_0_0_0_0, 0))
        self.connect((self.blocks_conjugate_cc_0, 0), (self.freq_xlating_fir_filter_xxx_0, 0))

        self.connect((self.freq_xlating_fir_filter_xxx_0, 0), (self.qtgui_sink_x_0_0_0, 0))
        self.connect((self.freq_xlating_fir_filter_xxx_0, 0), (self.lora_rx_1, 0))

        self.msg_connect((self.lora_rx_1, "out"), (self.blocks_message_debug_1, "log"))
        self.msg_connect((self.lora_rx_1, "out"), (self.blocks_message_debug_1, "print"))
        self.msg_connect((self.lora_rx_1, "out"), (self.blocks_message_debug_1, "store"))
        self.msg_connect((self.lora_rx_1, "out"), (self.blocks_message_debug_1, "print_pdu"))

        ##################################################
        # UI timer
        ##################################################
        self.ui_timer = Qt.QTimer(self)
        self.ui_timer.timeout.connect(self.update_playback_display)
        self.ui_timer.start(100)

        self.update_playback_display()

    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "reading_MEP")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()
        event.accept()

    ##################################################
    # Seek helpers
    ##################################################
    def get_drf_range(self):
        return self.drf_range

    def set_drf_range(self, drf_range):
        try:
            sec = max(0.0, min(float(drf_range), self.duration_s))
            self.drf_range = sec

            self.epy_block_0.set_seek_seconds(sec)

            # sync text box
            self.seek_edit.setText(f"{sec:.2f}")

            # sync slider
            slider_val = int(sec * self.slider_scale)
            slider_val = max(self.seek_slider.minimum(), min(slider_val, self.seek_slider.maximum()))
            self._updating_slider = True
            self.seek_slider.setValue(slider_val)
            self._updating_slider = False

            self.update_playback_display()

        except Exception as e:
            print("[GUI] Seek error:", e)

    def on_slider_released(self):
        if self._updating_slider:
            return
        sec = self.seek_slider.value() / self.slider_scale
        self.set_drf_range(sec)

    def on_seek_button_clicked(self):
        txt = self.seek_edit.text().strip()
        try:
            sec = float(txt)
            self.set_drf_range(sec)
        except ValueError:
            print(f"[GUI] Invalid seek value: {txt!r}")

    def update_playback_display(self):
        try:
            cur = self.epy_block_0.get_current_second()
            dur = self.duration_s
            self.playback_label.setText(f"{cur:.2f} s / {dur:.2f} s")

            # keep slider synced to playback unless user is dragging it
            if not self.seek_slider.isSliderDown():
                slider_val = int(cur * self.slider_scale)
                slider_val = max(self.seek_slider.minimum(), min(slider_val, self.seek_slider.maximum()))

                self._updating_slider = True
                self.seek_slider.setValue(slider_val)
                self._updating_slider = False

        except Exception as e:
            self.playback_label.setText("Playback: unavailable")
            print("[GUI] Playback label update error:", e)


def main(top_block_cls=reading_MEP, options=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="DigitalRF recording directory")
    parser.add_argument("--channel", help="DigitalRF channel name ")
    parser.add_argument("--start-sec", type=float, help="Initial seek time in seconds (relative)")
    parser.add_argument("--home-dir", type=str, help="Home directory for data files")

    # PC_ROOT = "/media/research1/T9"
    # WSL_ROOT = "/mnt/d"

    args, _ = parser.parse_known_args()

    if not os.path.isdir(args.data_dir):
        print(f"[ERROR] Directory does not exist: {args.data_dir}")
        sys.exit(1)
    
    args.data_dir = os.path.abspath(args.data_dir)

    print(f"[INFO] Using data directory: {args.data_dir}")

    # if args.home_dir == "pc":
    #     mep8_file = f"{PC_ROOT}/vla data/mep08_b081_250710/sr10MHz"
    #     mep10_file = f"{PC_ROOT}/vla data/mep10_b08a_250710/sr10MHz"
    #     mep12_file = f"{PC_ROOT}/vla data/mep12_b0f9_250710/sr10MHz/sr10MHz"
    # elif args.home_dir == "wsl":
    #     mep8_file = f"{WSL_ROOT}/vla data/mep08_b081_250710/sr10MHz"
    #     mep10_file = f"{WSL_ROOT}/vla data/mep10_b08a_250710/sr10MHz"
    #     mep12_file = f"{WSL_ROOT}/vla data/mep12_b0f9_250710/sr10MHz/sr10MHz"

    # if args.data_dir == "mep8":
    #     args.data_dir = mep8_file
    # elif args.data_dir == "mep10":
    #     args.data_dir = mep10_file
    # elif args.data_dir == "mep12":
    #     args.data_dir = mep12_file

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls(args.data_dir, channel=args.channel, start_sec=args.start_sec)
    tb.start()
    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()


if __name__ == "__main__":
    main()
