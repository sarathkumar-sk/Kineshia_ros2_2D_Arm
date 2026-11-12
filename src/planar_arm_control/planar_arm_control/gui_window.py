"""Qt widgets for the operator console (GUI thread only)."""

import time

import numpy as np
import pyqtgraph as pg
from action_msgs.msg import GoalStatus
from PyQt5 import QtCore, QtGui, QtWidgets

from planar_arm_control.timing import LoopTimer

JOINT_COLORS = ["#e6194b", "#3cb44b", "#4363d8"]
OK_STYLE = "color: #1b7f3b;"
WARN_STYLE = "color: #b36b00; font-weight: bold;"
ERR_STYLE = "color: #c0392b; font-weight: bold;"
SLIDER_SCALE = 10          # joint slider ticks per degree (0.1 deg resolution)
STALE_STATUS_S = 0.3       # ignore /controller_status this long after a command


def _spin(value, lo=-8.0, hi=8.0):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(2)
    s.setSingleStep(0.1)
    s.setValue(value)
    return s


class ArmView(pg.PlotWidget):
    """Live 2-D view of the arm, its reference, targets and the payload."""

    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        reach = sum(arm.link_lengths)
        self.setBackground("w")
        self.setAspectLocked(True)
        self.showGrid(x=True, y=True, alpha=0.25)
        self.setXRange(-reach - 0.5, reach + 0.5)
        self.setYRange(-1.0, reach + 0.5)
        self.setLabel("bottom", "x [m]")
        self.setLabel("left", "y [m]")
        self.setTitle("Arm (solid = measured, dashed = reference)")

        # ground + forbidden region
        below = pg.QtWidgets.QGraphicsRectItem(-20, -20, 40, 20)
        below.setBrush(pg.mkBrush(120, 120, 120, 40))
        below.setPen(pg.mkPen(None))
        self.addItem(below)
        self.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen("k", width=2)))
        # workspace boundary
        th = np.linspace(0, np.pi, 200)
        self.boundary = self.plot(reach * np.cos(th), reach * np.sin(th),
                                  pen=pg.mkPen((150, 150, 150), width=1,
                                               style=QtCore.Qt.DashLine))
        self.trail = self.plot(pen=pg.mkPen((100, 100, 255, 90), width=1))
        self.ref_arm = self.plot(pen=pg.mkPen((90, 90, 90, 160), width=2,
                                              style=QtCore.Qt.DashLine))
        # joint-slider goal preview (dotted; green = valid, red = below ground)
        self.preview_arm = self.plot(pen=None)
        self.arm_line = self.plot(pen=pg.mkPen("#2c3e50", width=6),
                                  symbol="o", symbolSize=11,
                                  symbolBrush="#f39c12", symbolPen="k")
        self.ee = pg.ScatterPlotItem(size=14, brush=pg.mkBrush("#c0392b"))
        self.addItem(self.ee)
        self.pick_mark = pg.ScatterPlotItem(size=18, symbol="s",
                                            pen=pg.mkPen("#2980b9", width=2),
                                            brush=pg.mkBrush(None))
        self.place_mark = pg.ScatterPlotItem(size=18, symbol="s",
                                             pen=pg.mkPen("#8e44ad", width=2),
                                             brush=pg.mkBrush(None))
        self.object = pg.ScatterPlotItem(size=14, symbol="s",
                                         brush=pg.mkBrush("#e67e22"),
                                         pen=pg.mkPen("k"))
        self.requested = pg.ScatterPlotItem(size=16, symbol="x",
                                            brush=pg.mkBrush("#c0392b"),
                                            pen=pg.mkPen("#c0392b", width=2))
        self.reachable = pg.ScatterPlotItem(size=16, symbol="o",
                                            pen=pg.mkPen("#27ae60", width=2),
                                            brush=pg.mkBrush(None))
        self.projection = self.plot(pen=pg.mkPen("#c0392b", width=1,
                                                 style=QtCore.Qt.DotLine))
        for item in (self.pick_mark, self.place_mark, self.object,
                     self.requested, self.reachable):
            self.addItem(item)
        self.ee_text = pg.TextItem(anchor=(0, 1), color="#2c3e50")
        self.addItem(self.ee_text)
        self.proj_text = pg.TextItem(anchor=(0.5, 1.0), color="#c0392b")
        self.addItem(self.proj_text)
        self._trail = []

    def update_arm(self, q, q_ref):
        if q is None:
            return
        pts = np.asarray(self.arm.forward_kinematics(q))
        self.arm_line.setData(pts[:, 0], pts[:, 1])
        self.ee.setData([pts[-1, 0]], [pts[-1, 1]])
        self.ee_text.setText(f"({pts[-1, 0]:.2f}, {pts[-1, 1]:.2f})")
        self.ee_text.setPos(pts[-1, 0] + 0.1, pts[-1, 1] + 0.1)
        self._trail.append(pts[-1])
        self._trail = self._trail[-400:]
        tr = np.asarray(self._trail)
        self.trail.setData(tr[:, 0], tr[:, 1])
        if q_ref is not None:
            r = np.asarray(self.arm.forward_kinematics(q_ref))
            self.ref_arm.setData(r[:, 0], r[:, 1])

    def set_preview(self, q, valid=True):
        """Show the joint-slider goal pose (q=None hides it)."""
        if q is None:
            self.preview_arm.setData([], [])
            return
        pts = np.asarray(self.arm.forward_kinematics(q))
        color = "#27ae60" if valid else "#c0392b"
        self.preview_arm.setData(
            pts[:, 0], pts[:, 1],
            pen=pg.mkPen(color, width=3, style=QtCore.Qt.DotLine),
            symbol="o", symbolSize=7, symbolBrush=color, symbolPen=None)

    def update_targets(self, status, pick=None, place=None):
        req = (status.requested_target.x, status.requested_target.y)
        rch = (status.reachable_target.x, status.reachable_target.y)
        if status.target_projected:
            self.requested.setData([req[0]], [req[1]])
            self.projection.setData([req[0], rch[0]], [req[1], rch[1]])
            self.proj_text.setText(f"requested ({req[0]:.1f}, {req[1]:.1f}): "
                                   f"unreachable → projected")
            self.proj_text.setPos(req[0], req[1] + 0.35)
            self.boundary.setPen(pg.mkPen("#c0392b", width=2,
                                          style=QtCore.Qt.DashLine))
        else:
            self.requested.setData([], [])
            self.projection.setData([], [])
            self.proj_text.setText("")
            self.boundary.setPen(pg.mkPen((150, 150, 150), width=1,
                                          style=QtCore.Qt.DashLine))
        self.reachable.setData([rch[0]], [rch[1]])
        self.object.setData([status.object_position.x],
                            [status.object_position.y])
        if pick is not None:
            self.pick_mark.setData([pick[0]], [pick[1]])
        if place is not None:
            self.place_mark.setData([place[0]], [place[1]])


class DemoRunner(QtCore.QObject):
    """
    Plays the fixed test scenario hands-free (gui_node param demo:=true),
    handy for the screen recording. Each step waits until the arm is idle
    and no action is running, plus a pause so the viewer can read the GUI.
    """

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.steps = [
            (3.0, "Test scenario: pick (4.0, 2.0) → place (−3.0, 3.0)",
             win.scenario_pick_place),
            (3.0, "Edge case: move to unreachable (7.0, 3.0)", win.scenario_edge_move),
            (4.0, "Edge case: pick at unreachable (7.0, 3.0)", win.scenario_edge_pick),
            (4.0, "Switching to current (torque) control with PID tracking",
             lambda: win.set_mode("current")),
            (2.0, "Pick & place again in current mode", win.scenario_pick_place),
            (4.0, "Back to position mode, home", lambda: (
                win.set_mode("position"),
                win.ros.call(win.node.request_trigger, "home"))),
        ]
        self.k = 0
        self.t_last = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(250)

    def tick(self):
        w = self.win
        st = w._status
        if st is None or w._action_running or st.state == "moving":
            self.t_last = None
            return
        now = time.monotonic()
        if self.t_last is None:
            self.t_last = now
            return
        if self.k >= len(self.steps):
            w.log_event("demo complete", OK_STYLE)
            self.timer.stop()
            return
        delay, title, fn = self.steps[self.k]
        if now - self.t_last >= delay:
            w.log_event(f"DEMO {self.k + 1}/{len(self.steps)}: {title}")
            fn()
            self.k += 1
            self.t_last = None


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, node, ros, signals):
        super().__init__()
        self.node, self.ros, self.sig = node, ros, signals
        self.setWindowTitle("Sarath's 3-DoF Planar Arm Console")
        self.resize(1500, 950)
        self.history = node.get_parameter("history_seconds").value
        self.render_timing = LoopTimer(1.0 / node.get_parameter("render_rate_hz").value,
                                       window=120)
        self._status = None
        self._pick = None
        self._place = None
        self._action_running = False
        # joint sliders: goal waiting to be sent / request in flight / our
        # joint motion executing. While any is set (or a slider is held) the
        # sliders show the goal; otherwise they follow the measured angles.
        self._joint_pending = None
        self._joint_inflight = False
        self._joint_motion = False
        self._joint_motion_t = 0.0
        self._grip_inflight = False
        self._grip_cmd_t = 0.0
        self._build_ui()

        self.sig.status.connect(self.on_status)
        self.sig.move_response.connect(self.on_move_response)
        self.sig.joints_response.connect(self.on_joints_response)
        self.sig.trigger_response.connect(self.on_trigger_response)
        self.sig.action_event.connect(self.on_action_event)
        self.sig.param_response.connect(self.on_param_response)

        self.render_timer = QtCore.QTimer(self)
        self.render_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.render_timer.timeout.connect(self.render)
        self.render_timer.start(int(1000 / node.get_parameter("render_rate_hz").value))
        self.demo = None
        if node.get_parameter("demo").value:
            self.demo = DemoRunner(self)
            screen = QtWidgets.QApplication.primaryScreen()
            if screen is not None:                 # fill the screen for recording
                self.setGeometry(screen.availableGeometry())

    # ================================================================== UI
    def _build_ui(self):
        self.arm_view = ArmView(self.node.arm)

        # ---- plots
        plots = pg.GraphicsLayoutWidget()
        plots.setBackground("w")
        self.angle_plot = plots.addPlot(row=0, col=0, title="Joint angles [deg] "
                                        "(solid = measured, dashed = reference)")
        self.angle_plot.addLegend(offset=(5, 5))
        self.angle_plot.showGrid(x=True, y=True, alpha=0.3)
        self.err_plot = plots.addPlot(row=1, col=0,
                                      title="Tracking error q_ref − q [deg]")
        self.err_plot.showGrid(x=True, y=True, alpha=0.3)
        self.err_plot.setXLink(self.angle_plot)
        self.err_plot.setLabel("bottom", "time [s]")
        self.q_curves, self.ref_curves, self.err_curves = [], [], []
        for i, c in enumerate(JOINT_COLORS):
            self.q_curves.append(self.angle_plot.plot(
                pen=pg.mkPen(c, width=2), name=f"q{i + 1}"))
            self.ref_curves.append(self.angle_plot.plot(
                pen=pg.mkPen(c, width=1, style=QtCore.Qt.DashLine)))
            self.err_curves.append(self.err_plot.plot(
                pen=pg.mkPen(c, width=2), name=f"e{i + 1}"))

        # ---- side panel
        panel = QtWidgets.QWidget()
        side = QtWidgets.QVBoxLayout(panel)

        self.banner = QtWidgets.QLabel("Waiting for controller…")
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet(WARN_STYLE)
        self.banner.setMinimumHeight(48)
        side.addWidget(self.banner)

        # status / telemetry
        tele = QtWidgets.QGroupBox("Telemetry")
        g = QtWidgets.QGridLayout(tele)
        self.lbl = {}
        rows = [("conn", "Link"), ("state", "State"), ("mode", "Mode / backend"),
                ("q", "Joints [deg]"), ("ee", "End-effector [m]"),
                ("grip", "Gripper / object"), ("loop", "Control loop"),
                ("gui", "GUI render")]
        for r, (k, title) in enumerate(rows):
            g.addWidget(QtWidgets.QLabel(f"<b>{title}</b>"), r, 0)
            self.lbl[k] = QtWidgets.QLabel("—")
            self.lbl[k].setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            g.addWidget(self.lbl[k], r, 1)
        self.motion_bar = QtWidgets.QProgressBar()
        self.motion_bar.setFormat("motion %p%")
        g.addWidget(self.motion_bar, len(rows), 0, 1, 2)
        side.addWidget(tele)

        # control mode
        mode_box = QtWidgets.QGroupBox("Control mode")
        mh = QtWidgets.QHBoxLayout(mode_box)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["position", "velocity", "current"])
        self.mode_combo.activated[str].connect(self.set_mode)
        mh.addWidget(QtWidgets.QLabel("mode:"))
        mh.addWidget(self.mode_combo, 1)
        side.addWidget(mode_box)

        # single move
        move_box = QtWidgets.QGroupBox("Move end-effector")
        mg = QtWidgets.QGridLayout(move_box)
        self.tx, self.ty = _spin(4.0), _spin(2.0)
        mg.addWidget(QtWidgets.QLabel("x"), 0, 0)
        mg.addWidget(self.tx, 0, 1)
        mg.addWidget(QtWidgets.QLabel("y"), 0, 2)
        mg.addWidget(self.ty, 0, 3)
        self.btn_move = QtWidgets.QPushButton("Move")
        self.btn_home = QtWidgets.QPushButton("Home")
        self.btn_stop = QtWidgets.QPushButton("STOP")
        self.btn_stop.setStyleSheet("background:#c0392b;color:white;font-weight:bold;")
        self.btn_move.clicked.connect(self.send_move)
        self.btn_home.clicked.connect(lambda: self.ros.call(self.node.request_trigger, "home"))
        self.btn_stop.clicked.connect(self.send_stop)
        mg.addWidget(self.btn_move, 1, 0, 1, 2)
        mg.addWidget(self.btn_home, 1, 2)
        mg.addWidget(self.btn_stop, 1, 3)
        side.addWidget(move_box)

        # joint + gripper sliders
        self.joint_box = QtWidgets.QGroupBox(
            "Joint control (move on release; dotted arm = goal preview)")
        jg = QtWidgets.QGridLayout(self.joint_box)
        self.joint_sliders, self.joint_vals = [], []
        lims = np.degrees(self.node.arm.joint_limits)
        for i, (lo, hi) in enumerate(lims):
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setRange(int(round(lo * SLIDER_SCALE)), int(round(hi * SLIDER_SCALE)))
            s.setSingleStep(SLIDER_SCALE)            # 1 deg (arrow keys)
            s.setPageStep(10 * SLIDER_SCALE)         # 10 deg (PgUp/PgDn, click)
            s.valueChanged.connect(self._on_joint_slider)
            s.sliderReleased.connect(self._queue_joint_cmd)
            s.setToolTip(f"q{i + 1} limits: {lo:.0f}° … {hi:.0f}°")
            val = QtWidgets.QLabel("—")
            val.setMinimumWidth(56)
            val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            name = QtWidgets.QLabel(f"<b>q{i + 1}</b>")
            name.setStyleSheet(f"color: {JOINT_COLORS[i]};")
            jg.addWidget(name, i, 0)
            jg.addWidget(val, i, 1)
            jg.addWidget(s, i, 2)
            self.joint_sliders.append(s)
            self.joint_vals.append(val)
        # keyboard / wheel / track clicks: send once the value stops changing
        self.joint_debounce = QtCore.QTimer(self)
        self.joint_debounce.setSingleShot(True)
        self.joint_debounce.setInterval(300)
        self.joint_debounce.timeout.connect(self._queue_joint_cmd)

        self.grip_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.grip_slider.setRange(0, 1)
        self.grip_slider.setPageStep(1)
        self.grip_slider.setMaximumWidth(80)
        self.grip_slider.valueChanged.connect(self._on_grip_slider)
        gr = len(lims)
        jg.addWidget(QtWidgets.QLabel("<b>gripper</b>"), gr, 0)
        grip_row = QtWidgets.QHBoxLayout()
        grip_row.addWidget(QtWidgets.QLabel("open"))
        grip_row.addWidget(self.grip_slider)
        grip_row.addWidget(QtWidgets.QLabel("closed"))
        grip_row.addStretch(1)
        jg.addLayout(grip_row, gr, 1, 1, 2)
        jg.setColumnStretch(2, 1)
        side.addWidget(self.joint_box)

        # pick & place
        pp = QtWidgets.QGroupBox("Pick && place (ROS 2 action)")
        pg_ = QtWidgets.QGridLayout(pp)
        self.px, self.py = _spin(4.0), _spin(2.0)
        self.lx, self.ly = _spin(-3.0), _spin(3.0)
        pg_.addWidget(QtWidgets.QLabel("pick x"), 0, 0)
        pg_.addWidget(self.px, 0, 1)
        pg_.addWidget(QtWidgets.QLabel("y"), 0, 2)
        pg_.addWidget(self.py, 0, 3)
        pg_.addWidget(QtWidgets.QLabel("place x"), 1, 0)
        pg_.addWidget(self.lx, 1, 1)
        pg_.addWidget(QtWidgets.QLabel("y"), 1, 2)
        pg_.addWidget(self.ly, 1, 3)
        self.chk_home = QtWidgets.QCheckBox("return home afterwards")
        pg_.addWidget(self.chk_home, 2, 0, 1, 4)
        self.btn_pp = QtWidgets.QPushButton("Run pick && place")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_pp.clicked.connect(self.send_pick_place)
        self.btn_cancel.clicked.connect(lambda: self.ros.call(self.node.request_cancel))
        pg_.addWidget(self.btn_pp, 3, 0, 1, 3)
        pg_.addWidget(self.btn_cancel, 3, 3)
        self.phase_lbl = QtWidgets.QLabel("idle")
        self.phase_bar = QtWidgets.QProgressBar()
        self.phase_bar.setFormat("sequence %p%")
        pg_.addWidget(self.phase_lbl, 4, 0, 1, 4)
        pg_.addWidget(self.phase_bar, 5, 0, 1, 4)
        side.addWidget(pp)

        # fixed test scenario shortcuts
        sc = QtWidgets.QGroupBox("Test scenario")
        sl = QtWidgets.QVBoxLayout(sc)
        b1 = QtWidgets.QPushButton("Pick (4.0, 2.0) → place (−3.0, 3.0)")
        b2 = QtWidgets.QPushButton("Edge case: move to (7.0, 3.0)")
        b3 = QtWidgets.QPushButton("Edge case: pick at (7.0, 3.0)")
        b1.clicked.connect(self.scenario_pick_place)
        b2.clicked.connect(self.scenario_edge_move)
        b3.clicked.connect(self.scenario_edge_pick)
        for b in (b1, b2, b3):
            sl.addWidget(b)
        side.addWidget(sc)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(300)
        side.addWidget(QtWidgets.QLabel("<b>Event log</b>"))
        side.addWidget(self.log, 1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(500)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        left = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        left.addWidget(self.arm_view)
        left.addWidget(plots)
        left.setSizes([520, 380])
        main = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main.addWidget(left)
        main.addWidget(scroll)
        main.setSizes([950, 520])
        self.setCentralWidget(main)

    # ============================================================ helpers
    def log_event(self, text, style=None):
        stamp = time.strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{stamp}] {text}")
        if style is not None:
            self.banner.setText(text)
            self.banner.setStyleSheet(style)

    # ============================================================ actions
    def send_move(self):
        x, y = self.tx.value(), self.ty.value()
        self.log_event(f"→ move_to_target ({x:.2f}, {y:.2f})")
        self.ros.call(self.node.request_move, x, y)

    def set_mode(self, mode):
        self.log_event(f"→ control_mode := {mode}")
        self.ros.call(self.node.request_mode, mode)

    def send_stop(self):
        self.joint_debounce.stop()
        self._joint_pending = None
        self._joint_motion = False
        self.ros.call(self.node.request_cancel)
        self.ros.call(self.node.request_trigger, "stop")

    # ------------------------------------------------------- joint sliders
    def _slider_q(self):
        return np.radians([s.value() / SLIDER_SCALE for s in self.joint_sliders])

    def _sliders_engaged(self):
        """True while the sliders show a goal rather than the measured pose."""
        return (any(s.isSliderDown() for s in self.joint_sliders)
                or self.joint_debounce.isActive()
                or self._joint_pending is not None
                or self._joint_inflight or self._joint_motion)

    def _goal_is_valid(self, q):
        return self.node.arm.arm_above_base(q, tolerance=1e-4)

    def _on_joint_slider(self, _value):
        q = self._slider_q()
        for lbl, v in zip(self.joint_vals, np.degrees(q)):
            lbl.setText(f"{v:.1f}°")
        self.arm_view.set_preview(q, self._goal_is_valid(q))
        if not any(s.isSliderDown() for s in self.joint_sliders):
            self.joint_debounce.start()      # keyboard / wheel / track click

    def _queue_joint_cmd(self):
        self.joint_debounce.stop()
        self._joint_pending = self._slider_q()
        self._flush_joint_cmd()

    def _flush_joint_cmd(self):
        """Send the latest slider goal once the controller is free."""
        st = self._status
        if (self._joint_pending is None or self._joint_inflight
                or self._joint_motion or self._action_running
                or st is None or st.state == "moving"):
            return
        q, self._joint_pending = self._joint_pending, None
        self._joint_inflight = True
        self.log_event("→ move_joints "
                       f"{np.round(np.degrees(q), 1).tolist()} deg")
        self.ros.call(self.node.request_move_joints, q)

    def _sync_sliders(self, q):
        for s, v in zip(self.joint_sliders, np.degrees(q)):
            s.blockSignals(True)
            s.setValue(int(round(v * SLIDER_SCALE)))
            s.blockSignals(False)
        for lbl, v in zip(self.joint_vals, np.degrees(q)):
            lbl.setText(f"{v:.1f}°")

    def on_joints_response(self, resp):
        self._joint_inflight = False
        if isinstance(resp, str):
            self.log_event(resp, ERR_STYLE)
        elif resp.accepted:
            self._joint_motion = True
            self._joint_motion_t = time.monotonic()
            self.log_event(f"✔ joint move ({resp.duration:.2f} s)", OK_STYLE)
        elif resp.message.startswith("busy") and not self._action_running:
            # another motion started first: retry when it is done
            if self._joint_pending is None:
                self._joint_pending = self._slider_q()
            return
        else:
            self.log_event(f"✖ {resp.message}", ERR_STYLE)
        self._flush_joint_cmd()

    def _on_grip_slider(self, value):
        closed = bool(value)
        self._grip_inflight = True
        self._grip_cmd_t = time.monotonic()
        self.log_event(f"→ gripper {'close' if closed else 'open'}")
        self.ros.call(self.node.request_gripper, closed)

    def _set_manual_enabled(self, enabled):
        self.joint_box.setEnabled(enabled)

    def send_pick_place(self):
        self._pick = (self.px.value(), self.py.value())
        self._place = (self.lx.value(), self.ly.value())
        self.log_event(f"→ pick_place pick={self._pick} place={self._place}")
        self.phase_bar.setValue(0)
        self.ros.call(self.node.request_pick_place, self._pick, self._place,
                      self.chk_home.isChecked())

    def scenario_pick_place(self):
        self.px.setValue(4.0)
        self.py.setValue(2.0)
        self.lx.setValue(-3.0)
        self.ly.setValue(3.0)
        self.send_pick_place()

    def scenario_edge_move(self):
        self.tx.setValue(7.0)
        self.ty.setValue(3.0)
        self.send_move()

    def scenario_edge_pick(self):
        self.px.setValue(7.0)
        self.py.setValue(3.0)
        self.lx.setValue(-3.0)
        self.ly.setValue(3.0)
        self.send_pick_place()

    # ============================================================ slots
    def on_status(self, st):
        self._status = st
        if self.mode_combo.currentText() != st.control_mode:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentText(st.control_mode)
            self.mode_combo.blockSignals(False)
        self.lbl["state"].setText(st.state)
        self.lbl["mode"].setText(f"{st.control_mode} / {st.backend}")
        self.lbl["grip"].setText(
            f"{'closed' if st.gripper_closed else 'open'} / "
            f"{'attached' if st.object_attached else 'free'} @ "
            f"({st.object_position.x:.2f}, {st.object_position.y:.2f})")
        self.lbl["loop"].setText(
            f"{st.control_period_mean_ms:.2f} ms (target "
            f"{st.control_period_target_ms:.1f}), p99 jitter "
            f"{st.control_jitter_p99_ms:.2f}, max {st.control_period_max_ms:.1f}")
        self.motion_bar.setValue(int(100 * st.progress))
        self.arm_view.update_targets(st, self._pick, self._place)

        now = time.monotonic()
        if (self._joint_motion and st.state != "moving"
                and now - self._joint_motion_t > STALE_STATUS_S):
            self._joint_motion = False           # slider goal reached / stopped
        self._flush_joint_cmd()
        if (not self._grip_inflight and not self.grip_slider.isSliderDown()
                and now - self._grip_cmd_t > STALE_STATUS_S
                and self.grip_slider.value() != int(st.gripper_closed)):
            self.grip_slider.blockSignals(True)
            self.grip_slider.setValue(int(st.gripper_closed))
            self.grip_slider.blockSignals(False)

    def on_move_response(self, resp):
        if isinstance(resp, str):
            self.log_event(resp, ERR_STYLE)
            return
        if not resp.accepted:
            self.log_event(f"✖ target rejected: {resp.message}", ERR_STYLE)
        elif resp.projected:
            r = resp.reachable_target
            self.log_event(
                f"⚠ Target ({self.tx.value():.2f}, {self.ty.value():.2f}) is outside "
                f"the workspace. Moving to the closest reachable point "
                f"({r.x:.3f}, {r.y:.3f}) on the boundary.", WARN_STYLE)
        else:
            self.log_event(f"✔ moving ({resp.duration:.2f} s): {resp.message}",
                           OK_STYLE)

    def on_trigger_response(self, which, resp):
        if which == "gripper":
            self._grip_inflight = False      # next status re-syncs the slider
            self._grip_cmd_t = time.monotonic()
        if isinstance(resp, str):
            self.log_event(resp, ERR_STYLE)
        else:
            self.log_event(f"{which}: {resp.message}",
                           OK_STYLE if resp.success else ERR_STYLE)

    def on_param_response(self, ok, text):
        self.log_event(text, OK_STYLE if ok else ERR_STYLE)

    def on_action_event(self, kind, payload):
        if kind == "accepted":
            self._action_running = True
            self.btn_cancel.setEnabled(True)
            self.btn_pp.setEnabled(False)
            self._set_manual_enabled(False)
            self.joint_debounce.stop()
            self._joint_pending = None
            self.log_event("pick_place accepted", OK_STYLE)
        elif kind == "rejected":
            self.log_event(f"✖ pick_place: {payload}", ERR_STYLE)
        elif kind == "feedback":
            fb = payload
            self.phase_lbl.setText(
                f"step {fb.step}/{fb.total_steps}: {fb.phase} "
                f"({100 * fb.phase_progress:.0f}%)")
            self.phase_bar.setValue(int(100 * fb.total_progress))
        elif kind == "cancel_sent":
            self.log_event("cancel requested…", WARN_STYLE)
        elif kind == "result":
            status, res = payload
            self._action_running = False
            self.btn_cancel.setEnabled(False)
            self.btn_pp.setEnabled(True)
            self._set_manual_enabled(True)
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.log_event(f"✔ pick & place done in {res.elapsed:.1f} s", OK_STYLE)
            elif status == GoalStatus.STATUS_CANCELED:
                self.log_event(f"pick & place canceled: {res.message}", WARN_STYLE)
                self.phase_lbl.setText("canceled")
            else:
                self.log_event(f"✖ pick & place failed: {res.message}", ERR_STYLE)
                self.phase_lbl.setText("failed")

    # ============================================================ render
    def render(self):
        self.render_timing.start()
        t, q, q_ref, q_now, ref_now, last_rx = self.node.buffer.snapshot()
        connected = last_rx > 0 and (time.monotonic() - last_rx) < 1.0
        self.lbl["conn"].setText("connected" if connected else "NO DATA")
        self.lbl["conn"].setStyleSheet(OK_STYLE if connected else ERR_STYLE)
        if q_now is not None:
            self.arm_view.update_arm(q_now, ref_now)
            d = np.degrees(q_now)
            self.lbl["q"].setText(f"{d[0]:7.2f}  {d[1]:7.2f}  {d[2]:7.2f}")
            ee = self.node.arm.end_effector(q_now)
            self.lbl["ee"].setText(f"x = {ee[0]:.3f}   y = {ee[1]:.3f}")
            if self._sliders_engaged():
                q_goal = self._slider_q()
                self.arm_view.set_preview(q_goal, self._goal_is_valid(q_goal))
            else:
                self._sync_sliders(q_now)    # follow the measured pose
                self.arm_view.set_preview(None)
        if t.size > 1:
            sel = t >= t[-1] - self.history
            tt = t[sel] - t[-1]
            qd, rd = np.degrees(q[sel]), np.degrees(q_ref[sel])
            for i in range(3):
                self.q_curves[i].setData(tt, qd[:, i])
                self.ref_curves[i].setData(tt, rd[:, i])
                self.err_curves[i].setData(tt, rd[:, i] - qd[:, i])
        st = self.render_timing.stats()
        if st["n"]:
            self.lbl["gui"].setText(f"{1000.0 / max(st['mean_ms'], 1e-3):.0f} fps, "
                                    f"p99 jitter {st['p99_jitter_ms']:.1f} ms")
        self.render_timing.stop()

    def shutdown(self):
        self.render_timer.stop()

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.shutdown()
        super().closeEvent(event)
