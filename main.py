import sys
import numpy as np
import nibabel as nib
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QPushButton, QFileDialog, QLabel, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QColor, QPalette
import pyvista as pv
from pyvistaqt import QtInteractor
import keras
from medicai.models import SwinUNETR
from medicai.transforms import Compose, NormalizeIntensity
from medicai.utils.inference import SlidingWindowInference
from skimage import measure


# ── Clinical palette ──────────────────────────────────────────────────────────
CLR_BG          = "#FFFFFF"   
CLR_SURFACE     = "#FFFFFF"   
CLR_BORDER      = "#FFFFFF"   
CLR_HEADER_BG   = "#FFFFFF"   
CLR_HEADER_TXT  = "#010E7E"
CLR_BTN_BG      = "#1A5F8A"   
CLR_BTN_HOVER   = "#145075"
CLR_BTN_TXT     = "#FFFFFF"
CLR_TXT_PRIMARY = "#1A2733"
CLR_TXT_MUTED   = "#5A6A7A"
CLR_SUCCESS     = "#1D6F42"
CLR_WARNING     = "#7A4F00"
CLR_DANGER      = "#8C1C13"
CLR_INFO_BG     = "#EBF4FA"
CLR_SUCCESS_BG  = "#FFFFFF"
CLR_WARNING_BG  = "#FFFFFF"
CLR_DANGER_BG   = "#FFFFFF"


def make_separator():
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet(f"color: {CLR_BORDER};")
    return line


class HeaderBar(QWidget):
    """Deep-navy top bar — institution branding + app title."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(56)
        self.setStyleSheet(f"background-color: {CLR_HEADER_BG};")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)

        # Left: logo placeholder + title
        title = QLabel("NeuroScan")
        title.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {CLR_HEADER_TXT}; letter-spacing: 1px;")

        subtitle = QLabel("Brain Tumor Segmentation  ·  v1.0")
        subtitle.setFont(QFont("Segoe UI", 14))
        subtitle.setStyleSheet(f"color: {CLR_HEADER_TXT};")

        vbox = QVBoxLayout()
        vbox.setSpacing(0)
        vbox.addWidget(title)
        vbox.addWidget(subtitle)

        layout.addLayout(vbox)
        layout.addStretch()


class StatusBadge(QLabel):
    """Pill-shaped status indicator with semantic colouring."""

    STATES = {
        "idle":       ("#5A6A7A", "#EDF1F5",  "●Idle"),
        "loading":    ("#7A4F00", "#FFF8EC",  "Loading…"),
        "processing": ("#7A4F00", "#FFF8EC",  "Processing…"),
        "ready":      ("#1D6F42", "#EAF4EE",  "Ready"),
        "success":    ("#1D6F42", "#EAF4EE",  "Segmentation complete"),
        "error":      ("#8C1C13", "#FAEAEA",  "Error"),
        "warn":       ("#7A4F00", "#FFF8EC",  "Warning"),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFont(QFont("Segoe UI", 10))
        self.set_state("idle")

    def set_state(self, state: str, custom_text: str = ""):
        fg, bg, default_text = self.STATES.get(state, self.STATES["idle"])
        text = custom_text if custom_text else default_text
        self.setText(text)
        self.setStyleSheet(
            f"color: {fg}; background: {bg}; border-radius: 14px;"
            f" padding: 0 14px; font-weight: 600;"
        )


class ToolbarPanel(QWidget):
    """White bottom toolbar with action button + status badge."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(68)
        self.setStyleSheet(
            f"background: {CLR_SURFACE}; border-top: 1px solid {CLR_BORDER};"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(16)

        self.btn_load = QPushButton("  Load NIfTI Files")
        self.btn_load.setFixedHeight(38)
        self.btn_load.setMinimumWidth(210)
        self.btn_load.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
        self.btn_load.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_load.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {CLR_BTN_BG};
                color: {CLR_BTN_TXT};
                border: none;
                border-radius: 6px;
                padding: 0 20px;
                font-size: 10pt;
            }}
            QPushButton:hover  {{ background-color: {CLR_BTN_HOVER}; }}
            QPushButton:pressed {{ background-color: #0E3A55; }}
            QPushButton:disabled {{ background-color: #9AAFBF; }}
            """
        )

        self.status = StatusBadge()

        layout.addWidget(self.btn_load)
        layout.addWidget(self.status)
        layout.addStretch()

        # Legend
        legend_items = [
            ("Tumour Core",    "#CC2200"),
            ("Edema",   "#1565C0"),
            ("Enhancing",      "#2E7D32"),
            ("Brain (ref.)",        "#9E9E9E"),
        ]
        for label_text, color in legend_items:
            dot = QLabel("●")
            dot.setStyleSheet(f"color: {color}; font-size: 14px;")
            lbl = QLabel(label_text)
            lbl.setFont(QFont("Segoe UI", 9))
            lbl.setStyleSheet(f"color: {CLR_TXT_MUTED};")
            layout.addWidget(dot)
            layout.addWidget(lbl)
            layout.addSpacing(10)


class BrainTumorViewer(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroScan — Brain Tumor Segmentation")
        self.setGeometry(100, 100, 1440, 920)
        self._apply_app_palette()

        # ── Root layout ──────────────────────────────────────────────────────
        root = QWidget()
        root.setStyleSheet(f"background: {CLR_BG};")
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Header
        self.header = HeaderBar()
        root_layout.addWidget(self.header)

        # Viewport card
        viewport_wrapper = QWidget()
        viewport_wrapper.setStyleSheet(
            f"background: {CLR_SURFACE}; border-bottom: 1px solid {CLR_BORDER};"
        )
        vw_layout = QVBoxLayout(viewport_wrapper)
        vw_layout.setContentsMargins(12, 12, 12, 12)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("#F0F4F8")       # light clinical grey-blue
        self.plotter.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        vw_layout.addWidget(self.plotter)
        root_layout.addWidget(viewport_wrapper, stretch=1)

        # Toolbar
        self.toolbar = ToolbarPanel()
        self.toolbar.btn_load.clicked.connect(self.load_files)
        root_layout.addWidget(self.toolbar)

        QTimer.singleShot(0, self.load_model)

    # ── Palette ───────────────────────────────────────────────────────────────

    def _apply_app_palette(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window,      QColor(CLR_BG))
        palette.setColor(QPalette.ColorRole.WindowText,  QColor(CLR_TXT_PRIMARY))
        palette.setColor(QPalette.ColorRole.Base,        QColor(CLR_SURFACE))
        palette.setColor(QPalette.ColorRole.Text,        QColor(CLR_TXT_PRIMARY))
        QApplication.instance().setPalette(palette)

    # ── Model ─────────────────────────────────────────────────────────────────

    def load_model(self):
        self.toolbar.btn_load.setEnabled(False)
        self.toolbar.status.set_state("loading", "⏳  Loading model weights…")
        QApplication.processEvents()
        try:
            self.model = SwinUNETR(
                encoder_name="swin_tiny_v2",
                input_shape=(96, 96, 96, 4),
                num_classes=3,
                classifier_activation=None,
            )
            self.model.load_weights("brats.model.weights.h5")
            self.swi = SlidingWindowInference(
                self.model, num_classes=3, roi_size=(96, 96, 96),
                sw_batch_size=4, overlap=0.5, mode="gaussian",
            )
            self.toolbar.status.set_state("ready")
            self.toolbar.btn_load.setEnabled(True)
        except Exception as e:
            self.toolbar.status.set_state("error", f"✖  {str(e)[:60]}…")
            import traceback; traceback.print_exc()

    # ── File loading ──────────────────────────────────────────────────────────

    def load_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select 4 NIfTI modality files (FLAIR · T1 · T1ce · T2)",
            "",
            "NIfTI Files (*.nii *.nii.gz)",
        )
        if len(files) != 4:
            self.toolbar.status.set_state(
                "warn", f"⚠  Select exactly 4 files (got {len(files)})"
            )
            return

        self.toolbar.btn_load.setEnabled(False)
        self.toolbar.status.set_state("processing")
        QApplication.processEvents()

        try:
            imgs = [nib.load(f).get_fdata().astype(np.float32) for f in files]
            vol  = np.stack(imgs, axis=-1)

            dummy_label = np.zeros(vol.shape[:3], dtype=np.float32)
            meta        = {"affine": np.eye(4)}
            pipeline    = Compose([
                NormalizeIntensity(keys=["image"], nonzero=True, channel_wise=True)
            ])

            proc_result = pipeline({"image": vol, "label": dummy_label}, meta)
            proc_vol    = proc_result["image"]

            logits = self.swi(proc_vol[None, ...])
            probs  = keras.ops.sigmoid(logits).numpy().squeeze()
            mask   = (probs > 0.5).astype(np.uint8)

            self.plotter.clear()

            # Semi-transparent brain reference mesh
            brain_mask = vol.mean(axis=-1) > np.percentile(vol.mean(axis=-1), 10)
            if brain_mask.sum() > 0:
                brain_mesh = self._create_mesh(brain_mask)
                if brain_mesh:
                    self.plotter.add_mesh(
                        brain_mesh, color="#F990D1", opacity=0.18, name="brain"
                    )

            # Tumour regions — clinical colouring
            regions = {
                "tc": ("#FC2A008B", 0.65),   # Tumour Core  — deep red
                "wt": ("#1565C0", 0.50),   # Whole Tumour — clinical blue
                "et": ("#2E7D32", 1.00),   # Enhancing    — clinical green
            }
            for i, (key, (color, opacity)) in enumerate(regions.items()):
                if mask[..., i].sum() > 0:
                    tm = self._create_mesh(mask[..., i])
                    if tm:
                        self.plotter.add_mesh(tm, color=color, opacity=opacity, name=key)

            self.plotter.reset_camera()
            self.toolbar.status.set_state("success")

        except Exception as e:
            import traceback; traceback.print_exc()
            self.toolbar.status.set_state("error", f"✖  {str(e)[:60]}…")

        finally:
            self.toolbar.btn_load.setEnabled(True)

    # ── Mesh helper ───────────────────────────────────────────────────────────

    def _create_mesh(self, mask):
        if mask.sum() == 0:
            return None
        verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
        faces_pv = np.hstack(
            [np.full((faces.shape[0], 1), 3), faces]
        ).flatten()
        mesh = pv.PolyData(verts, faces_pv)
        mesh.compute_normals(inplace=True)
        return mesh


if __name__ == "__main__":
    import os
    os.environ["KERAS_BACKEND"] = "tensorflow"

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))

    window = BrainTumorViewer()
    window.show()
    sys.exit(app.exec())