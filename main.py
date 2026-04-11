import sys
import numpy as np
import nibabel as nib
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog, QLabel, QHBoxLayout
from PyQt6.QtCore import Qt, QTimer
import pyvista as pv
from pyvistaqt import QtInteractor
import keras
from medicai.models import SwinUNETR
from medicai.transforms import Compose, NormalizeIntensity
from medicai.utils.inference import SlidingWindowInference
from skimage import measure

class BrainTumorViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroScan - Brain Tumor Segmentation")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # 3D Viewer (PyVista Qt Widget)
        self.plotter = QtInteractor(self)
        self.plotter.set_background("black")
        layout.addWidget(self.plotter)
        
        # Controls panel
        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        
        btn_load = QPushButton("📂 Load NIfTI Files (FLAIR, T1, T1ce, T2)")
        btn_load.clicked.connect(self.load_files)
        controls_layout.addWidget(btn_load)
        
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        controls_layout.addWidget(self.status_label)
        
        layout.addWidget(controls)
        
        # ⏳ Defer heavy model loading until after window is shown
        QTimer.singleShot(0, self.load_model)

    def load_model(self):
        """Load SwinUNETR model asynchronously"""
        self.status_label.setText("⏳ Loading model weights...")
        QApplication.processEvents()
        try:
            self.model = SwinUNETR(
                encoder_name="swin_tiny_v2",
                input_shape=(96, 96, 96, 4),
                num_classes=3,
                classifier_activation=None
            )
            self.model.load_weights("brats.model.weights.h5")
            self.swi = SlidingWindowInference(
                self.model, num_classes=3, roi_size=(96,96,96),
                sw_batch_size=4, overlap=0.5, mode="gaussian"
            )
            self.status_label.setText("✅ Model loaded. Ready to segment.")
            self.status_label.setStyleSheet("color: #4caf50; font-weight: bold;")
        except Exception as e:
            self.status_label.setText(f"❌ Model load failed: {str(e)[:50]}...")
            self.status_label.setStyleSheet("color: #f44336; font-weight: bold;")
            import traceback
            traceback.print_exc()

    def load_files(self):
        """Open file dialog for 4 modalities"""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select 4 NIfTI files (FLAIR, T1, T1ce, T2)", "", "NIfTI Files (*.nii *.nii.gz)"
        )
        if len(files) != 4:
            self.status_label.setText("❌ Please select exactly 4 files")
            self.status_label.setStyleSheet("color: #f44336;")
            return
            
        self.status_label.setText("⏳ Processing & inferring...")
        self.status_label.setStyleSheet("color: orange;")
        QApplication.processEvents()
        
        try:
            # 1️⃣ Load & stack volumes
            imgs = [nib.load(f).get_fdata().astype(np.float32) for f in files]
            vol = np.stack(imgs, axis=-1)  # Shape: (H, W, D, 4)
            
            # 2️⃣ Preprocess (Matches Notebook API)
            dummy_label = np.zeros(vol.shape[:3], dtype=np.float32)
            meta = {"affine": np.eye(4)}  # Dummy affine matrix
            pipeline = Compose([NormalizeIntensity(keys=["image"], nonzero=True, channel_wise=True)])
            
            # ✅ FIX: Pass 'meta' as the 2nd positional argument
            proc_result = pipeline({"image": vol, "label": dummy_label}, meta)
            proc_vol = proc_result["image"]
            
            # 3️⃣ Inference
            logits = self.swi(proc_vol[None, ...])
            probs = keras.ops.sigmoid(logits).numpy().squeeze()
            mask = (probs > 0.5).astype(np.uint8)
            
            # 4️⃣ Clear & rebuild 3D scene
            self.plotter.clear()
            
            # Brain surface (translucent)
            brain_mask = vol.mean(axis=-1) > np.percentile(vol.mean(axis=-1), 10)
            if brain_mask.sum() > 0:
                brain_mesh = self.create_mesh(brain_mask)
                self.plotter.add_mesh(brain_mesh, color="gray", opacity=0.25, name="brain")
            
            # Tumor regions
            colors = {"tc": "red", "wt": "dodgerblue", "et": "limegreen"}
            opacities = {"tc": 0.9, "wt": 0.7, "et": 1.0}
            
            for i, key in enumerate(["tc", "wt", "et"]):
                if mask[..., i].sum() > 0:
                    tumor_mesh = self.create_mesh(mask[..., i])
                    self.plotter.add_mesh(tumor_mesh, color=colors[key], opacity=opacities[key], name=key)
            
            self.plotter.reset_camera()
            self.status_label.setText("✅ Segmentation complete. Interact with mouse.")
            self.status_label.setStyleSheet("color: #4caf50;")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_label.setText(f"❌ Error: {str(e)[:60]}...")
            self.status_label.setStyleSheet("color: #f44336;")

    def create_mesh(self, mask):
        """Convert binary mask to PyVista PolyData mesh"""
        if mask.sum() == 0:
            return None
        verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
        # PyVista expects faces as [n, v1, v2, v3, ...] format
        faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten()
        mesh = pv.PolyData(verts, faces_pv)
        mesh.compute_normals(inplace=True)
        return mesh

if __name__ == "__main__":
    # Set Keras backend before importing medicai/keras models
    import os
    os.environ["KERAS_BACKEND"] = "tensorflow"
    
    app = QApplication(sys.argv)
    window = BrainTumorViewer()
    window.show()
    sys.exit(app.exec())