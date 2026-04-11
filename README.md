# NeuroScan: AI-Powered 3D Brain Tumor Segmentation

## Complete Technical & User Documentation

---

## Project Overview

**NeuroScan** is a desktop application that leverages deep learning to automatically segment brain tumors from multi-modal MRI scans. Built on the **BraTS 2020 dataset** and a **SwinUNETR transformer architecture**, it processes clinical-grade NIfTI volumes and renders interactive 3D visualizations of tumor sub-regions (Whole Tumor, Tumor Core, Enhancing Tumor) overlaid on a translucent brain surface.

Designed for researchers, clinicians, and developers, NeuroScan runs entirely locally, requires no cloud dependency, and bridges cutting-edge medical AI with intuitive desktop interaction.

---

## 1. Dataset & Medical Imaging Fundamentals

### 1.1 The BraTS 2020 Dataset

- **Full Name:** Multimodal Brain Tumor Segmentation Challenge 2020
- **Source:** MICCAI BraTS Initiative (https://www.med.upenn.edu/cbica/brats2020/)
- **Subjects:** ~350+ glioma patients with expert-radiologist annotations
- **Format:** Raw data distributed as TFRecord shards (training) and NIfTI volumes (clinical/inference)
- **Preprocessing Standard:**
  - Skull-stripped & co-registered to a common 1×1×1 mm³ anatomical template
  - Linear intensity normalization applied
  - Aligned across all four MRI modalities

### 1.2 What are `.nii` / `.nii.gz` Files?

NIfTI (**Neuroimaging Informatics Technology Initiative**) is the standard file format for medical volumetric imaging (MRI, CT, fMRI).

| Property            | Description                                                                                                                                   |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Extension**       | `.nii` (uncompressed) or `.nii.gz` (gzip-compressed)                                                                                          |
| **Data Structure**  | 3D or 4D array of voxels (volume pixels)                                                                                                      |
| **Metadata Header** | Stores affine transformation matrix, voxel dimensions, orientation (RAS/LPS), data type, and acquisition parameters                           |
| **Why Used**        | Preserves spatial relationships, supports multi-channel volumes, universally compatible with medical software (ITK-SNAP, 3D Slicer, FSL, SPM) |

In NeuroScan, `nibabel` parses the header + data, extracts the voxel grid, and converts it to a NumPy array for AI processing.

### 1.3 MRI Modalities: FLAIR, T1, T1ce, T2

Gliomas exhibit different contrast behaviors across MRI sequences. The model ingests all four as separate channels to capture complementary biological information:

| Modality  | Full Name                           | Clinical Role                                                             | Appearance of Tumor                                                            |
| --------- | ----------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **FLAIR** | Fluid Attenuated Inversion Recovery | Suppresses cerebrospinal fluid (CSF) signal; highlights peritumoral edema | Hyperintense (bright) around tumor; critical for **Whole Tumor (WT)** boundary |
| **T1**    | T1-weighted                         | Anatomical reference; shows gray/white matter differentiation             | Hypointense (dark) in tumor region; provides structural context                |
| **T1ce**  | T1 Contrast-Enhanced                | Gadolinium contrast highlights blood-brain barrier breakdown              | Bright enhancing rim; gold standard for **Enhancing Tumor (ET)**               |
| **T2**    | T2-weighted                         | Sensitive to water content; shows edema & cystic components               | Hyperintense; outlines full tumor extent including non-enhancing regions       |

**Model Input:** `(Depth, Height, Width, 4)` → Channels ordered as `[FLAIR, T1, T1ce, T2]`

### 1.4 Label Encoding & Tumor Sub-Regions

BraTS uses a 4-class label map: `0=Background, 1=Necrotic/Cystic, 2=Edema, 4=Enhancing Tumor`. The model predicts 3 binary channels:

| Channel                  | Definition                           | BraTS Mapping              |
| ------------------------ | ------------------------------------ | -------------------------- |
| **TC** (Tumor Core)      | Necrotic + Enhancing regions         | `label == 1 OR label == 4` |
| **WT** (Whole Tumor)     | All tumor regions combined           | `TC OR label == 2`         |
| **ET** (Enhancing Tumor) | Actively growing, vascularized tumor | `label == 4`               |

---

## 2. Model Architecture & Training Pipeline

### 2.1 Architecture: SwinUNETR

A hybrid **Swin Transformer + U-Net** designed for 3D medical segmentation.

| Component            | Specification                                                       |
| -------------------- | ------------------------------------------------------------------- |
| **Encoder**          | `SwinTinyV2` 3D (hierarchical shifted-window self-attention)        |
| **Patch Size**       | 2×2×2 (reduces spatial dims by 8× while expanding channels)         |
| **Window Size**      | 7×7×7 (local attention within cubic windows)                        |
| **Feature Sizes**    | 48 → 96 → 192 → 384 → 768 (progressive downsampling)                |
| **Decoder**          | UNETR-style transposed convolutions + skip connections from encoder |
| **Output Head**      | 1×1×1 conv → 3-channel logits (no activation; raw scores)           |
| **Input Shape**      | `(96, 96, 96, 4)` volumetric patches                                |
| **Total Parameters** | ~186M (710 MB weights, ~237 MB trainable)                           |

### 2.2 Training Configuration (From Notebook)

```python
keras.mixed_precision.set_global_policy("mixed_float16")
model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-5),
    loss=BinaryDiceCELoss(from_logits=True, num_classes=3),
    metrics=[BinaryDiceMetric(from_logits=True, ignore_empty=True, num_classes=3)]
)
```

- **Loss:** Combined Binary Dice + Cross-Entropy (handles class imbalance)
- **Optimizer:** AdamW (decoupled weight decay improves generalization)
- **Precision:** `mixed_float16` reduces VRAM usage, speeds up matrix ops
- **Epochs:** 4 (demonstration; production typically uses 50-100+)
- **Validation:** Sliding window inference on held-out shard every 2 epochs

### 2.3 Inference Pipeline

1. **Sliding Window:** `SlidingWindowInference` with `roi_size=(96,96,96)`, `overlap=0.5`, `mode="gaussian"`
2. **Aggregation:** Gaussian weighting prevents boundary artifacts
3. **Activation:** `sigmoid(logits) > 0.5` → binary masks
4. **Post-processing:** Optional morphological cleanup (not applied in prototype; raw model output used)

### 2.4 Reported Performance (BraTS 2020 Subset)

| Metric                   | Dice Score |
| ------------------------ | ---------- |
| **Whole Tumor (WT)**     | 0.9184     |
| **Enhancing Tumor (ET)** | 0.8447     |
| **Tumor Core (TC)**      | 0.6186     |
| **Overall Mean**         | 0.7939     |

_Note: TC is inherently harder due to heterogeneous necrosis/cystic regions. Clinical deployment requires full BraTS training + external validation._

---

## 3. Desktop Application Architecture

### 3.1 Technology Stack

| Layer            | Technology                    | Purpose                                         |
| ---------------- | ----------------------------- | ----------------------------------------------- |
| **UI Framework** | PyQt6                         | Native cross-platform desktop interface         |
| **3D Rendering** | PyVista + QtInteractor        | VTK-based OpenGL viewer embedded in Qt          |
| **Medical I/O**  | nibabel                       | Parse NIfTI headers, extract voxel grids        |
| **AI Engine**    | TensorFlow 2.19 + Keras 3     | SwinUNETR inference                             |
| **Transforms**   | `medic-ai`                    | Preprocessing pipeline matching training        |
| **Mesh Gen**     | scikit-image `marching_cubes` | Convert binary masks → triangle meshes          |
| **Async UI**     | `QTimer.singleShot`           | Defer heavy model loading to keep UI responsive |

### 3.2 System Flow Diagram

```
[User] → Select 4 NIfTI files
   ↓
[nibabel] → Load volumes → Stack (D,H,W,4)
   ↓
[medicai.transforms] → NormalizeIntensity (channel-wise, nonzero)
   ↓
[SlidingWindowInference] → Forward pass → (D,H,W,3) logits
   ↓
[sigmoid + threshold] → Binary masks (TC, WT, ET)
   ↓
[skimage.measure.marching_cubes] → Vertices + Faces
   ↓
[PyVista.PolyData] → Compute normals → Attach colors/opacities
   ↓
[QtInteractor] → Interactive 3D viewport (rotate/pan/zoom)
```

### 3.3 Key Design Decisions

- **Local-First:** No cloud dependency; HIPAA-friendly data handling
- **Deferred Initialization:** `QTimer.singleShot(0, load_model)` prevents UI freeze on startup
- **Memory Efficiency:** Sliding window inference processes volumes in patches; only final meshes reside in VRAM
- **Clinical Visualization:** Translucent brain (25% opacity) + solid tumor regions preserves anatomical context
- **Error Boundaries:** Try/except blocks with user-friendly status feedback

---

## 4. Installation & Environment Setup

### 4.1 System Requirements

| Component   | Minimum                     | Recommended                          |
| ----------- | --------------------------- | ------------------------------------ |
| **OS**      | Windows 10/11, Linux, macOS | Windows 11 / Ubuntu 22.04            |
| **Python**  | 3.10 – 3.12                 | 3.11                                 |
| **RAM**     | 8 GB                        | 16+ GB                               |
| **Storage** | 2 GB free                   | SSD recommended                      |
| **GPU**     | CPU-only works              | NVIDIA RTX 3060+ (via WSL2/DirectML) |

### 4.2 Step-by-Step Setup

```bash
# 1. Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place model weights in project root
cp /path/to/brats.model.weights.h5 ./brats.model.weights.h5

# 4. Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import pyvista; print(pyvista.__version__)"
```

### 4.3 `requirements.txt`

```text
PyQt6>=6.6.0
pyvista>=0.43.0
pyvistaqt>=0.11.0
vtk>=9.3.0
scikit-image>=0.23.0
nibabel>=5.2.0
numpy>=1.26.0,<2.1.0
tensorflow==2.19.0
keras==3.13.2
medic-ai @ git+https://github.com/innat/medic-ai.git
pillow>=10.0.0
```

### 4.4 GPU Acceleration (Optional)

Native Windows TensorFlow lacks CUDA support. Options:

1. **WSL2 (Recommended):** Install Ubuntu on WSL2, follow Linux GPU guide
2. **DirectML Plugin:** `pip install tensorflow-directml` (experimental, works on AMD/NVIDIA)
3. **CPU Fallback:** Fully functional; inference takes ~1-3 minutes per scan on modern CPUs

---

## 5. 📘 Comprehensive User Guide

### 5.1 Launching the Application

```bash
python main.py
```

- A window titled `NeuroScan - Brain Tumor Segmentation` opens
- Status bar shows `Loading model weights...`
- After ~5-10 seconds: `Model loaded. Ready to segment.`

### 5.2 Loading MRI Data

1. Click **Load NIfTI Files (FLAIR, T1, T1ce, T2)**
2. In the file dialog, **select exactly 4 `.nii` or `.nii.gz` files**
3. Hold `Ctrl` (Windows/Linux) or `Cmd` (macOS) to multi-select
4. Click **Open**

**File Order Matters:** The pipeline expects `[FLAIR, T1, T1ce, T2]` in the order selected. Mismatched order will degrade segmentation quality.

### 5.3 Processing & Visualization

- Status changes to `Processing & inferring...`
- CPU inference typically takes **60-180 seconds**
- Upon completion: `Segmentation complete. Interact with mouse.`

**3D Viewport Controls:**
| Action | Mouse Input |
|--------|-------------|
| **Rotate** | Left-click + drag |
| **Pan** | Right-click + drag |
| **Zoom** | Scroll wheel |
| **Reset View** | Press `R` or use app button (if implemented) |

### 5.4 Understanding the Output

| Color                       | Region               | Clinical Meaning                                    |
| --------------------------- | -------------------- | --------------------------------------------------- |
| **Gray (25% opacity)**      | Brain Surface        | Anatomical reference; skull-stripped cortex         |
| 🔵 **Blue (70% opacity)**   | Whole Tumor (WT)     | Complete tumor extent including edema               |
| 🔴 **Red (90% opacity)**    | Tumor Core (TC)      | Necrotic + enhancing regions; surgical target       |
| 🟢 **Green (100% opacity)** | Enhancing Tumor (ET) | Active, vascularized tumor; biopsy/radiation target |

### 5.5 Best Practices

- Use preprocessed BraTS-format NIfTI files (co-registered, skull-stripped)
- Ensure all 4 modalities cover the same spatial extent
- Verify file names match modality order before selection
- Raw clinical DICOMs require preprocessing (HD-BET, registration, resampling) before use
- For quantitative analysis, export masks or use ITK-SNAP alongside NeuroScan

---

## 6. 🔧 Technical Reference & Troubleshooting

### 6.1 Common Errors & Fixes

| Error                                                                     | Cause                     | Solution                                                           |
| ------------------------------------------------------------------------- | ------------------------- | ------------------------------------------------------------------ |
| `FileNotFoundError: brats.model.weights.h5`                               | Weights missing from root | Place `brats.model.weights.h5` in same directory as `main.py`      |
| `Please select exactly 4 files`                                           | Wrong file count          | Select exactly 4 NIfTI files                                       |
| `AttributeError: 'NoneType' object has no attribute 'get_fdata'`          | Corrupted/empty NIfTI     | Verify files open in ITK-SNAP or 3D Slicer                         |
| `MemoryError` during inference                                            | Insufficient RAM/CPU swap | Close other apps; consider GPU acceleration                        |
| `TypeError: Compose.__call__() got an unexpected keyword argument 'meta'` | Outdated medic-ai API     | Reinstall: `pip install git+https://github.com/innat/medic-ai.git` |

### 6.2 Performance Optimization

- **SSD Storage:** Reduces NIfTI load time by 3-5×
- **Background Apps:** Close browsers/IDEs to free RAM
- **Patch Size:** Modifying `roi_size` in `SlidingWindowInference` trades speed vs. memory
- **TensorRT/ONNX:** Export model for 2-4× inference speedup (advanced)

### 6.3 Future Development Roadmap

- 2D orthogonal slice viewers (axial/coronal/sagittal)
- DICOM import + automated preprocessing pipeline
- Tumor volume quantification & export (CSV/NIfTI mask)
- Multi-subject batch processing
- HIPAA-compliant audit logging & data encryption
- Cloud sync & collaborative annotation

---

## Appendix: Project Structure

```
NeuroScan/
├── main.py                  # PyQt6 desktop application entry point
├── brats.model.weights.h5   # Trained SwinUNETR weights (710 MB)
├── requirements.txt         # Python dependencies
├── README.md                # This documentation
├── .gitignore               # Excludes venv, cache, large files
```

---

## Disclaimer & Clinical Use Notice

NeuroScan is a **research prototype** built on the BraTS 2020 dataset. It has not undergone FDA/CE certification or clinical validation. **Do not use for diagnostic or treatment decisions.** For clinical deployment, integrate with certified PACS, implement rigorous validation pipelines, and follow local regulatory guidelines.

---

**Developed with:** TensorFlow 2.19, Keras 3.13, PyQt6, PyVista, medic-ai, scikit-image, nibabel  
**Dataset:** BraTS 2020 (UPenn/CBICA)  
**License:** MIT / Research Use Only

_For technical support, bug reports, or contribution guidelines, refer to the repository issues page._
