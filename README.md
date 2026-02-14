# CLAHE - Contrast Limited Adaptive Histogram Equalization

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.0%2B-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Group 4 - DCIT 407 Image Processing Project**

A standalone implementation of Contrast Limited Adaptive Histogram Equalization (CLAHE) for advanced image enhancement with automated parameter testing and comprehensive metrics reporting.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Parameters](#parameters)
- [Output & Metrics](#output--metrics)
- [How CLAHE Works](#how-clahe-works)
- [Team](#team)

---

## 🎯 Overview

CLAHE (Contrast Limited Adaptive Histogram Equalization) is an advanced image enhancement technique that improves local contrast while preventing noise amplification. Unlike global histogram equalization, CLAHE:

- ✅ Divides images into tiles and applies histogram equalization locally
- ✅ Limits contrast amplification to prevent over-enhancement
- ✅ Preserves fine details and natural appearance
- ✅ Works with multiple color spaces (LAB, HSV, YCrCb)

---

## ✨ Features

- **Standalone Script**: Single-file execution with no external dependencies beyond OpenCV
- **Multiple Color Spaces**: Support for LAB, HSV, and YCrCb color spaces
- **Grayscale Processing**: Optional grayscale conversion for specialized use cases
- **Parameter Testing**: Automated testing with multiple parameter combinations
- **Batch Processing**: Process entire directories of images at once
- **Comprehensive Metrics**: Automatic calculation of PSNR, entropy, contrast, sharpness, and brightness
- **JSON Export**: All results and metrics exported to structured JSON format
- **Visual Feedback**: Optional side-by-side comparison display

---

## 📁 Project Structure

```
Group_4_CLAHE_Adaptive_Histogram_Equalization/
├── src/
│   └── clahe.py              # Main CLAHE implementation (standalone)
├── data/
│   ├── input/                # Place your input images here
│   └── output/               # Enhanced images saved here
├── group4/                   # Virtual environment
├── metrics.json              # Generated metrics and results
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/DCIT407-IMAGE-PROCESSING/Group_4_CLAHE_Adaptive_Histogram_Equalization.git
cd Group_4_CLAHE_Adaptive_Histogram_Equalization
```

### Step 2: Activate Virtual Environment

**Windows:**

```bash
.\group4\Scripts\Activate
```

**Linux/Mac:**

```bash
source group4/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**

- `opencv-python` - Image processing library
- `numpy` - Numerical operations

### Step 4: Verify Installation

```bash
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

---

## ⚡ Quick Start

### 1. Add Your Images

Place your images in the `data/input/` directory:

```bash
data/input/
├── image1.jpg
├── image2.png
└── photo.bmp
```

### 2. Run CLAHE (Basic)

```bash
python src/clahe.py --input data/input/image.jpg --show
```

This will:

- Enhance `image.jpg` using default CLAHE parameters
- Save output to `data/output/`
- Display before/after comparison
- Generate `metrics.json` with quality metrics

---

## 📖 Usage Examples

### Example 1: Single Image with Custom Parameters

```bash
python src/clahe.py --input data/input/image.jpg --clip-limit 3.0 --tile-size 16 --color-space HSV --show
```

**Output:**

- `data/output/image_clahe_clip3.0_tile16_HSV.png`

### Example 2: Grayscale Processing

```bash
python src/clahe.py --input data/input/image.jpg --grayscale --show
```

**Output:**

- `data/output/image_clahe_clip2.0_tile8_gray.png`

### Example 3: Batch Processing (All Images)

```bash
python src/clahe.py
```

This processes all images in `data/input/` with default parameters.

### Example 4: Multi-Parameter Testing

```bash
python src/clahe.py --test-params
```

This tests each image with **5 different configurations**:

1. `clip=2.0, tile=8, LAB` (default)
2. `clip=3.0, tile=16, HSV` (aggressive)
3. `clip=2.0, tile=8, YCrCb` (video-style)
4. `clip=4.0, tile=8, LAB` (high contrast)
5. `clip=2.0, tile=8, grayscale` (monochrome)

### Example 5: Custom Metrics Output Path

```bash
python src/clahe.py --input data/input/image.jpg --metrics-output results.json
```

---

## ⚙️ Parameters

### Command-Line Arguments

| Argument           | Type   | Default        | Description                                 |
| ------------------ | ------ | -------------- | ------------------------------------------- |
| `--input`          | string | None           | Path to single input image                  |
| `--input-dir`      | string | `data/input`   | Directory for batch processing              |
| `--output`         | string | `data/output`  | Output directory for enhanced images        |
| `--clip-limit`     | float  | `2.0`          | Contrast limiting threshold (1.0-4.0)       |
| `--tile-size`      | int    | `8`            | Grid size for local equalization (4, 8, 16) |
| `--color-space`    | string | `LAB`          | Color space: `LAB`, `HSV`, or `YCrCb`       |
| `--grayscale`      | flag   | `False`        | Convert to grayscale before processing      |
| `--show`           | flag   | `False`        | Display before/after comparison             |
| `--test-params`    | flag   | `False`        | Test multiple parameter combinations        |
| `--metrics-output` | string | `metrics.json` | Path to save metrics JSON file              |

### Parameter Guidelines

#### Clip Limit

- **1.0-2.0**: Subtle, natural enhancement
- **2.0-3.0**: Moderate contrast boost (recommended)
- **3.0-4.0**: Aggressive enhancement (risk of over-processing)

#### Tile Size

- **4x4**: Very local adaptation (may create blocking artifacts)
- **8x8**: Balanced local/global enhancement (recommended)
- **16x16**: Smoother transitions, more global effect

#### Color Space

- **LAB**: Best for natural images and portraits (perceptually uniform)
- **HSV**: Good for colorful scenes (preserves hue)
- **YCrCb**: Ideal for video processing (broadcast standard)

---

## 📊 Output & Metrics

### Enhanced Images

Output files are saved with descriptive names:

```
data/output/
├── image_clahe_clip2.0_tile8_LAB.png
├── image_clahe_clip3.0_tile16_HSV.png
└── image_clahe_clip2.0_tile8_gray.png
```

### Metrics JSON

The `metrics.json` file contains:

```json
{
  "total_images_processed": 5,
  "timestamp": "2026-02-14 15:30:45",
  "results": [
    {
      "input_path": "data/input/image.jpg",
      "output_path": "data/output/image_clahe_clip2.0_tile8_LAB.png",
      "parameters": {
        "clip_limit": 2.0,
        "tile_size": 8,
        "color_space": "LAB",
        "grayscale": false
      },
      "metrics": {
        "entropy": 7.245,
        "contrast": 45.32,
        "brightness": 128.67,
        "sharpness": 342.18,
        "psnr": 28.45,
        "width": 1920,
        "height": 1080,
        "channels": 3
      },
      "processing_time_seconds": 0.156
    }
  ],
  "average_metrics": {
    "avg_entropy": 7.12,
    "avg_contrast": 43.87,
    "avg_brightness": 125.34,
    "avg_sharpness": 298.56,
    "avg_psnr": 27.89
  }
}
```

### Metrics Explained

| Metric         | Range    | Description                          | Better Value             |
| -------------- | -------- | ------------------------------------ | ------------------------ |
| **Entropy**    | 0-8 bits | Information content                  | Higher                   |
| **Contrast**   | 0-255    | RMS contrast                         | Higher (for enhancement) |
| **Brightness** | 0-255    | Average intensity                    | Context-dependent        |
| **Sharpness**  | 0-∞      | Edge definition (Laplacian variance) | Higher                   |
| **PSNR**       | 0-∞ dB   | Peak Signal-to-Noise Ratio           | Higher (but not always)  |

---

## 🔬 How CLAHE Works

### Algorithm Overview

1. **Image Division**: Input image is divided into small tiles (e.g., 8×8)
2. **Local Histogram Equalization**: Each tile's histogram is equalized independently
3. **Contrast Limiting**: Histogram is clipped at a threshold to prevent noise amplification
4. **Bilinear Interpolation**: Tile boundaries are smoothed to avoid artifacts
5. **Color Space Conversion**: For color images, only the luminance channel is enhanced

### Mathematical Foundation

For each tile, the cumulative distribution function (CDF) is computed:

```
CDF(i) = Σ(j=0 to i) P(j)
```

Where `P(j)` is the probability of intensity level `j`. The enhanced pixel value is:

```
enhanced(x,y) = CDF(original(x,y)) × 255
```

Contrast limiting is applied by clipping the histogram before computing the CDF:

```
if H(i) > clip_limit:
    H(i) = clip_limit
```

### Color Space Processing

- **LAB**: Enhances L (lightness) channel → Perceptually uniform
- **HSV**: Enhances V (value) channel → Preserves color saturation
- **YCrCb**: Enhances Y (luminance) channel → Broadcast standard

---

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'cv2'"

**Solution:**

```bash
pip install opencv-python
```

### Issue: "No images found in data/input"

**Solution:**

```bash
# Create directory and add images
mkdir -p data/input
cp /path/to/your/images/* data/input/
```

### Issue: "Invalid number of channels"

**Solution:** Ensure your image is in a supported format (JPG, PNG, BMP). Try explicitly specifying `--grayscale` flag.

### Issue: Display window doesn't show

**Solution:** Remove the `--show` flag or ensure your system supports GUI display (not available on headless servers).

---

## 👥 Team

**Group 4 - DCIT 407 Image Processing**

- **Member 1**: Implementation and algorithm design
- **Member 2**: Testing and validation
- **Member 3**: Documentation and analysis
- **Member 4**: Metrics and evaluation

---

## 📚 References

1. **Zuiderveld, K.** (1994). "Contrast Limited Adaptive Histogram Equalization." _Graphics Gems IV_, Academic Press, pp. 474-485.

2. **Pizer, S. M., et al.** (1987). "Adaptive histogram equalization and its variations." _Computer Vision, Graphics, and Image Processing_, 39(3), 355-368.

3. **Gonzalez, R. C., & Woods, R. E.** (2018). _Digital Image Processing_ (4th ed.). Pearson.

---

## 📄 License

This project is developed for academic purposes as part of DCIT 407 coursework.

---

## 🤝 Contributing

This is an academic project. For questions or suggestions, please contact the team members.

---

## 🌟 Acknowledgments

- University of Ghana, Department of Computer Science
- DCIT 407 Course Instructors
- OpenCV Community

---

**Made by Group 4**
