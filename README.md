# Real-Time Face and Mouth Detection System

A deep learning-based computer vision system for real-time face detection and mouth state classification using a custom 3-head VGG16 neural network.

## Features

- **Face Detection**: Binary classification to detect presence of faces
- **Face Localization**: Bounding box regression to locate faces in frame
- **Mouth State Detection**: Binary classification to detect open/closed mouth state
- **Real-time Inference**: Live webcam tracking with visual feedback

## Architecture

### 3-Head VGG16-Based Model

The model uses VGG16 (pretrained on ImageNet) as a shared feature extractor with three specialized prediction heads:

```
Input (120x120x3)
    ↓
VGG16 Feature Extractor (pretrained, frozen)
    ↓
   ┌─────────────┬─────────────┬─────────────┐
   │   Head 1    │   Head 2    │   Head 3    │
   │  Face Class │  Face Bbox  │ Mouth Open  │
   │  (sigmoid)  │  (sigmoid)  │  (sigmoid)  │
   │  Output: 1  │  Output: 4  │  Output: 1  │
   └─────────────┴─────────────┴─────────────┘
```

**Head 1**: Face Classification
- Dense(2048, relu) → Dense(1, sigmoid)
- Output: Binary probability [0-1] indicating face presence

**Head 2**: Face Bounding Box Regression  
- Dense(2048, relu) → Dense(4, sigmoid)
- Output: Normalized coordinates [x1, y1, x2, y2]

**Head 3**: Mouth Open Classification
- Dense(2048, relu) → Dense(1, sigmoid)
- Output: Binary probability [0-1] indicating mouth open state

### Loss Functions

- **Face Classification**: Binary Cross-Entropy
- **Bounding Box Regression**: Custom Localization Loss (coordinate + size deltas)
- **Mouth Classification**: Binary Cross-Entropy
- **Combined Loss**: `bbox_loss + 0.5*face_class_loss + 0.5*mouth_loss`

## Project Structure

```
reels-detect/
├── src/
│   ├── get_data.py          # Webcam image capture
│   ├── aug_pipeline.py      # Data augmentation (711x400 → 60x per image)
│   ├── train_pipeline.py    # Custom training loop for 3-head model
│   ├── trainer.py           # Model building and training orchestration
│   ├── driver.py            # Real-time inference with webcam
│   ├── facedetection.keras  # Trained model weights
│   ├── data/                # Original labeled data
│   │   ├── images/          # Raw webcam captures
│   │   ├── labels/          # JSON annotations
│   │   ├── train/           # 70% split
│   │   ├── val/             # 15% split
│   │   └── test/            # 15% split
│   └── aug_data/            # Augmented training data (7,200 images)
│       ├── train/
│       ├── val/
│       └── test/
├── requirements.txt
└── README.md
```

## Label Format

Labels are stored as JSON files with the following schema:

```json
{
  "image": "filename.jpg",
  "class": 1,                    // 0 = no face, 1 = face present
  "bbox": [0.4, 0.3, 0.7, 0.8],  // [x1, y1, x2, y2] normalized to [0-1]
  "mouth_open": 1                // 0 = closed, 1 = open
}
```

### Creating Labels

use !labelme

When manually labeling data:
1. Use a tool like LabelMe or labelImg to draw bounding boxes around faces
2. Add a `mouth_open` field to the JSON:
   - `0` if mouth is closed
   - `1` if mouth is open

## Installation

```bash
# Clone repository
git clone https://github.com/sammy12bammy/reels-detect.git
cd reels-detect

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Collection

Capture training images from webcam:

```python
from get_data import ImageCapture

capture = ImageCapture(images_path='data/images', camera_id=0)
capture.capture(num_images=120, delay=0.5, show_preview=True)
```

### 2. Manual Labeling

Label faces and mouth state in captured images. Ensure JSON labels include `mouth_open` field.

### 3. Data Augmentation

Generate augmented training data:

```python
from aug_pipeline import AugmentationPipeline

pipeline = AugmentationPipeline(
    data_dir='data',
    output_dir='aug_data',
    target_width=711,
    target_height=400,
    augmentations_per_image=60
)
pipeline.run()
```

This generates:
- **7,200 training samples** from 120 base images
- Random transformations: flips, brightness, gamma, RGB shifts
- Automatic bbox coordinate transformation

### 4. Training

```bash
python src/trainer.py
```

Training configuration:
- **Batch size**: 8
- **Epochs**: 10
- **Optimizer**: Adam (lr=0.0001, decay)
- **Train/Val/Test split**: 70/15/15

### 5. Real-time Inference

```bash
python src/driver.py
```

Features:
- Green bounding box around detected face
- Red label "MOUTH OPEN" when mouth detected as open
- Blue label "Mouth Closed" when mouth detected as closed
- Confidence scores displayed at bottom
- Press 'q' to quit

## Model Performance

The model is trained on:
- **120 base images** → **7,200 augmented samples**
- **Image resolution**: 120x120 (resized from 711x400 augmented images)
- **Normalization**: Pixel values scaled to [0, 1]

## Technical Details

### Augmentation Pipeline

Images undergo the following preprocessing:
1. **Resize**: Scale height to 400px (preserve aspect ratio)
2. **Center Crop**: Crop width to 711px
3. **Random Augmentations**:
   - Horizontal flip (p=0.5)
   - Vertical flip (p=0.5)
   - Brightness/Contrast (p=0.2)
   - Gamma adjustment (p=0.2)
   - RGB shift (p=0.2)

### Training Pipeline

Custom `FaceTracker` class handles:
- Multi-output predictions
- Weighted loss combination
- Gradient computation and optimization
- Separate tracking of face_class, bbox, and mouth losses

## Requirements

- Python 3.9+
- TensorFlow 2.x
- OpenCV
- NumPy
- Albumentations
- tqdm

See `requirements.txt` for full dependency list.

## Future Improvements

- [ ] Add eye open/closed detection (4th head)
- [ ] Increase training data size
- [ ] Experiment with different backbone architectures (ResNet, EfficientNet)
- [ ] Deploy to edge devices (Raspberry Pi, Jetson Nano)
- [ ] Add emotion classification
- [ ] Support multiple face detection

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

Samuel - [GitHub](https://github.com/sammy12bammy)