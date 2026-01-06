# Mouth Detection Feature - User Checklist

## Code Updates Complete

All code has been updated to support the 3-head model with mouth detection:

- [x] `trainer.py` - Model architecture updated to 3 heads
- [x] `trainer.py` - Label loading updated for mouth_open field
- [x] `train_pipeline.py` - Training loop handles 3 outputs
- [x] `aug_pipeline.py` - Augmentation preserves mouth_open labels
- [x] `driver.py` - Live inference displays mouth state
- [x] `README.md` - Comprehensive documentation added

## 📋 What You Need to Do

### 1. Label Your Training Data

For each image in `data/images/`, create or update the corresponding JSON in `data/labels/`:

**Add this field to each label:**
```json
{
  "shapes": [...],
  "class": 1,
  "bbox": [...],
  "mouth_open": 0  // ← ADD THIS: 0 = closed, 1 = open
}
```

**Labeling Guidelines:**
- `mouth_open: 1` - Teeth visible, clear opening
- `mouth_open: 0` - Lips closed or barely parted

### 2. Partition Your Data

Once labels are updated, partition into train/val/test:

```python
from trainer import move_all_data
move_all_data(total_images=120)  # Adjust for your dataset size
```

This splits:
- 70% → `data/train/`
- 15% → `data/val/`
- 15% → `data/test/`

### 3. Run Augmentation

Generate 60 augmented versions of each image:

```python
from aug_pipeline import AugmentationPipeline

pipeline = AugmentationPipeline()
pipeline.run()
```

Expected output:
```
Processing 120 images x 60 augmentations = 7200 total outputs
Augmenting train: 100%|████████| 84/84
Augmenting test: 100%|█████████| 18/18
Augmenting val: 100%|██████████| 18/18
✓ Generated 7200 augmented images from 120 originals.
```

### 4. Train the Model

```bash
cd src
python trainer.py
```

Watch for 4 loss metrics:
- `total_loss` - Combined loss
- `face_class_loss` - Face detection loss
- `bbox_loss` - Bounding box accuracy loss
- `mouth_loss` - Mouth state classification loss ← NEW!

### 5. Test Live Inference

```bash
python driver.py
```

You should see:
- Green box around your face
- "MOUTH OPEN" (red) when you open your mouth
- "Mouth Closed" (blue) when you close your mouth
- Press 'q' to quit

## 🔍 Troubleshooting

### "KeyError: 'mouth_open'" during augmentation

**Cause**: Some labels don't have the `mouth_open` field

**Fix**: The code handles this with `.get('mouth_open', 0)`, but verify your labels:
```bash
cd data/labels
grep -L "mouth_open" *.json  # Lists files missing the field
```

### All predictions show "Mouth Closed"

**Causes**:
1. Not enough "mouth open" examples in training data
2. Model hasn't trained enough epochs
3. Threshold too high (default 0.5)

**Fix**: 
- Ensure balanced dataset (roughly 50/50 open/closed)
- Train for more epochs
- Lower threshold in `driver.py`: `mouth_is_open = mouth_open_pred[0][0] > 0.3`

### Model trains but mouth detection is random

**Cause**: Labels might be incorrectly set

**Fix**: Verify labels are accurate:
```python
import json
import os

for f in os.listdir('aug_data/train/labels')[:10]:
    with open(f'aug_data/train/labels/{f}') as file:
        data = json.load(file)
        print(f"{f}: mouth_open={data.get('mouth_open', 'MISSING')}")
```

## 📊 Expected Results

After training:
- **Face detection**: ~95%+ accuracy (should be very high)
- **Bounding box**: IoU > 0.7 (reasonable overlap)
- **Mouth detection**: ~80-90% accuracy (depends on data quality)

## 🚀 Next Steps

Once working:
1. Collect more diverse training data (different lighting, angles)
2. Balance dataset (equal open/closed examples)
3. Train for more epochs (20-30)
4. Fine-tune loss weights if one head dominates
5. Consider adding temporal smoothing to reduce flickering

## 📝 Quick Reference

**Label Format:**
```json
{
  "image": "abc123.jpg",
  "class": 1,              // 1 = face present
  "bbox": [0.4, 0.3, 0.7, 0.8],  // [x1, y1, x2, y2]
  "mouth_open": 0          // 0 = closed, 1 = open
}
```

**Model Outputs:**
1. `face_class` - [0-1] probability of face
2. `face_bbox` - [x1, y1, x2, y2] normalized coordinates
3. `mouth_open` - [0-1] probability of mouth being open

**Loss Weights:**
- `bbox_loss` - 1.0
- `face_class_loss` - 0.5
- `mouth_loss` - 0.5

---

**Need Help?** Check `MOUTH_DETECTION_UPGRADE.md` for technical details or `README.md` for full documentation.
