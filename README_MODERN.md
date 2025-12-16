
### 3.4 Training:

#### 3.4.1 Traditional Training (Original Script)

```bash
python train.py -b [batch_size] -lr [learning rate] -e [epochs]

# Example: Train MobileFaceNet on CASIA-WebFace
python train.py -net mobilefacenet -b 200 -w 4 -d casia-webface -e 20 -s 9981
```

#### 3.4.2 Modern Training (Recommended)

**New training script with improved features:**
- No PIL dependency (uses cv2 only)
- Better progress tracking with tqdm
- Simplified data loading
- TensorBoard integration

**Step 1: Check Dataset**
```bash
# Check if your dataset format is compatible with MobileFaceNet (112x112)
python check_dataset.py
```

**Step 2: Train Model**
```bash
# Basic usage
python train_modern.py -d datasets/casia-webface -b 200 -w 4 -e 20

# With initial step (continue training or adjust step counter)
python train_modern.py -d datasets/casia-webface -b 200 -w 4 -e 20 -s 9981

# Resume from checkpoint
python train_modern.py -d datasets/casia-webface -b 200 -w 4 -e 15 -s 11461 \
    -r work_space/models/mobilefacenet_epoch20_step11461_final.pth
```

**Step 3: Monitor Training**
```bash
# View training curves in TensorBoard
tensorboard --logdir=work_space/log
```

**Training Parameters:**
- `-d, --data_path`: Dataset path (default: `datasets/casia-webface`)
- `-b, --batch_size`: Batch size (default: 200)
- `-e, --epochs`: Number of epochs (default: 10)
- `-lr, --lr`: Learning rate (default: 0.001)
- `-w, --num_workers`: Number of workers (default: 4)
- `-s, --initial_step`: Initial step number for display (default: 0)
- `-r, --resume`: Resume from checkpoint (optional)

**Output:**
- Models saved in: `work_space/models/`
- TensorBoard logs in: `work_space/log/`

---

### 3.5 Model Conversion & Deployment (RK3568)

#### 3.5.1 PyTorch → ONNX Conversion

Convert trained PyTorch model to ONNX format:

```bash
# Convert pretrained model
python convert_to_onnx.py -i mobilefacenet.pth -o mobilefacenet.onnx

# Convert trained model
python convert_to_onnx.py -i work_space/models/mobilefacenet_epoch20_step11461_final.pth \
                          -o mobilefacenet_trained.onnx

# Custom batch size
python convert_to_onnx.py -i mobilefacenet.pth -o mobilefacenet.onnx --batch-size 1
```

**Requirements:**
```bash
pip install onnx onnxruntime
```

**Output:**
- ONNX model with verified accuracy
- Automatic validation against PyTorch output
- Model size: ~4.8 MB

#### 3.5.2 ONNX → RKNN Conversion (For RK3568 NPU)

**⚠️ Must run on Linux (VMware VM recommended)**

**Prerequisites:**
```bash
# Install RKNN-Toolkit2
pip install rknn-toolkit2==2.3.2
```

**Prepare Calibration Dataset:**
```bash
# Create filtered label file (only for int_data folders)
grep -E "casia-webface/000696|casia-webface/000697" \
    datasets/int_data/casia-webface.txt > datasets/int_data/int_data_labels.txt
```

**Convert to RKNN:**
```bash
# With INT8 quantization (recommended, 50 calibration images)
python convert_onnx_to_rknn.py -i mobilefacenet.onnx -o mobilefacenet.rknn

# Custom calibration image count
python convert_onnx_to_rknn.py -i mobilefacenet.onnx -o mobilefacenet.rknn --max-calib-images 100

# Without quantization (FP16, slower but more accurate)
python convert_onnx_to_rknn.py -i mobilefacenet.onnx -o mobilefacenet.rknn --no-quantization

# Custom dataset path
python convert_onnx_to_rknn.py -i mobilefacenet.onnx -o mobilefacenet.rknn \
    -d datasets/int_data -l datasets/int_data/int_data_labels.txt
```

**Conversion Parameters:**
- Target Platform: RK3568
- Quantization: INT8 (w8a8 - 权重8位 + 激活8位)
- Optimization Level: 3
- Preprocessing: RGB, mean=[127.5, 127.5, 127.5], std=[127.5, 127.5, 127.5]

**Output:**
- RKNN model: ~1-2 MB (quantized)
- Quantization accuracy loss: ~1-3%

#### 3.5.3 Deploy on RK3568

**Inference Example:**
```python
import cv2
import numpy as np
from rknnlite.api import RKNNLite

# Load model
rknn = RKNNLite()
rknn.load_rknn('mobilefacenet.rknn')
rknn.init_runtime()

# Prepare input (112x112 RGB image, uint8, [0-255])
img = cv2.imread('face.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (112, 112))

# Add batch dimension: (112, 112, 3) -> (1, 112, 112, 3)
img = np.expand_dims(img, axis=0)

# Inference (specify NHWC format)
outputs = rknn.inference(inputs=[img], data_format='nhwc')
embedding = outputs[0][0]  # 512-dim feature vector

# Compare faces (cosine similarity)
similarity = np.dot(embedding1, embedding2)  # threshold: 0.3-0.5

rknn.release()
```

**Input Specification:**
- Size: 112×112
- Format: RGB, HWC (Height, Width, Channels)
- Type: uint8
- Range: [0, 255]
- Preprocessing: Automatically handled by RKNN model

**Output:**
- 512-dimensional feature vector (float32)
- L2-normalized

---

## 4. Project Structure

```
InsightFace_Pytorch/
├── train.py                    # Original training script
├── train_modern.py             # Modern training script (recommended)
├── check_dataset.py            # Dataset format checker
├── convert_to_onnx.py          # PyTorch → ONNX converter
├── convert_onnx_to_rknn.py     # ONNX → RKNN converter (Linux only)
├── model.py                    # Model definitions (MobileFaceNet, IR-SE, ArcFace)
├── config.py                   # Configuration file
├── Learner.py                  # Training learner class
├── data/                       # Data processing utilities
│   └── data_pipe.py
├── mtcnn_pytorch/              # Face detection & alignment
│   └── src/
│       └── align_trans.py      # Face alignment (for deployment)
├── datasets/                   # Training datasets
│   ├── casia-webface/          # Main training dataset
│   └── int_data/               # Calibration dataset for RKNN
│       ├── 000696/
│       ├── 000697/
│       └── int_data_labels.txt
├── work_space/                 # Training outputs
│   ├── models/                 # Saved models
│   └── log/                    # TensorBoard logs
└── mobilefacenet.pth          # Pretrained model
```

---

## 5. References 

- This repo is mainly inspired by [deepinsight/insightface](https://github.com/deepinsight/insightface) , [InsightFace_TF](https://github.com/auroua/InsightFace_TF) and [InsightFace_Pytorch]https://github.com/TreB1eN/InsightFace_Pytorch?tab=readme-ov-file