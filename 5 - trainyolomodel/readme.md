<!DOCTYPE html>
<html lang="en">

<body>
    <h1>YOLOv8 Model Training</h1>

  <h2>Overview</h2>
  <p>This script trains a YOLOv8 model using the <code>ultralytics</code> library. It supports custom data annotations and allows configuration of key training parameters like epochs, batch size, and image size.</p>

  <h2>Requirements</h2>
  <ul>
      <li>Python 3.7 or higher</li>
      <li>Required libraries:
          <ul>
              <li><code>ultralytics</code></li>
          </ul>
      </li>
      <li>GPU with CUDA support (optional but recommended for faster training)</li>
  </ul>

  <h3>Install dependencies:</h3>
  <pre><code>pip install ultralytics</code></pre>

  <h2>Usage</h2>
  <ol>
      <li>Edit the script to specify your training data and parameters:
          <ul>
              <li><code>data</code>: Path to the YOLO format <code>.yaml</code> file that defines your dataset.</li>
              <li><code>epochs</code>: Number of epochs for training.</li>
              <li><code>imgsz</code>: Image size for training.</li>
              <li><code>batch</code>: Batch size for training.</li>
              <li><code>device</code>: Specify <code>0</code> for GPU or <code>'cpu'</code> for CPU training.</li>
          </ul>
      </li>
      <li>Run the script:
          <pre><code>python train.py</code></pre>
      </li>
  </ol>

  <h2>Example Configuration</h2>
  <pre><code>
model.train(
    data='/path/to/custom.yaml',
    epochs=150,
    imgsz=640,
    batch=16,
    workers=4,
    device=0
)
    </code></pre>

  <h2>Output</h2>
  <p>The script generates the following outputs:</p>
  <ul>
      <li>Model checkpoints in the <code>runs/train/</code> directory.</li>
      <li>Training logs and metrics, including loss and mAP values.</li>
      <li>Trained model weights for deployment or further fine-tuning.</li>
  </ul>

  <h2>Directory Structure</h2>
  <p>Ensure your dataset follows the YOLO format and directory structure:</p>
  <pre><code>
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── custom.yaml  # Dataset configuration file
    </code></pre>

  <h2>Dataset Configuration File (<code>custom.yaml</code>)</h2>
  <p>An example dataset configuration file:</p>
  <pre><code>
path: /path/to/dataset
train: images/train
val: images/val

nc: 2  # Number of classes
names: ['class1', 'class2']  # Class names
    </code></pre>


  <h2>License</h2>
  <p>This script is distributed under the MIT License.</p>
</body>
</html>
