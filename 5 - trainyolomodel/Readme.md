<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLOv8 Train and Predict</title>
</head>
<body>
<h1>YOLOv8 Train and Predict</h1>

<h2>Training</h2>
<p>To train a YOLOv8 model, use the following command:</p>
<pre><code>python train.py \
--model_path yolov8n.pt \
--data_path /path/to/custom.yaml \
--epochs 150 \
--img_size 640 \
--batch_size 16 \
--workers 4 \
--device 0</code></pre>

<h3>Arguments:</h3>
<ul>
    <li><strong>--model_path</strong>: Path to the YOLOv8 weights file (e.g., <code>yolov8n.pt</code>).</li>
    <li><strong>--data_path</strong>: Path to the dataset YAML file.</li>
    <li><strong>--epochs</strong>: Number of training epochs.</li>
    <li><strong>--img_size</strong>: Size of the training images (default: 640).</li>
    <li><strong>--batch_size</strong>: Batch size for training (default: 16).</li>
    <li><strong>--workers</strong>: Number of workers for data loading (default: 4).</li>
    <li><strong>--device</strong>: Device to use (<code>0</code> for GPU, <code>cpu</code> for CPU).</li>
</ul>

<h2>Prediction</h2>
<p>To run inference on a set of images, use the following command:</p>
<pre><code>python predict.py \
--model_path /path/to/best.pt \
--imagesdata /path/to/images \
--save_path /path/to/save/annotated/images</code></pre>

<h3>Arguments:</h3>
<ul>
    <li><strong>--model_path</strong>: Path to the trained YOLO model weights (e.g., <code>best.pt</code>).</li>
    <li><strong>--imagesdata</strong>: Path to the directory containing input images.</li>
    <li><strong>--save_path</strong>: Path to save the annotated images.</li>
</ul>

<h2>Notes</h2>
<ul>
    <li>Ensure the required Python packages are installed (e.g., <code>ultralytics</code>, <code>opencv-python</code>).</li>
    <li>Check the paths and file permissions before running the scripts.</li>
</ul>
</body>
</html>
