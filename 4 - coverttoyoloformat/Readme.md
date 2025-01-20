<!DOCTYPE html>
<html lang="en">

<body>
    <h1>YOLO Dataset Conversion and Splitting</h1>

  <h2>Overview</h2>
  <p>This project provides Python scripts for:</p>
  <ul>
      <li>Converting bounding box annotations and images into YOLO format.</li>
      <li>Splitting the dataset into training and testing sets with balanced class distribution.</li>
  </ul>

  <h2>Scripts</h2>
  <h3>1. <code>convert.py</code></h3>
  <p>The <code>convert.py</code> script processes images and bounding box annotations from the specified input format and converts them into YOLO format. It saves the converted images and labels in separate directories.</p>

  <h4>Key Features:</h4>
  <ul>
      <li>Normalizes bounding box coordinates to YOLO format.</li>
      <li>Supports reading from structured directories.</li>
      <li>Ensures proper directory creation for saving processed files.</li>
  </ul>

  <h4>Usage:</h4>
  <pre><code>python convert.py</code></pre>

  <h3>2. <code>train_test_split.py</code></h3>
  <p>The <code>train_test_split.py</code> script splits the YOLO dataset into training and testing sets. It ensures a balanced class distribution by splitting files based on class prefixes.</p>

  <h4>Key Features:</h4>
  <ul>
      <li>Splits data into training and testing sets based on a specified ratio.</li>
      <li>Creates separate directories for images and labels in the training and testing sets.</li>
      <li>Ensures the existence of corresponding label files before splitting.</li>
  </ul>

  <h4>Usage:</h4>
  <pre><code>python train_test_split.py</code></pre>

  <h2>Directory Structure</h2>
  <p>After conversion and splitting, the directory structure will look like this:</p>
  <pre><code>
yolo_dataset/
├── train/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
</code></pre>


</body>
</html>
