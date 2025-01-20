<!DOCTYPE html>
<html lang="en">

<body>
    <h1>Generate FEM with YOLOv8</h1>

  <h2>Overview</h2>
  <p>This script uses a YOLOv8 model to generate Feature Explanation Maps (FEMs) for images, providing visual insights into the model's focus during predictions.</p>

  <h2>Requirements</h2>
  <ul>
      <li>Python 3.7 or higher</li>
      <li>Required libraries:
          <ul>
              <li><code>opencv-python</code></li>
              <li><code>torch</code></li>
              <li><code>torchvision</code></li>
              <li><code>numpy</code></li>
              <li><code>matplotlib</code></li>
              <li><code>ultralytics</code></li>
          </ul>
      </li>
      <li>YOLOv8 model weights (e.g., <code>best.pt</code> from a trained model)</li>
  </ul>

  <h3>Install dependencies:</h3>
  <pre><code>pip install opencv-python torch torchvision numpy matplotlib ultralytics</code></pre>

  <h2>Usage</h2>
  <ol>
      <li>Ensure your YOLOv8 weights file (<code>best.pt</code>) is available and specify its path in the script:
          <pre><code>model = YOLO("/path/to/your/weights/best.pt")</code></pre>
      </li>
      <li>Set the path to the input images directory:
          <pre><code>input_images_path = "/path/to/your/images"</code></pre>
      </li>
      <li>Run the script:
          <pre><code>python generate_fem.py</code></pre>
      </li>
  </ol>

  <h2>Output</h2>
  <p>The script generates the following outputs:</p>
  <ul>
      <li><strong>Feature Explanation Maps:</strong> Heatmaps showing model focus areas for predictions.</li>
      <li><strong>Annotated Frames:</strong> Images with YOLO-detected bounding boxes and labels.</li>
      <li>Both outputs are displayed using <code>matplotlib</code> for visualization.</li>
  </ul>

  <h2>Directory Structure</h2>
  <p>Ensure your project is organized as follows:</p>
  <pre><code>
project_directory/
├── generate_fem.py           # This script
├── weights/
│   └── best.pt               # YOLOv8 model weights
├── images/
│   └── test/                 # Input images for generating FEMs
└── output/                   # Directory to save output images (optional)
    </code></pre>

  <h2>Customization</h2>
  <ul>
      <li><strong>Target Layers:</strong> Modify <code>targetlayers</code> in the script to change the layers used for generating FEMs.</li>
      <li><strong>Visualization:</strong> Adjust the <code>image_weight</code> parameter in <code>show_cam_on_image</code> to change the blending of the FEM and the original image.</li>
  </ul>

</body>
</html>
