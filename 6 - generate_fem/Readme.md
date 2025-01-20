<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FEM Scripts</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
        }
        h1 {
            color: #2c3e50;
        }
        h2 {
            color: #34495e;
        }
        pre {
            background-color: #f4f4f4;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            overflow-x: auto;
        }
        code {
            font-family: Consolas, "Courier New", monospace;
        }
        .command {
            background-color: #eef9ff;
            padding: 5px;
            border: 1px solid #ccc;
            border-radius: 5px;
            display: inline-block;
        }
    </style>
</head>
<body>
<h1>FEM Scripts</h1>

<h2>1. generate_fem.py</h2>
<p>This script generates and visualizes Class Activation Maps (CAMs) for a set of input images using a YOLO model.</p>

<h3>Usage</h3>
<pre class="command">python generate_fem.py --model_path &lt;path_to_model&gt; --input_images_path &lt;path_to_input_images&gt;</pre>

<h3>Arguments</h3>
<ul>
    <li><strong>--model_path</strong>: Path to the YOLO model file (e.g., <code>best.pt</code>).</li>
    <li><strong>--input_images_path</strong>: Path to the folder containing input images for visualization.</li>
</ul>

<h3>Output</h3>
<p>The script visualizes:</p>
<ul>
    <li>CAMs from different target layers (scale 80, 40, 20).</li>
    <li>The annotated frame with YOLO predictions.</li>
</ul>
<p>Visualizations are displayed as a grid of four subplots.</p>

<h2>2. save_femimages.py</h2>
<p>This script generates and saves CAMs as image files for a set of input images using a YOLO model.</p>

<h3>Usage</h3>
<pre class="command">python save_femimages.py --model_path &lt;path_to_model&gt; --input_images_path &lt;path_to_input_images&gt; --output_path &lt;path_to_save_results&gt;</pre>

<h3>Arguments</h3>
<ul>
    <li><strong>--model_path</strong>: Path to the YOLO model file (e.g., <code>best.pt</code>).</li>
    <li><strong>--input_images_path</strong>: Path to the folder containing input images for CAM generation.</li>
    <li><strong>--output_path</strong>: Path to save the generated CAM images.</li>
</ul>

<h3>Output</h3>
<p>The script generates CAM overlays for the target layer and saves them in the specified output directory. Each CAM file is named after the input image with the layer information appended (e.g., <code>image_s2.png</code>).</p>

<h2>Prerequisites</h2>
<ul>
    <li>Python 3.8 or higher</li>
    <li>Required libraries:
        <ul>
            <li><code>torch</code></li>
            <li><code>opencv-python</code></li>
            <li><code>numpy</code></li>
            <li><code>ultralytics</code></li>
            <li>Custom library: <code>Fem</code></li>
        </ul>
    </li>
    <li>A trained YOLO model file (e.g., <code>best.pt</code>).</li>
</ul>

<h2>Notes</h2>
<ul>
    <li>Ensure that the <code>Fem</code> module is properly installed or accessible in your Python environment.</li>
    <li>Both scripts assume that the YOLO model and input images are correctly formatted and compatible with the YOLO framework.</li>
</ul>
</body>
</html>
