<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PCC Evaluation Script - README</title>
</head>
<body>
<h1>PCC Evaluation Script</h1>
<p>This script computes the Pearson Correlation Coefficient (PCC) between pairs of images from two directories (saliency maps and explainer maps). The results are saved to a CSV file.</p>

<h2>Features</h2>
<ul>
    <li>Loads images from specified directories.</li>
    <li>Resizes images to a target shape.</li>
    <li>Computes PCC for matching filenames between directories.</li>
    <li>Outputs results, including a mean PCC, to a CSV file.</li>
</ul>

<h2>Requirements</h2>
<p>Install the following Python libraries:</p>
<ul>
    <li><code>numpy</code></li>
    <li><code>Pillow</code></li>
    <li><code>scipy</code></li>
    <li><code>tqdm</code></li>
    <li><code>argparse</code> (built-in with Python)</li>
</ul>

<h2>Usage</h2>
<h3>Command-line Execution</h3>
<p>Run the script with the following command:</p>
<pre><code>python pcc_evaluation.py --saliency_folder &lt;path_to_saliency_folder&gt; \
                        --explainer_folder &lt;path_to_explainer_folder&gt; \
                        --output_csv &lt;path_to_output_csv&gt; \
                        --target_shape &lt;height&gt; &lt;width&gt;</code></pre>

<h3>Example</h3>
<pre><code>python pcc_evaluation.py --saliency_folder /path/to/saliency \
                        --explainer_folder /path/to/explainer \
                        --output_csv results.csv \
                        --target_shape 1080 1920</code></pre>

<h2>Arguments</h2>
<ul>
    <li><strong>--saliency_folder</strong>: Path to the folder containing saliency maps (required).</li>
    <li><strong>--explainer_folder</strong>: Path to the folder containing explainer maps (required).</li>
    <li><strong>--output_csv</strong>: Path to save the output CSV file (required).</li>
    <li><strong>--target_shape</strong>: Target image dimensions as <code>height</code> and <code>width</code> (optional, default is <code>384x640</code>).</li>
</ul>

<h2>Output</h2>
<p>The script generates a CSV file containing the following columns:</p>
<ul>
    <li><strong>Filename</strong>: Name of the saliency image.</li>
    <li><strong>PCC</strong>: Pearson Correlation Coefficient between the saliency image and the corresponding explainer image.</li>
</ul>
<p>The last row of the CSV file contains the mean PCC across all evaluated image pairs.</p>

<h2>License</h2>
<p>MIT License</p>
</body>
</html>
