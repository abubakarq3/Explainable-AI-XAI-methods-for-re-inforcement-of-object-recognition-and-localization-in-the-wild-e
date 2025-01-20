<!DOCTYPE html>
<html lang="en">

<body>
<h1>YOLO Format Conversion</h1>
<p>This script converts a dataset into the YOLO format by processing bounding box annotations and saving them alongside corresponding images in the desired format.</p>

<h2>Usage</h2>
<p>Run the script using Python with the following command:</p>
<pre><code>python convert.py --objectpath &lt;path_to_dataset&gt; --imagesdata &lt;path_to_save_images&gt; --annotationdata &lt;path_to_save_annotations&gt; [--class_id &lt;class_id&gt;]</code></pre>

<h2>Arguments</h2>
<ul>
    <li><code>--objectpath</code>: Path to the dataset containing object sequences. (Required)</li>
    <li><code>--imagesdata</code>: Path to save the processed images. (Required)</li>
    <li><code>--annotationdata</code>: Path to save the YOLO format annotations. (Required)</li>
    <li><code>--class_id</code>: (Optional) Class ID for YOLO format. Default is <code>3</code>.</li>
</ul>

<h2>Example</h2>
<pre><code>python script.py \
--objectpath /path/to/dataset \
--imagesdata /path/to/images \
--annotationdata /path/to/annotations</code></pre>

<h2>Output</h2>
<p>The script saves:</p>
<ul>
    <li>Processed images in the specified <code>--imagesdata</code> directory.</li>
    <li>YOLO format annotations in the specified <code>--annotationdata</code> directory.</li>
</ul>

<h2>Notes</h2>
<ul>
    <li>Ensure the <code>normalize_bbox</code> function is correctly implemented in your <code>utils</code> module.</li>
    <li>Check that the input dataset is organized as expected with bounding box files and frames.</li>
</ul>
</body>
</html>
