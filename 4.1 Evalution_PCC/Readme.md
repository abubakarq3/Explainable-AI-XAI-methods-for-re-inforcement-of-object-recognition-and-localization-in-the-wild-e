<!DOCTYPE html>
<html lang="en">

<body>
    <h1>Evaluation PCC Script</h1>

  <h2>Overview</h2>
  <p>This script calculates the Pearson Correlation Coefficient (PCC) between pairs of saliency maps and explainer-generated maps, enabling quantitative comparison of visual explainability methods.</p>

  <h2>Requirements</h2>
  <ul>
      <li>Python 3.6 or higher</li>
      <li>Required libraries:
          <ul>
              <li><code>Pillow</code></li>
              <li><code>numpy</code></li>
              <li><code>scipy</code></li>
              <li><code>tqdm</code></li>
              <li><code>csv</code></li>
          </ul>
      </li>
  </ul>

  <h3>Install dependencies:</h3>
  <pre><code>pip install pillow numpy scipy tqdm</code></pre>

  <h2>Directory Structure</h2>
  <p>Ensure your directories are structured as follows:</p>
  <pre><code>
  project_directory/
  ├── saliency_folder/     # Directory containing saliency maps
  ├── explainer_folder/    # Directory containing explainer-generated maps
  └── Evaluation_explainers.py  # This script
  </code></pre>

  <h2>Usage</h2>
  <ol>
      <li>Update the following paths in the script:
          <ul>
              <li><code>saliency_folder</code>: Path to the folder containing saliency maps.</li>
              <li><code>explainer_folder</code>: Path to the folder containing explainer-generated maps.</li>
              <li><code>output_csv</code>: Path to save the PCC results in CSV format.</li>
          </ul>
      </li>
      <li>Run the script:</li>
  </ol>
  <pre><code>python Evaluation_explainers.py</code></pre>

  <h2>Output</h2>
  <ul>
      <li>A CSV file containing the PCC values for each pair of matching images and the mean PCC value.</li>
      <li>Example output in the terminal:
          <pre><code>
          PCC Results:
          image1.png: 0.85
          image2.png: 0.78
          Mean: 0.82
          Results saved to Eigencam_M1.csv
          </code></pre>
      </li>
  </ul>

</body>
</html>
