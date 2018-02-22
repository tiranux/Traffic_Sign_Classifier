<div tabindex="-1" id="notebook" class="border-box-sizing">

<div class="container" id="notebook-container">

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

# Self-Driving Car Engineer Nanodegree[¶](#Self-Driving-Car-Engineer-Nanodegree)

## Deep Learning[¶](#Deep-Learning)

## Project: Build a Traffic Sign Recognition Classifier[¶](#Project:-Build-a-Traffic-Sign-Recognition-Classifier)

In this notebook, a template is provided for you to implement your functionality in stages which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission, if necessary. Sections that begin with **'Implementation'** in the header indicate where you should begin your implementation for your project. Note that some sections of implementation are optional, and will be marked with **'Optional'** in the header.

In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.

> **Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

* * *

## Step 0: Load The Data[¶](#Step-0:-Load-The-Data)

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [1]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="c1"># Load pickled data</span>
<span class="kn">import</span> <span class="nn">pickle</span>

<span class="c1"># TODO: Fill this in based on where you saved the training and testing data</span>

<span class="n">training_file</span> <span class="o">=</span> <span class="s1">'train.p'</span>
<span class="n">testing_file</span> <span class="o">=</span> <span class="s1">'test.p'</span>

<span class="k">with</span> <span class="nb">open</span><span class="p">(</span><span class="n">training_file</span><span class="p">,</span> <span class="n">mode</span><span class="o">=</span><span class="s1">'rb'</span><span class="p">)</span> <span class="k">as</span> <span class="n">f</span><span class="p">:</span>
    <span class="n">train</span> <span class="o">=</span> <span class="n">pickle</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="n">f</span><span class="p">)</span>
<span class="k">with</span> <span class="nb">open</span><span class="p">(</span><span class="n">testing_file</span><span class="p">,</span> <span class="n">mode</span><span class="o">=</span><span class="s1">'rb'</span><span class="p">)</span> <span class="k">as</span> <span class="n">f</span><span class="p">:</span>
    <span class="n">test</span> <span class="o">=</span> <span class="n">pickle</span><span class="o">.</span><span class="n">load</span><span class="p">(</span><span class="n">f</span><span class="p">)</span>

<span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span> <span class="o">=</span> <span class="n">train</span><span class="p">[</span><span class="s1">'features'</span><span class="p">],</span> <span class="n">train</span><span class="p">[</span><span class="s1">'labels'</span><span class="p">]</span>
<span class="n">X_test</span><span class="p">,</span> <span class="n">y_test</span> <span class="o">=</span> <span class="n">test</span><span class="p">[</span><span class="s1">'features'</span><span class="p">],</span> <span class="n">test</span><span class="p">[</span><span class="s1">'labels'</span><span class="p">]</span>
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

* * *

## Step 1: Dataset Summary & Exploration[¶](#Step-1:-Dataset-Summary-&-Exploration)

The pickled data is a dictionary with 4 key/value pairs:

*   `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
*   `'labels'` is a 2D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
*   `'sizes'` is a list containing tuples, (width, height) representing the the original width and height the image.
*   `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**

Complete the basic data summary below.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [2]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="c1">### Replace each question mark with the appropriate value.</span>

<span class="c1"># TODO: Number of training examples</span>
<span class="n">n_train</span> <span class="o">=</span> <span class="n">X_train</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>

<span class="c1"># TODO: Number of testing examples.</span>
<span class="n">n_test</span> <span class="o">=</span> <span class="n">X_test</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span>

<span class="c1"># TODO: What's the shape of an traffic sign image?</span>
<span class="n">image_shape</span> <span class="o">=</span> <span class="n">X_train</span><span class="p">[</span><span class="mi">0</span><span class="p">]</span><span class="o">.</span><span class="n">shape</span>

<span class="c1"># TODO: How many unique classes/labels there are in the dataset.</span>
<span class="n">n_classes</span> <span class="o">=</span> <span class="mi">43</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">"Number of training examples ="</span><span class="p">,</span> <span class="n">n_train</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"Number of testing examples ="</span><span class="p">,</span> <span class="n">n_test</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"Image data shape ="</span><span class="p">,</span> <span class="n">image_shape</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"Number of classes ="</span><span class="p">,</span> <span class="n">n_classes</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre>Number of training examples = 39209
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc.

The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.

**NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [3]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="c1">### Data exploration visualization goes here.</span>
<span class="c1">### Feel free to use as many code cells as needed.</span>
<span class="kn">import</span> <span class="nn">matplotlib.pyplot</span> <span class="k">as</span> <span class="nn">plt</span>
<span class="kn">import</span> <span class="nn">random</span>
<span class="c1"># Visualizations will be shown in the notebook.</span>
<span class="o">%</span><span class="k">matplotlib</span> inline

<span class="n">index</span> <span class="o">=</span> <span class="n">random</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="n">X_train</span><span class="p">))</span>
<span class="n">image</span> <span class="o">=</span> <span class="n">X_train</span><span class="p">[</span><span class="n">index</span><span class="p">]</span><span class="o">.</span><span class="n">squeeze</span><span class="p">()</span>

<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">))</span>
<span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">image</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="n">y_train</span><span class="p">[</span><span class="n">index</span><span class="p">])</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre>6
</pre>

</div>

</div>

<div class="output_area">

<div class="output_png output_subarea ">![](data:image/png;base64,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
)</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

* * *

## Step 2: Design and Test a Model Architecture[¶](#Step-2:-Design-and-Test-a-Model-Architecture)

Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

There are various aspects to consider when thinking about this problem:

*   Neural network architecture
*   Play around preprocessing techniques (normalization, rgb to grayscale, etc)
*   Number of examples per label (some have more than others).
*   Generate fake data.

Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

**NOTE:** The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play!

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Implementation[¶](#Implementation)

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project. Once you have completed your implementation and are satisfied with the results, be sure to thoroughly answer the questions that follow.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [4]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="kn">import</span> <span class="nn">tensorflow</span> <span class="k">as</span> <span class="nn">tf</span>

<span class="n">EPOCHS</span> <span class="o">=</span> <span class="mi">25</span>
<span class="n">BATCH_SIZE</span> <span class="o">=</span> <span class="mi">128</span>
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [5]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="n">x</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">placeholder</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">float32</span><span class="p">,</span> <span class="p">(</span><span class="kc">None</span><span class="p">,</span> <span class="mi">32</span><span class="p">,</span> <span class="mi">32</span><span class="p">,</span> <span class="mi">3</span><span class="p">))</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">placeholder</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">int32</span><span class="p">,</span> <span class="p">(</span><span class="kc">None</span><span class="p">))</span>
<span class="n">one_hot_y</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">one_hot</span><span class="p">(</span><span class="n">y</span><span class="p">,</span> <span class="mi">43</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Question 1[¶](#Question-1)

_Describe how you preprocessed the data. Why did you choose that technique?_

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

**Answer:**

*   Adaptative histogram equalization was applied, as decribed in Wikipedia: "Adaptive histogram equalization (AHE) is a computer image processing technique used to improve contrast in images. It differs from ordinary histogram equalization in the respect that the adaptive method computes several histograms, each corresponding to a distinct section of the image, and uses them to redistribute the lightness values of the image. It is therefore suitable for improving the local contrast and enhancing the definitions of edges in each region of an image." [https://en.wikipedia.org/wiki/Adaptive_histogram_equalization](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization)

    *   Therefore that makes better pictures that are clearer.
*   I also applied random rotations in order to simulate situation where the camera detects the signs from a different angle. It also proved to improve the results.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [6]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="c1">### Preprocess the data here.</span>
<span class="c1">### Feel free to use as many code cells as needed.</span>
<span class="kn">from</span> <span class="nn">sklearn.model_selection</span> <span class="k">import</span> <span class="n">train_test_split</span>

<span class="c1">#Splitting training into training and validation</span>
<span class="n">X_train</span><span class="p">,</span> <span class="n">X_validation</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span> <span class="n">y_validation</span> <span class="o">=</span>  <span class="n">train_test_split</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span>  <span class="n">y_train</span><span class="p">,</span> <span class="n">test_size</span><span class="o">=</span><span class="mf">0.2</span><span class="p">,</span> <span class="n">random_state</span><span class="o">=</span><span class="mi">42</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [7]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="c1">### Generate data additional data (OPTIONAL!)</span>
<span class="c1">### and split the data into training/validation/testing sets here.</span>
<span class="c1">### Feel free to use as many code cells as needed.</span>

<span class="kn">from</span> <span class="nn">sklearn.utils</span> <span class="k">import</span> <span class="n">shuffle</span>
<span class="kn">from</span> <span class="nn">scipy</span> <span class="k">import</span> <span class="n">ndimage</span>
<span class="kn">from</span> <span class="nn">skimage</span> <span class="k">import</span> <span class="n">exposure</span>
<span class="kn">import</span> <span class="nn">numpy</span> <span class="k">as</span> <span class="nn">np</span>

<span class="c1">#Shuffle training data</span>
<span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span> <span class="o">=</span> <span class="n">shuffle</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>

<span class="c1">#Temp auxiliar vector for data augmentation</span>
<span class="n">aux</span><span class="o">=</span><span class="p">[]</span>

<span class="c1">#Applying equalizer to validation set</span>
<span class="k">for</span> <span class="n">image</span> <span class="ow">in</span> <span class="n">X_train</span><span class="p">:</span>
    <span class="n">image</span> <span class="o">=</span> <span class="n">exposure</span><span class="o">.</span><span class="n">equalize_adapthist</span><span class="p">(</span><span class="n">image</span><span class="p">,</span> <span class="n">clip_limit</span><span class="o">=</span><span class="mf">0.03</span><span class="p">)</span>
    <span class="n">aux</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">image</span><span class="p">)</span>

<span class="n">X_train</span> <span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">aux</span><span class="p">)</span>  

<span class="c1">#Creating a copy of training set in order to apply data augmentation</span>
<span class="n">X_train2</span><span class="p">,</span> <span class="n">y_train2</span> <span class="o">=</span> <span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span>

<span class="c1">#Creating a second dataset with random rotations</span>
<span class="n">aux</span><span class="o">=</span><span class="p">[]</span>
<span class="k">for</span> <span class="n">image</span> <span class="ow">in</span> <span class="n">X_train2</span><span class="p">:</span>
    <span class="n">randomNumber</span><span class="o">=</span><span class="n">random</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span><span class="mi">5</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">randomNumber</span><span class="o">>=</span> <span class="mi">3</span><span class="p">:</span>
        <span class="n">image</span> <span class="o">=</span> <span class="n">ndimage</span><span class="o">.</span><span class="n">rotate</span><span class="p">(</span><span class="n">image</span><span class="p">,</span> <span class="n">random</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="o">-</span><span class="mi">10</span><span class="p">,</span><span class="mi">10</span><span class="p">),</span> <span class="n">reshape</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
    <span class="k">elif</span> <span class="n">randomNumber</span> <span class="o"><</span> <span class="mi">3</span><span class="p">:</span>
        <span class="n">image</span> <span class="o">=</span> <span class="n">ndimage</span><span class="o">.</span><span class="n">rotate</span><span class="p">(</span><span class="n">image</span><span class="p">,</span> <span class="n">random</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="o">-</span><span class="mi">45</span><span class="p">,</span><span class="mi">45</span><span class="p">),</span> <span class="n">reshape</span><span class="o">=</span><span class="kc">False</span><span class="p">)</span>
    <span class="n">aux</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">image</span><span class="p">)</span>

<span class="n">X_train2</span> <span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">aux</span><span class="p">)</span>  

<span class="c1">#Visualizing a sample picture after equalization</span>
<span class="n">index</span> <span class="o">=</span> <span class="n">random</span><span class="o">.</span><span class="n">randint</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="n">X_train</span><span class="p">))</span>
<span class="n">sample_img</span> <span class="o">=</span> <span class="n">X_train</span><span class="p">[</span><span class="n">index</span><span class="p">]</span><span class="o">.</span><span class="n">squeeze</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">))</span>
<span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">sample_img</span><span class="p">)</span>

<span class="c1">#Applying equalizer to validation set</span>
<span class="n">aux</span><span class="o">=</span><span class="p">[]</span>
<span class="k">for</span> <span class="n">image</span> <span class="ow">in</span> <span class="n">X_validation</span><span class="p">:</span>
    <span class="n">image</span> <span class="o">=</span> <span class="n">exposure</span><span class="o">.</span><span class="n">equalize_adapthist</span><span class="p">(</span><span class="n">image</span><span class="p">,</span> <span class="n">clip_limit</span><span class="o">=</span><span class="mf">0.03</span><span class="p">)</span>
    <span class="n">aux</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">image</span><span class="p">)</span>

<span class="n">X_validation</span> <span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">aux</span><span class="p">)</span> 

<span class="c1">#Applying equalizer to test set</span>
<span class="n">aux</span><span class="o">=</span><span class="p">[]</span>
<span class="k">for</span> <span class="n">image</span> <span class="ow">in</span> <span class="n">X_test</span><span class="p">:</span>
    <span class="n">image</span> <span class="o">=</span> <span class="n">exposure</span><span class="o">.</span><span class="n">equalize_adapthist</span><span class="p">(</span><span class="n">image</span><span class="p">,</span> <span class="n">clip_limit</span><span class="o">=</span><span class="mf">0.03</span><span class="p">)</span>
    <span class="n">aux</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">image</span><span class="p">)</span>

<span class="n">X_test</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">aux</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stderr output_text">

<pre>/home/israel/anaconda3/envs/CarND-Traffic-Sign-Classifier-Project/lib/python3.5/site-packages/skimage/util/dtype.py:110: UserWarning: Possible precision loss when converting from float64 to uint16
  "%s to %s" % (dtypeobj_in, dtypeobj))
</pre>

</div>

</div>

<div class="output_area">

<div class="output_png output_subarea ">![](data:image/png;base64,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
)</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Question 2[¶](#Question-2)

_Describe how you set up the training, validation and testing data for your model. **Optional**: If you generated additional data, how did you generate the data? Why did you generate the data? What are the differences in the new dataset (with generated data) from the original dataset?_

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

**Answer:**

*   The training set was splitted in order to create an evaluation set.
*   As described in the 1st question, the contrast of the pictures was improved with adaptative histogram equalization, this was applied to the training, validation y test datasets.
*   A copy of trainign was created with adding random rotation in the pictures (also described in 1st question)

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [8]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="c1">### Define your architecture here.</span>
<span class="c1">### Feel free to use as many code cells as needed.</span>

<span class="kn">from</span> <span class="nn">tensorflow.contrib.layers</span> <span class="k">import</span> <span class="n">flatten</span>

<span class="n">keep_prob</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">placeholder</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">float32</span><span class="p">)</span> 

<span class="k">def</span> <span class="nf">LeNet</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="n">train</span><span class="o">=</span><span class="kc">False</span><span class="p">):</span>    
    <span class="c1"># Hyperparameters</span>
    <span class="n">mu</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="n">sigma</span> <span class="o">=</span> <span class="mf">0.1</span>

    <span class="c1">#Normalizing input data</span>
    <span class="n">x</span><span class="o">=</span><span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">l2_normalize</span><span class="p">(</span><span class="n">x</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="n">epsilon</span><span class="o">=</span><span class="mi">1</span><span class="n">e</span><span class="o">-</span><span class="mi">12</span><span class="p">,</span> <span class="n">name</span><span class="o">=</span><span class="kc">None</span><span class="p">)</span>

    <span class="c1"># Layer 1: Convolutional. Input = 32x32x3\. Output = 28x28x70.</span>
    <span class="n">conv1_W</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">truncated_normal</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">5</span><span class="p">,</span> <span class="mi">5</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="mi">70</span><span class="p">),</span> <span class="n">mean</span> <span class="o">=</span> <span class="n">mu</span><span class="p">,</span> <span class="n">stddev</span> <span class="o">=</span> <span class="n">sigma</span><span class="p">))</span>
    <span class="n">conv1_b</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="mi">70</span><span class="p">))</span>
    <span class="n">conv1</span>   <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">conv2d</span><span class="p">(</span> <span class="n">x</span><span class="p">,</span> <span class="n">conv1_W</span><span class="p">,</span> <span class="n">strides</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="n">padding</span><span class="o">=</span><span class="s1">'VALID'</span><span class="p">)</span> <span class="o">+</span> <span class="n">conv1_b</span>
    <span class="c1"># Activation.</span>
    <span class="n">conv1</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">elu</span><span class="p">(</span><span class="n">conv1</span><span class="p">)</span>
    <span class="c1"># Pooling. Input = 28x28x70\. Output = 14x14x70.</span>
    <span class="n">conv1</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">max_pool</span><span class="p">(</span><span class="n">conv1</span><span class="p">,</span> <span class="n">ksize</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="n">strides</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="n">padding</span><span class="o">=</span><span class="s1">'VALID'</span><span class="p">)</span>

    <span class="c1"># Layer 2: Convolutional. Input = 14x14x70  Output = 12x12x120.</span>
    <span class="n">conv2_W</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">truncated_normal</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">3</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="mi">70</span><span class="p">,</span> <span class="mi">120</span><span class="p">),</span> <span class="n">mean</span> <span class="o">=</span> <span class="n">mu</span><span class="p">,</span> <span class="n">stddev</span> <span class="o">=</span> <span class="n">sigma</span><span class="p">))</span>
    <span class="n">conv2_b</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="mi">120</span><span class="p">))</span>
    <span class="n">conv2</span>   <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">conv2d</span><span class="p">(</span><span class="n">conv1</span><span class="p">,</span> <span class="n">conv2_W</span><span class="p">,</span> <span class="n">strides</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="n">padding</span><span class="o">=</span><span class="s1">'VALID'</span><span class="p">)</span> <span class="o">+</span> <span class="n">conv2_b</span>
    <span class="c1"># Activation.</span>
    <span class="n">conv2</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">elu</span><span class="p">(</span><span class="n">conv2</span><span class="p">)</span>
    <span class="c1"># Pooling. Input = 12x12x120\. Output = 6x6x120.</span>
    <span class="n">conv2</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">max_pool</span><span class="p">(</span><span class="n">conv2</span><span class="p">,</span> <span class="n">ksize</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="n">strides</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="n">padding</span><span class="o">=</span><span class="s1">'VALID'</span><span class="p">)</span>

     <span class="c1"># Layer 3 1x1: Convolutional. Input = 6x6x120  Output = 6x6x120.</span>
    <span class="n">conv1x1_W</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">truncated_normal</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">120</span><span class="p">,</span> <span class="mi">120</span><span class="p">),</span> <span class="n">mean</span> <span class="o">=</span> <span class="n">mu</span><span class="p">,</span> <span class="n">stddev</span> <span class="o">=</span> <span class="n">sigma</span><span class="p">))</span>
    <span class="n">conv1x1_b</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="mi">120</span><span class="p">))</span>
    <span class="n">conv1x1</span>   <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">conv2d</span><span class="p">(</span><span class="n">conv2</span><span class="p">,</span> <span class="n">conv1x1_W</span><span class="p">,</span> <span class="n">strides</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="n">padding</span><span class="o">=</span><span class="s1">'VALID'</span><span class="p">)</span> <span class="o">+</span> <span class="n">conv1x1_b</span>
    <span class="c1"># Activation.</span>
    <span class="n">conv1x1</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">elu</span><span class="p">(</span><span class="n">conv1x1</span><span class="p">)</span>

    <span class="c1"># Layer 4: Convolutional. Input = 6x6x120   Output = 4x4x256.</span>
    <span class="n">conv3_W</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">truncated_normal</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">3</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="mi">120</span><span class="p">,</span> <span class="mi">256</span><span class="p">),</span> <span class="n">mean</span> <span class="o">=</span> <span class="n">mu</span><span class="p">,</span> <span class="n">stddev</span> <span class="o">=</span> <span class="n">sigma</span><span class="p">))</span>
    <span class="n">conv3_b</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="mi">256</span><span class="p">))</span>
    <span class="n">conv3</span>   <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">conv2d</span><span class="p">(</span><span class="n">conv1x1</span><span class="p">,</span> <span class="n">conv3_W</span><span class="p">,</span> <span class="n">strides</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="n">padding</span><span class="o">=</span><span class="s1">'VALID'</span><span class="p">)</span> <span class="o">+</span> <span class="n">conv3_b</span>
    <span class="c1"># Activation.</span>
    <span class="n">conv3</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">elu</span><span class="p">(</span><span class="n">conv3</span><span class="p">)</span>
    <span class="c1"># SOLUTION: Pooling. Input = 4x4x256\. Output = 2x2x256.</span>
    <span class="n">conv3</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">max_pool</span><span class="p">(</span><span class="n">conv3</span><span class="p">,</span> <span class="n">ksize</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="n">strides</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">2</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="n">padding</span><span class="o">=</span><span class="s1">'VALID'</span><span class="p">)</span>
    <span class="n">conv3</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">dropout</span><span class="p">(</span><span class="n">conv3</span><span class="p">,</span> <span class="n">keep_prob</span><span class="p">)</span>

    <span class="c1"># Layer 5 1x1: Convolutional. Input = 2x2x256  Output = 2x2x256.</span>
    <span class="n">conv1x_W</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">truncated_normal</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">256</span><span class="p">,</span> <span class="mi">256</span><span class="p">),</span> <span class="n">mean</span> <span class="o">=</span> <span class="n">mu</span><span class="p">,</span> <span class="n">stddev</span> <span class="o">=</span> <span class="n">sigma</span><span class="p">))</span>
    <span class="n">conv1x_b</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="mi">256</span><span class="p">))</span>
    <span class="n">conv1x</span>   <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">conv2d</span><span class="p">(</span><span class="n">conv3</span><span class="p">,</span> <span class="n">conv1x_W</span><span class="p">,</span> <span class="n">strides</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="n">padding</span><span class="o">=</span><span class="s1">'VALID'</span><span class="p">)</span> <span class="o">+</span> <span class="n">conv1x_b</span>
    <span class="c1"># Activation.</span>
    <span class="n">conv1x</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">elu</span><span class="p">(</span><span class="n">conv1x</span><span class="p">)</span>

    <span class="c1"># Layer 6 1x1: Convolutional. Input = 2x2x256  Output = 2x2x256.</span>
    <span class="n">conv2x_W</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">truncated_normal</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">256</span><span class="p">,</span> <span class="mi">256</span><span class="p">),</span> <span class="n">mean</span> <span class="o">=</span> <span class="n">mu</span><span class="p">,</span> <span class="n">stddev</span> <span class="o">=</span> <span class="n">sigma</span><span class="p">))</span>
    <span class="n">conv2x_b</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="mi">256</span><span class="p">))</span>
    <span class="n">conv2x</span>   <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">conv2d</span><span class="p">(</span><span class="n">conv1x</span><span class="p">,</span> <span class="n">conv2x_W</span><span class="p">,</span> <span class="n">strides</span><span class="o">=</span><span class="p">[</span><span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">,</span> <span class="mi">1</span><span class="p">],</span> <span class="n">padding</span><span class="o">=</span><span class="s1">'VALID'</span><span class="p">)</span> <span class="o">+</span> <span class="n">conv2x_b</span>
    <span class="c1"># Activation.</span>
    <span class="n">conv2x</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">elu</span><span class="p">(</span><span class="n">conv2x</span><span class="p">)</span>
    <span class="n">conv2x</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">dropout</span><span class="p">(</span><span class="n">conv2x</span><span class="p">,</span> <span class="n">keep_prob</span><span class="p">)</span>

    <span class="c1"># Flatten:  Input = 2x2x256  Output = 1024.</span>
    <span class="n">fc0</span>  <span class="o">=</span> <span class="n">flatten</span><span class="p">(</span><span class="n">conv1x</span><span class="p">)</span>

    <span class="c1"># Layer 7: Fully Connected. Input = 1024\. Output = 400.</span>
    <span class="n">fc1_W</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">truncated_normal</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">1024</span><span class="p">,</span> <span class="mi">400</span><span class="p">),</span> <span class="n">mean</span> <span class="o">=</span> <span class="n">mu</span><span class="p">,</span> <span class="n">stddev</span> <span class="o">=</span> <span class="n">sigma</span><span class="p">))</span>
    <span class="n">fc1_b</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="mi">400</span><span class="p">))</span>
    <span class="n">fc1</span>   <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">matmul</span><span class="p">(</span><span class="n">fc0</span><span class="p">,</span> <span class="n">fc1_W</span><span class="p">)</span> <span class="o">+</span> <span class="n">fc1_b</span>
    <span class="c1"># Activation.</span>
    <span class="n">fc1</span>    <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">elu</span><span class="p">(</span><span class="n">fc1</span><span class="p">)</span>
    <span class="c1">#fc1 dropout</span>
    <span class="n">fc1</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">dropout</span><span class="p">(</span><span class="n">fc1</span><span class="p">,</span> <span class="n">keep_prob</span><span class="p">)</span>

    <span class="c1"># Layer 8: Fully Connected. Input = 400\. Output = 43.</span>
    <span class="n">fc2_W</span>  <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">truncated_normal</span><span class="p">(</span><span class="n">shape</span><span class="o">=</span><span class="p">(</span><span class="mi">400</span><span class="p">,</span> <span class="mi">43</span><span class="p">),</span> <span class="n">mean</span> <span class="o">=</span> <span class="n">mu</span><span class="p">,</span> <span class="n">stddev</span> <span class="o">=</span> <span class="n">sigma</span><span class="p">))</span>
    <span class="n">fc2_b</span>  <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">Variable</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">zeros</span><span class="p">(</span><span class="mi">43</span><span class="p">))</span>
    <span class="n">fc2_W</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">dropout</span><span class="p">(</span><span class="n">fc2_W</span><span class="p">,</span> <span class="n">keep_prob</span><span class="p">)</span>
    <span class="n">logits</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">matmul</span><span class="p">(</span><span class="n">fc1</span><span class="p">,</span> <span class="n">fc2_W</span><span class="p">)</span> <span class="o">+</span> <span class="n">fc2_b</span>
    <span class="c1">#logits = tf.nn.dropout(logits, keep_prob)</span>

    <span class="c1">#losses = tf.nn.l2_loss(conv1_W) + tf.nn.l2_loss(conv2_W) + tf.nn.l2_loss(conv3_W) + tf.nn.l2_loss(conv3x_W) + tf.nn.l2_loss(conv2x_W) + tf.nn.l2_loss(conv1x_W) + tf.nn.l2_loss(fc1_W) + tf.nn.l2_loss(fc3_W)</span>
    <span class="c1">#losses += tf.nn.l2_loss(conv1_b) + tf.nn.l2_loss(conv2_b) + tf.nn.l2_loss(conv3_b) + tf.nn.l2_loss(conv3x_b) + tf.nn.l2_loss(conv2x_b) + tf.nn.l2_loss(conv1x_b) + tf.nn.l2_loss(fc1_b) + tf.nn.l2_loss(fc3_b)</span>

    <span class="k">return</span> <span class="n">logits</span>
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [9]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="n">rate</span> <span class="o">=</span> <span class="mf">0.001</span>

<span class="n">logits</span> <span class="o">=</span> <span class="n">LeNet</span><span class="p">(</span><span class="n">x</span><span class="p">)</span>
<span class="n">cross_entropy</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">softmax_cross_entropy_with_logits</span><span class="p">(</span><span class="n">logits</span><span class="p">,</span> <span class="n">one_hot_y</span><span class="p">)</span>
<span class="n">loss_operation</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">reduce_mean</span><span class="p">(</span><span class="n">cross_entropy</span> <span class="p">)</span>
<span class="n">optimizer</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">train</span><span class="o">.</span><span class="n">AdamOptimizer</span><span class="p">(</span><span class="n">learning_rate</span> <span class="o">=</span> <span class="n">rate</span><span class="p">)</span>
<span class="n">training_operation</span> <span class="o">=</span> <span class="n">optimizer</span><span class="o">.</span><span class="n">minimize</span><span class="p">(</span><span class="n">loss_operation</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [10]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="n">correct_prediction</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">equal</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">argmax</span><span class="p">(</span><span class="n">logits</span><span class="p">,</span> <span class="mi">1</span><span class="p">),</span> <span class="n">tf</span><span class="o">.</span><span class="n">argmax</span><span class="p">(</span><span class="n">one_hot_y</span><span class="p">,</span> <span class="mi">1</span><span class="p">))</span>
<span class="n">accuracy_operation</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">reduce_mean</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">cast</span><span class="p">(</span><span class="n">correct_prediction</span><span class="p">,</span> <span class="n">tf</span><span class="o">.</span><span class="n">float32</span><span class="p">))</span>

<span class="k">def</span> <span class="nf">evaluate</span><span class="p">(</span><span class="n">X_data</span><span class="p">,</span> <span class="n">y_data</span><span class="p">):</span>
    <span class="n">num_examples</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">X_data</span><span class="p">)</span>
    <span class="n">total_accuracy</span> <span class="o">=</span> <span class="mi">0</span>
    <span class="n">sess</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">get_default_session</span><span class="p">()</span>
    <span class="k">for</span> <span class="n">offset</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">num_examples</span><span class="p">,</span> <span class="n">BATCH_SIZE</span><span class="p">):</span>
        <span class="n">batch_x</span><span class="p">,</span> <span class="n">batch_y</span> <span class="o">=</span> <span class="n">X_data</span><span class="p">[</span><span class="n">offset</span><span class="p">:</span><span class="n">offset</span><span class="o">+</span><span class="n">BATCH_SIZE</span><span class="p">],</span> <span class="n">y_data</span><span class="p">[</span><span class="n">offset</span><span class="p">:</span><span class="n">offset</span><span class="o">+</span><span class="n">BATCH_SIZE</span><span class="p">]</span>
        <span class="n">accuracy</span> <span class="o">=</span> <span class="n">sess</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">accuracy_operation</span><span class="p">,</span> <span class="n">feed_dict</span><span class="o">=</span><span class="p">{</span><span class="n">x</span><span class="p">:</span> <span class="n">batch_x</span><span class="p">,</span> <span class="n">y</span><span class="p">:</span> <span class="n">batch_y</span><span class="p">,</span> <span class="n">keep_prob</span><span class="p">:</span> <span class="mf">1.0</span><span class="p">})</span>
        <span class="n">total_accuracy</span> <span class="o">+=</span> <span class="p">(</span><span class="n">accuracy</span> <span class="o">*</span> <span class="nb">len</span><span class="p">(</span><span class="n">batch_x</span><span class="p">))</span>
    <span class="k">return</span> <span class="n">total_accuracy</span> <span class="o">/</span> <span class="n">num_examples</span>
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Question 3[¶](#Question-3)

_What does your final architecture look like? (Type of model, layers, sizes, connectivity, etc.) For reference on how to build a deep neural network using TensorFlow, see [Deep Neural Network in TensorFlow](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/b516a270-8600-4f93-a0a3-20dfeabe5da6/concepts/83a3a2a2-a9bd-4b7b-95b0-eb924ab14432) from the classroom._

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

**Answer:** ![Traffic_Sign Architecture](TSC_architecture.PNG)

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [11]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="c1">### Train your model here.</span>
<span class="c1">### Feel free to use as many code cells as needed.</span>

<span class="k">with</span> <span class="n">tf</span><span class="o">.</span><span class="n">Session</span><span class="p">()</span> <span class="k">as</span> <span class="n">sess</span><span class="p">:</span>
    <span class="n">sess</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">tf</span><span class="o">.</span><span class="n">global_variables_initializer</span><span class="p">())</span>
    <span class="n">num_examples</span> <span class="o">=</span> <span class="nb">len</span><span class="p">(</span><span class="n">X_train</span><span class="p">)</span>

    <span class="nb">print</span><span class="p">(</span><span class="s2">"Training..."</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">()</span>
    <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">EPOCHS</span><span class="p">):</span>
        <span class="n">X_train_rot</span><span class="p">,</span> <span class="n">y_train_rot</span> <span class="o">=</span> <span class="n">shuffle</span><span class="p">(</span><span class="n">X_train2</span><span class="p">,</span> <span class="n">y_train2</span><span class="p">)</span>
        <span class="k">for</span> <span class="n">offset</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">num_examples</span><span class="p">,</span> <span class="n">BATCH_SIZE</span><span class="p">):</span>
            <span class="n">end</span> <span class="o">=</span> <span class="n">offset</span> <span class="o">+</span> <span class="n">BATCH_SIZE</span>
            <span class="n">batch_x</span><span class="p">,</span> <span class="n">batch_y</span> <span class="o">=</span> <span class="n">X_train_rot</span><span class="p">[</span><span class="n">offset</span><span class="p">:</span><span class="n">end</span><span class="p">],</span> <span class="n">y_train_rot</span><span class="p">[</span><span class="n">offset</span><span class="p">:</span><span class="n">end</span><span class="p">]</span>
            <span class="n">sess</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">training_operation</span><span class="p">,</span> <span class="n">feed_dict</span><span class="o">=</span><span class="p">{</span><span class="n">x</span><span class="p">:</span> <span class="n">batch_x</span><span class="p">,</span> <span class="n">y</span><span class="p">:</span> <span class="n">batch_y</span><span class="p">,</span> <span class="n">keep_prob</span><span class="p">:</span> <span class="mf">0.5</span><span class="p">})</span>

        <span class="n">validation_accuracy</span> <span class="o">=</span> <span class="n">evaluate</span><span class="p">(</span><span class="n">X_validation</span><span class="p">,</span> <span class="n">y_validation</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="s2">"EPOCH</span> <span class="si">{}</span> <span class="s2">..."</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">))</span>
        <span class="nb">print</span><span class="p">(</span><span class="s2">"Validation Accuracy =</span> <span class="si">{:.3f}</span><span class="s2">"</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">validation_accuracy</span><span class="p">))</span>
        <span class="nb">print</span><span class="p">()</span>
        <span class="k">if</span> <span class="nb">round</span><span class="p">(</span><span class="n">validation_accuracy</span><span class="p">,</span> <span class="mi">3</span><span class="p">)</span><span class="o">>=</span><span class="mf">0.991</span><span class="p">:</span>
            <span class="k">break</span>

    <span class="n">logits1</span> <span class="o">=</span> <span class="n">logits</span>

    <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">EPOCHS</span><span class="p">):</span>
        <span class="n">X_train_eq</span><span class="p">,</span> <span class="n">y_train_eq</span> <span class="o">=</span> <span class="n">shuffle</span><span class="p">(</span><span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">)</span>
        <span class="k">for</span> <span class="n">offset</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">0</span><span class="p">,</span> <span class="n">num_examples</span><span class="p">,</span> <span class="n">BATCH_SIZE</span><span class="p">):</span>
            <span class="n">end</span> <span class="o">=</span> <span class="n">offset</span> <span class="o">+</span> <span class="n">BATCH_SIZE</span>
            <span class="n">batch_x</span><span class="p">,</span> <span class="n">batch_y</span> <span class="o">=</span> <span class="n">X_train_eq</span><span class="p">[</span><span class="n">offset</span><span class="p">:</span><span class="n">end</span><span class="p">],</span> <span class="n">y_train_eq</span><span class="p">[</span><span class="n">offset</span><span class="p">:</span><span class="n">end</span><span class="p">]</span>
            <span class="n">sess</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">training_operation</span><span class="p">,</span> <span class="n">feed_dict</span><span class="o">=</span><span class="p">{</span><span class="n">x</span><span class="p">:</span> <span class="n">batch_x</span><span class="p">,</span> <span class="n">y</span><span class="p">:</span> <span class="n">batch_y</span><span class="p">,</span> <span class="n">keep_prob</span><span class="p">:</span> <span class="mf">0.5</span><span class="p">})</span>

        <span class="n">validation_accuracy</span> <span class="o">=</span> <span class="n">evaluate</span><span class="p">(</span><span class="n">X_validation</span><span class="p">,</span> <span class="n">y_validation</span><span class="p">)</span>
        <span class="nb">print</span><span class="p">(</span><span class="s2">"EPOCH</span> <span class="si">{}</span> <span class="s2">..."</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">i</span><span class="o">+</span><span class="mi">1</span><span class="p">))</span>
        <span class="nb">print</span><span class="p">(</span><span class="s2">"Validation Accuracy =</span> <span class="si">{:.3f}</span><span class="s2">"</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">validation_accuracy</span><span class="p">))</span>
        <span class="nb">print</span><span class="p">()</span>
        <span class="k">if</span> <span class="nb">round</span><span class="p">(</span><span class="n">validation_accuracy</span><span class="p">,</span> <span class="mi">3</span><span class="p">)</span><span class="o">>=</span><span class="mf">0.995</span><span class="p">:</span>
            <span class="k">break</span>

    <span class="n">logits</span> <span class="o">=</span> <span class="p">(</span><span class="n">logits</span> <span class="o">+</span> <span class="n">logits1</span><span class="p">)</span> <span class="o">/</span><span class="mi">2</span>    

    <span class="k">try</span><span class="p">:</span>
        <span class="n">saver</span>
    <span class="k">except</span> <span class="ne">NameError</span><span class="p">:</span>
        <span class="n">saver</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">train</span><span class="o">.</span><span class="n">Saver</span><span class="p">()</span>
    <span class="n">saver</span><span class="o">.</span><span class="n">save</span><span class="p">(</span><span class="n">sess</span><span class="p">,</span> <span class="s1">'lenet'</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">"Model saved"</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre>Training...

EPOCH 1 ...
Validation Accuracy = 0.779

EPOCH 2 ...
Validation Accuracy = 0.939

EPOCH 3 ...
Validation Accuracy = 0.960

EPOCH 4 ...
Validation Accuracy = 0.970

EPOCH 5 ...
Validation Accuracy = 0.974

EPOCH 6 ...
Validation Accuracy = 0.979

EPOCH 7 ...
Validation Accuracy = 0.979

EPOCH 8 ...
Validation Accuracy = 0.984

EPOCH 9 ...
Validation Accuracy = 0.988

EPOCH 10 ...
Validation Accuracy = 0.989

EPOCH 11 ...
Validation Accuracy = 0.990

EPOCH 12 ...
Validation Accuracy = 0.990

EPOCH 13 ...
Validation Accuracy = 0.988

EPOCH 14 ...
Validation Accuracy = 0.988

EPOCH 15 ...
Validation Accuracy = 0.981

EPOCH 16 ...
Validation Accuracy = 0.988

EPOCH 17 ...
Validation Accuracy = 0.988

EPOCH 18 ...
Validation Accuracy = 0.991

EPOCH 1 ...
Validation Accuracy = 0.992

EPOCH 2 ...
Validation Accuracy = 0.994

EPOCH 3 ...
Validation Accuracy = 0.994

EPOCH 4 ...
Validation Accuracy = 0.994

EPOCH 5 ...
Validation Accuracy = 0.995

Model saved
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [12]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="k">with</span> <span class="n">tf</span><span class="o">.</span><span class="n">Session</span><span class="p">()</span> <span class="k">as</span> <span class="n">sess</span><span class="p">:</span>
    <span class="c1">#sess.run(tf.global_variables_initializer())</span>
    <span class="n">loader</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">train</span><span class="o">.</span><span class="n">import_meta_graph</span><span class="p">(</span><span class="s1">'lenet.meta'</span><span class="p">)</span>
    <span class="n">loader</span><span class="o">.</span><span class="n">restore</span><span class="p">(</span><span class="n">sess</span><span class="p">,</span> <span class="n">tf</span><span class="o">.</span><span class="n">train</span><span class="o">.</span><span class="n">latest_checkpoint</span><span class="p">(</span><span class="s1">'./'</span><span class="p">))</span>

    <span class="n">test_accuracy</span> <span class="o">=</span> <span class="n">evaluate</span><span class="p">(</span><span class="n">X_test</span><span class="p">,</span> <span class="n">y_test</span><span class="p">)</span>
    <span class="c1">#sess.run(test_accuracy)</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">"Test Accuracy =</span> <span class="si">{:.3f}</span><span class="s2">"</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">test_accuracy</span><span class="p">))</span>

</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre>Test Accuracy = 0.977
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Question 4[¶](#Question-4)

_How did you train your model? (Type of optimizer, batch size, epochs, hyperparameters, etc.)_

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

**Answer:**

*   I chose ELU instead of RELU due to its slightly better perfomance.
*   I did try with other optimizer but it did not give a good result, so I kept AdamOptimizer.
*   After some training, a good batch size was 128.
*   I chose 25 epochs but I also assigned an early termination with a max value according to the learning improvement that I noticed.
*   The right learning_rate is 0.001, a greater value messes up the training and a lower values does not really improve it.
*   I also used two training session, one after another, with both dataset equalized but one of them with random rotations. I realized that this way the result improved in comparison to use only one of them (these are described in question one and two).

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Question 5[¶](#Question-5)

_What approach did you take in coming up with a solution to this problem? It may have been a process of trial and error, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think this is suitable for the current problem._

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

**Answer:**

*   It was mainly a trial and error process since I did not reall have a deep knowledge of DNN/CNN. I started with the architecture of LeNet provided in the lectures and started tunning the layers and hyperparameters in order to get a better prediction. At some point I researched for papers related to CNN and also solutions specificly applied to this well known problem and took some ideas, but I definitely did not want to simply copy one architecture, I better created my own solution.

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

* * *

## Step 3: Test a Model on New Images[¶](#Step-3:-Test-a-Model-on-New-Images)

Take several pictures of traffic signs that you find on the web or around you (at least five), and run them through your classifier on your computer to produce example results. The classifier might not recognize some local signs but it could prove interesting nonetheless.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Implementation[¶](#Implementation)

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project. Once you have completed your implementation and are satisfied with the results, be sure to thoroughly answer the questions that follow.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [13]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="c1">### Load the images and plot them here.</span>
<span class="c1">### Feel free to use as many code cells as needed.</span>

<span class="kn">from</span> <span class="nn">skimage</span> <span class="k">import</span> <span class="n">io</span>

<span class="c1">## Load new pics image collection</span>
<span class="n">new_pics_x</span> <span class="o">=</span> <span class="n">io</span><span class="o">.</span><span class="n">imread_collection</span><span class="p">(</span><span class="s1">'./new_pics/*.jpg'</span><span class="p">)</span>
<span class="n">new_pics_x</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">io</span><span class="o">.</span><span class="n">concatenate_images</span><span class="p">(</span><span class="n">new_pics_x</span><span class="p">))</span>
<span class="c1">## These are the labels for the new images</span>
<span class="n">new_pics_y</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">([</span><span class="mi">14</span><span class="p">,</span> <span class="mi">18</span><span class="p">,</span> <span class="mi">3</span><span class="p">,</span> <span class="mi">11</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">0</span><span class="p">,</span> <span class="mi">39</span> <span class="p">])</span>

<span class="n">aux</span><span class="o">=</span><span class="p">[]</span>
<span class="k">for</span> <span class="n">image</span> <span class="ow">in</span> <span class="n">new_pics_x</span><span class="p">:</span>
    <span class="n">image</span> <span class="o">=</span> <span class="n">exposure</span><span class="o">.</span><span class="n">equalize_adapthist</span><span class="p">(</span><span class="n">image</span><span class="p">,</span> <span class="n">clip_limit</span><span class="o">=</span><span class="mf">0.03</span><span class="p">)</span>
    <span class="n">aux</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">image</span><span class="p">)</span>

<span class="n">new_pics_x</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">aux</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stderr output_text">

<pre>/home/israel/anaconda3/envs/CarND-Traffic-Sign-Classifier-Project/lib/python3.5/site-packages/skimage/util/dtype.py:110: UserWarning: Possible precision loss when converting from float64 to uint16
  "%s to %s" % (dtypeobj_in, dtypeobj))
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Question 6[¶](#Question-6)

_Choose five candidate images of traffic signs and provide them in the report. Are there any particular qualities of the image(s) that might make classification difficult? It could be helpful to plot the images in the notebook._

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

**Answer:**

*   The speed limit signs may be difficult as per my experience, due to the similitudes among the other speed limit signs and, the "noise" surrounding the sign and also due to the angle in which there were taken. Many of the signs for label 0 have extra "noise" (a part of a sign above it is visible in the pic). However it is expected that random rotation in the pre processing improve the results.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [14]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="c1">### Run the predictions here.</span>
<span class="c1">### Feel free to use as many code cells as needed.</span>

<span class="c1"># Visualizations will be shown in the notebook.</span>
<span class="o">%</span><span class="k">matplotlib</span> inline

<span class="c1">#Visualizing the pictures</span>
<span class="k">for</span> <span class="n">image</span><span class="p">,</span> <span class="n">label</span> <span class="ow">in</span> <span class="nb">zip</span><span class="p">(</span><span class="n">new_pics_x</span><span class="p">,</span> <span class="n">new_pics_y</span><span class="p">):</span>
    <span class="n">image</span> <span class="o">=</span> <span class="n">image</span><span class="o">.</span><span class="n">squeeze</span><span class="p">()</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">1</span><span class="p">,</span><span class="mi">1</span><span class="p">))</span>
    <span class="n">plt</span><span class="o">.</span><span class="n">imshow</span><span class="p">(</span><span class="n">image</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">label</span><span class="p">)</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre>14
18
3
11
0
0
39
</pre>

</div>

</div>

<div class="output_area">

<div class="output_png output_subarea ">![](data:image/png;base64,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
b1BsGv4Gxf8DIFRH7T4X3mwAAAAASUVORK5CYII=
)</div>

</div>

<div class="output_area">

<div class="output_png output_subarea ">![](data:image/png;base64,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
awAAAABJRU5ErkJggg==
)</div>

</div>

<div class="output_area">

<div class="output_png output_subarea ">![](data:image/png;base64,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
)</div>

</div>

<div class="output_area">

<div class="output_png output_subarea ">![](data:image/png;base64,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
RK5CYII=
)</div>

</div>

<div class="output_area">

<div class="output_png output_subarea ">![](data:image/png;base64,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
MTX+dVuqqAAAAABJRU5ErkJggg==
)</div>

</div>

<div class="output_area">

<div class="output_png output_subarea ">![](data:image/png;base64,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
)</div>

</div>

<div class="output_area">

<div class="output_png output_subarea ">![](data:image/png;base64,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
+lu1Kh0AAAAASUVORK5CYII=
)</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Question 7[¶](#Question-7)

_Is your model able to perform equally well on captured pictures when compared to testing on the dataset? The simplest way to do this check the accuracy of the predictions. For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate._

_**NOTE:** You could check the accuracy manually by using `signnames.csv` (same directory). This file has a mapping from the class id (0-42) to the corresponding sign name. So, you could take the class id the model outputs, lookup the name in `signnames.csv` and see if it matches the sign from the image._

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

**Answer:**

*   It performs well in comparison to test dataset. Of course the percentage is lower since one single failure has more impact due to the number of samples. The result and more detail is displayed below

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [15]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="c1">### Visualize the softmax probabilities here.</span>
<span class="c1">### Feel free to use as many code cells as needed.</span>

<span class="k">with</span> <span class="n">tf</span><span class="o">.</span><span class="n">Session</span><span class="p">()</span> <span class="k">as</span> <span class="n">sess</span><span class="p">:</span>
    <span class="c1">#sess.run(tf.global_variables_initializer())</span>
    <span class="n">loader</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">train</span><span class="o">.</span><span class="n">import_meta_graph</span><span class="p">(</span><span class="s1">'lenet.meta'</span><span class="p">)</span>
    <span class="n">loader</span><span class="o">.</span><span class="n">restore</span><span class="p">(</span><span class="n">sess</span><span class="p">,</span> <span class="n">tf</span><span class="o">.</span><span class="n">train</span><span class="o">.</span><span class="n">latest_checkpoint</span><span class="p">(</span><span class="s1">'./'</span><span class="p">))</span>

    <span class="n">test_accuracy</span> <span class="o">=</span> <span class="n">evaluate</span><span class="p">(</span><span class="n">new_pics_x</span><span class="p">,</span> <span class="n">new_pics_y</span><span class="p">)</span>
    <span class="nb">print</span><span class="p">(</span><span class="s2">"Test Accuracy =</span> <span class="si">{:.3f}</span><span class="s2">"</span><span class="o">.</span><span class="n">format</span><span class="p">(</span><span class="n">test_accuracy</span><span class="p">))</span>
</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre>Test Accuracy = 0.857
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

### Question 8[¶](#Question-8)

_Use the model's softmax probabilities to visualize the **certainty** of its predictions, [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. Which predictions is the model certain of? Uncertain? If the model was incorrect in its initial prediction, does the correct prediction appear in the top k? (k should be 5 at most)_

`tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.

Take this numpy array as an example:

    # (5, 6) array
    a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
             0.12789202],
           [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
             0.15899337],
           [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
             0.23892179],
           [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
             0.16505091],
           [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
             0.09155967]])

Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:

    TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
           [ 0.28086119,  0.27569815,  0.18063401],
           [ 0.26076848,  0.23892179,  0.23664738],
           [ 0.29198961,  0.26234032,  0.16505091],
           [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
           [0, 1, 4],
           [0, 5, 1],
           [1, 3, 5],
           [1, 4, 3]], dtype=int32))

Looking just at the first row we get `[ 0.34763842, 0.24879643, 0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

</div>

</div>

</div>

<div class="cell border-box-sizing code_cell rendered">

<div class="input">

<div class="prompt input_prompt">In [16]:</div>

<div class="inner_cell">

<div class="input_area">

<div class=" highlight hl-ipython3">

<pre><span></span><span class="n">softmax_prob</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">softmax</span><span class="p">(</span><span class="n">logits</span><span class="p">)</span>
<span class="n">analyze_prediction</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">nn</span><span class="o">.</span><span class="n">top_k</span><span class="p">(</span><span class="n">softmax_prob</span><span class="p">,</span><span class="n">k</span><span class="o">=</span><span class="mi">5</span><span class="p">)</span>

<span class="k">with</span> <span class="n">tf</span><span class="o">.</span><span class="n">Session</span><span class="p">()</span> <span class="k">as</span> <span class="n">sess</span><span class="p">:</span>

    <span class="n">loader</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">train</span><span class="o">.</span><span class="n">import_meta_graph</span><span class="p">(</span><span class="s1">'lenet.meta'</span><span class="p">)</span>
    <span class="n">loader</span><span class="o">.</span><span class="n">restore</span><span class="p">(</span><span class="n">sess</span><span class="p">,</span> <span class="n">tf</span><span class="o">.</span><span class="n">train</span><span class="o">.</span><span class="n">latest_checkpoint</span><span class="p">(</span><span class="s1">'./'</span><span class="p">))</span>

    <span class="n">result</span><span class="o">=</span> <span class="n">sess</span><span class="o">.</span><span class="n">run</span><span class="p">(</span><span class="n">analyze_prediction</span><span class="p">,</span> <span class="n">feed_dict</span><span class="o">=</span><span class="p">{</span><span class="n">x</span><span class="p">:</span> <span class="n">new_pics_x</span><span class="p">,</span> <span class="n">keep_prob</span><span class="p">:</span> <span class="mf">1.0</span><span class="p">})</span>
    <span class="nb">print</span><span class="p">(</span><span class="n">result</span><span class="p">)</span>

</pre>

</div>

</div>

</div>

</div>

<div class="output_wrapper">

<div class="output">

<div class="output_area">

<div class="output_subarea output_stream output_stdout output_text">

<pre>TopKV2(values=array([[  1.00000000e+00,   6.61333424e-22,   2.36539046e-22,
          1.82361468e-26,   7.04956766e-27],
       [  1.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          0.00000000e+00,   0.00000000e+00],
       [  1.00000000e+00,   7.74026503e-12,   2.57123072e-13,
          1.34087204e-14,   1.18607610e-15],
       [  1.00000000e+00,   3.25684981e-13,   8.77288032e-15,
          1.18839040e-15,   1.04357085e-15],
       [  9.25975144e-01,   6.42549992e-02,   6.10099407e-03,
          2.31874920e-03,   1.32016768e-03],
       [  9.91974890e-01,   3.94156063e-03,   3.59344180e-03,
          3.30875599e-04,   1.07546279e-04],
       [  1.00000000e+00,   1.51473651e-22,   6.61753818e-25,
          4.88173567e-26,   1.24016195e-27]], dtype=float32), indices=array([[14,  3, 29,  4, 35],
       [18,  0,  1,  2,  3],
       [ 3, 38,  5, 29,  2],
       [11, 16, 30, 27, 40],
       [ 0,  8,  3, 34, 35],
       [ 1, 29,  0, 21,  8],
       [39, 33, 37, 40, 20]], dtype=int32))
</pre>

</div>

</div>

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

**Answer:**

*   There are two signs for speed limit of 20, the first of them was properly recognized with a high value of confidence ([ 0, 8, 3, 34, 35] zero is the correct value), the second one failed at the first attemp, however the correct predicion appears within the top 5 values (the third most probably [ 1, 29, 0, 21, 8]). This, as suggested in my previous comment, might be caused by the noise surrounding the picture, the angle in which the picture was taken or the similarities among the different speed limit signs. All other signs were recognized with a high level of confidence.

*   Perhaps an adjustment in dropout values is neccesary in order to increase/reduce generalization/overfitting a little bit. Also another preprocessing in the pictures in order to clean the "noise" surrounding the signs (I tried a black mask filter but did do better).

</div>

</div>

</div>

<div class="cell border-box-sizing text_cell rendered">

<div class="inner_cell">

<div class="text_cell_render border-box-sizing rendered_html">

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to \n", "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

</div>

</div>

</div>

</div>

</div>