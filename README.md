Self driving car model using Raspberry Pi and TensorFlow.

Raspberry Pi 3 Model B<br/>
Pi Camera module version 1<br/>
L298N board<br/>
DC motors<br/>
Python 3.6.5<br/>
TensorFlow 1.13.1<br/>

<b>Setting up the environment</b><br/>
Not using conda<br/>

TensorFlow 1.13.1 supports upto Python 3.7.<br/>
<ol>
<li> CUDA 10.0 (latest version)</li>
Download link : https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Windows&target_arch=x86_64 <br/>
<li> cuDNN </li>
Download link : https://developer.nvidia.com/rdp/cudnn-download <br/>
Needs login. Download the version for CUDA 10.0

<li> Install Python </li>
I worked with Python 3.6.5.
Download link : https://www.python.org/downloads/release/python-365/

<li>Create and activate a virtual environmet</li>
<ul>
  <li>Install <b>virtualenv</b> for python : <code>pip install virtualenv</code></li>  
  <li>Create a virtual environment from command prompt : <code>virtualenv foldername</code></li>
  <li>Navigate to foldername/Scripts in the command prompt and enter <code>activate</code></li>
  <li>The virtual environment is activated and you can see it's name in brackets before the path in the command prompt</li>
</ul>

<li>Install the required packages</li>
<ul>
  <li>numpy : <code>pip install numpy</code></li>
  <li>pandas : <code>pip install pandas</code></li>
  <li>openCV : <code>pip install opencv-python</code></li>
  <li>imutils : <code>pip install imutils</code></li>
  <li>matplotlib : <code>pip install matplotlib</code></li>
  <li>tensorflow : If you have a Nvidia GPU, <code> pip install tensorflow-gpu==1.13.1</code><br/>
    ,else <code> pip install tensorflow==1.13.1</code>
  </li>
  <li> spyder(IDE) : <code>pip install spyder</li>
 Note : Installation of CUDA and cuDNN is required only when you have an NVIDIA GPU, if not you can skip the installation
