#  Magnetisation_reconstruction

Magnetization reconstruction is a package allowing the reconstruction of the source quantity from the measured magnetic field. 
The source quantity can be the magnetisation or the current density in 2 dimensions.
The reconstruction of current density in 3D is in progress.
The task is completed by a untrained physics informed neural networks that learn on the fly on each new single image.


### Table of Contents
* [0. Structure](#0-Structure)
* [1. Installation and Requirements](#1-Installation and Requirements)
  * [1.1. Required Libraries](#11-Required Libraries)
  * [1.2. Installation](#12-installation)
  * [1.3. GPU Processing](#13-gpu-processing)
* [2. Usage](#2-Usage)
  * [2.1. Data format](#21-Data format)
  * [2.2. Measurements parameters](#22- Measurements parameters)
  * [2.3. Networks](#23-Networks)
  * [2.4. Training parameters](#13-raining parameters)
* [3. Admin](#3-Admin)
  * [3.1 Citation](#31-Documentation)
  * [3.2. Documentation](#32-Documentation)
  * [3.3. Collaboration](#33-Contributors)
  * [3.4. License](#34-License)
  * [3.5. Contributors](#35-Contributors)



##	0. Structure

```
Magnetisation_reconstruction/
└── 2D/ # contains the reconstruction of magnetisation and current density in 2 dimensions
    └──data/
    └──Magnetisation/
    	├── Evaluate.py 	#Evaluate and plot the results obtain at the end of the training
        └── Generator.py 	#contains the different neural networks for the inverse solutions   
        └── Propagator.py 	#contains the fourier transform for the forward solutions
        ├── Train.py		#trains a network from Generator.py thanks to its corresponding forward solution from Propagator.py
        └── utils.py		#load the data
    └──Notebooks/
    	├── Experiment
        └── Test Current Density Reconstruction
        └── Test Magnetisation Reconstruction
        └── Test Propagation
└── 3D/ # contains the reconstruction of current density in 3 dimensions
```

## 1.Installation and Requirements

### 1.1. Installation

The software cloned with:
```
git@gitlab.com:qnami-labq/magnetisation_reconstruction.git
```

### 1.1. Required libraries
The system requires the following:

	
- [Python](https://www.python.org/downloads/): Python 3 by default (works for python 2, but no future guarantees).
- [Pytorch](https://www.pytorch.org/): The Deep Learning library for back end.
- [matplotlib](http://matplotlib.org/): visualization library
- [matplotlib-scalebar](https://pypi.org/project/matplotlib-scalebar/): Provides a new artist for matplotlib to display a scale bar
- [numpy](http://www.numpy.org/) : General purpose array-processing package.
- [tqdm](http://www.tqdm.org/) : progress bar


```
pip install -r requirements.txt
```
#### Install using Conda or a Virtual Environment
If you do not have sudo/root privileges on a system, we suggest you install using Conda a Virtualenv.
From a **bash shell**, create a virtual environment in a folder that you wish.

Using **Conda**:
```bash
conda create -p FOLDER_FOR_ENVS/ve_tf_dmtf python=3.6.5 -y
source activate FOLDER_FOR_ENVS/ve_tf_dmtf
```

Using **Virtualenv**:
```bash
virtualenv -p python3 FOLDER_FOR_ENVS/ve_tf_dmtf     # Use python up to 3.6
source FOLDER_FOR_ENVS/ve_tf_dmtf/bin/activate       # If using csh, source ve_tf_dmtf/bin/activate.csh
```


### 1.3. GPU processing

As the network learn a single image, it can be run on the cpu without any problem.

##  2. Usage

Simple example of use can be found in notebooks test notebooks (E.g. Test Magnetisation Reconstruction)



### 2.1. Data format
The input file should be in json format with the following structure:
```
{
"MagneticFieldImage (2D double array)": magnetic field image 
"[Tesla] PixleSizeX (double)": size of the pixels [meters] 
"PixleSizeY (double)": size of the pixels [meters]
"NVTheta (double)": Angle from z [degrees] 
"NVPhi (double)": in-plane angle, relative to the image [degrees] 
"NVHeight (double)": Scanning height from the sample [meters] 

#Optional (better if we know these values though) 

"MagnetisationType (string)": In-plane or out-of-plane 
"MagnetisationAngle (double)": If in-plane and the magnetisation has been set by an external field. We don't need this but it is a good check. 

#House keeping info 

"SampleComposition (string)": what the sample is 
"SampleIdentification (string)": Unique identifier for the sample 
"SampleSource (string)": The research institute/group did the sample came from 
"SamplePI (string)": Name(s) of the PI(s) for the sample fabrication 
"SampleFabricator (string)": Name of person who made the sample 
"MeasurementIdentification (string)": Unique identifier for the measurement 
"MeasurementPersonel (string)": name of person/people who took the measurement 
"MeasurementInstitute (string)": The name of the institute where the measurement was taken. 
"MeasurementPI (string)": Name(s) of the PI(s) for the measurement 
"MeasurementApparatus (string)": Widefield or scanning include the name of the setup 
"MeasurementType (string)": the measurement sequence type (ODMR, pulsed ODMR, Ramsey, etc)
}
```

### 2.2 Measurements parameters
The parameters of the measurements are directly taken from the input file. 
They can be modified or infered by the model.

```
optional arguments:
	--['NV']['Height'] = Height of NV measurement
	--['PixelSize'] = size of pixel
	--['ImageShape']= size of the image

	--['NV']['Theta'] = Nv theta angle
	--['NV']['FindTheta']= False #True to infer the nv theta angle
	--['NV']['Phi']= Nv phi angle
	--['NV']['FindPhi']= False #True to infer the nv phi angle

	--['Magnetisation']['Theta']=  magnetisation theta angle
	--['Magnetisation']['FindTheta']= False  #True to infer the magnetisation theta angle
	--['Magnetisation']['Phi'] =  magnetisation phi angle
	--['Magnetisation']['FindPhi']= False #True to infer the magnetisation Phi angle
  
```


### 2.3 Networks 
Different networks are available depending of the reconstruction task(magnetisation or current density)

```
1) generator_CNN : Convolutionnal network for magnetization

optional arguments:
  --Size            size of the network (1,2,3)
  --ImageSize		size of the image
  --kernel  		size of the kernel
  --stride 			size of the stride


2)generator_MLP : Fully connected network for magnetization

optional arguments:
  --Size            size of the image


3)generator_CNN_J : Fully connected for current density

optional arguments:
  --Size            size of the network (1,2,3)
  --ImageSize		size of the image
  --kernel  		size of the kernel
  --stride 			size of the stride
```


### 2.4. Training parameters
the training can be controlled with the following options :
```
	--['mlp']=True for fully connected network
	--['LossFunction']= L1 or L2 
	--['Magnetization']= True for a given magnetization to compare with
	--['IntegerOnly']= True to output only integer values 
	--['PositiveMagnetisationOnly']= True to output only postive values
	--['PrintLossValue']= True to print loss values
	--['Epochs']= Number of epochs
```

## 3. Admin

#### 3.1. Citation
If you are publishing scientific results, mentioning this package in your methods description is the least you can do as good scientific practice. 
You should cite our paper: 

"Untrained physically informed neural network for image reconstruction of magnetic field sources"
by A. E. E. Dubois, D. A. Broadway, A. Stark, et al.


### 3.2. Documentation

Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements

### 3.3 Collaboration

If you wanna collaborate or have questions please contact one of the contributors.

For AI related questions: Adrien Dubois (adr.dubois@gmail.com)<br>
For physics related questions: David Broadway (davidaaron.broadway@unibas.ch)<br>
For (commercial) license related question: Alexander Stark (software@qnami.ch)<br>

### 3.4. License

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.

Copyright (c) the Developers. See the COPYRIGHT.txt file at the
top-level directory of this distribution.

**For the sake of clarity and the rights carried by this license, Qnami AG 
grants to the user a non-exclusive, free and non-commercial license on all 
patents filed in its name relating to the open-source code (the "Patents") 
for the sole purpose of evaluation, development, research, prototyping 
and experimentation.**


### 3.5. Contributors

Adrien Dubois (adr.dubois@gmail.com)<br>
David Broadway (davidaaron.broadway@unibas.ch)<br>
Alexander Stark (software@qnami.ch)<br>

[1] **David Aaron Broadway**, Lillie S.E., Scholten S.C.e, Rohner D.,  D. , Dontschuk N. , Maletinsky P. , Tetienne J.-P. , Hollenberg, L.C.L. “[Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements][paper1]”, *American Physical Society, 2020*.


[//]: # (reference links)

   [paper1]: <https://link.aps.org/doi/10.1103/PhysRevApplied.14.024076>[paper1]
