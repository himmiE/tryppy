# Tryppy

**Tryppy** is an open-source python package designed to simplify segmentation, feature extraction, and classification of microscopy data from *Trypanosoma brucei*.  
It has been developed specifically for the [TrypTag](http://tryptag.org/) dataset. The performance on other microscopy datasets has not been evaluated and may vary.

## How to install
We provided several options to make the functionality of tryppy available to you. You can choose to install the package via pip or github into your python environment. We are also working on making the code available via Docker.

### From Pypi
```pip install tryppy```

Here you find the [official pypi website](https://pypi.org/project/tryppy/0.1.0/).

### From Github
```pip install git+https://github.com/himmiE/tryppy.git```

Here you get to the [git repository](https://github.com/himmiE/tryppy).

### Using Docker Hub

*Comming soon*

## How to use
To use tryppy efficiently you should look into the functionality of the *config.json* file.
A default version of this file is found in the ressources folder of this package and will
be used automatically, when no custom file is provided.
This file is to be edited by you whenever you want to change something about the workflow.
It is probably a good idea to save changes in a new config file, in case you break something.
You can also change the filename to have different versions.
You will be able to choose the right config file later in your code.

### The Config File
When you first run your code a basic config-file will be created for you.

| option               |                                            description                                             |
|----------------------|:--------------------------------------------------------------------------------------------------:|
| input_folder_name    |            name of the folder inside your workdir, in which your input data is located             |
| keep_input_filenames | true: keep original filename as prefix for output to better assign them to the corresponding input |
| weights_path         |                     path where the weights-file for the unet model is located                      |
| model_url            |          the url from which to load the model when it is not preloaded (when using pypi)           |
| output_dir           |            name of the folder inside your workdir, in which your output data is located            |

Under tasks you can find full_image_masks, crop, feature extraction and classification.
Using the enable option, you can decide which tasks to perform.
For every enabled task, make sure to only enable saving of the data,
if you are interested in the data, to save operation time and memory.

### Workflow
Tryppies workflow can include up to 4 steps:
segmentation of the cells, cropping of single cells, feature extraction and classification.

cells -> cropping -> feature extraction -> classification

The input data required depends on the transformation tasks that are to be performed.
When starting with the segmentation task, a tiff file with 3 channels is expected.

### Use In Code


```
from tryppy.tryppy import Tryppy

# define the path for your data.
# if you have a custom config.json file, it should go here.
data_path = directory_where_data_should_do

tryppy_instance = Tryppy(data_path, config_filename = 'my_config.json')
tryppy_instance.run()
```


## License
Distributed under the MIT License. See LICENSE.txt for more information.

