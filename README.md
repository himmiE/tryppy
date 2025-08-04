# tryppy

**Tryppy** is an open-source Python package designed to simplify segmentation, feature extraction, and classification of microscopy data from *Trypanosoma brucei*.  
It has been developed specifically for the **TrypTag** dataset. The performance on other microscopy datasets has not been evaluated and may vary.

## How to install
We provided several options to make the functionality of tryppy available to you. You can choose to install the package via pip or github into your python environment. Alternatively you can choose to use our code via the provided Docker container. This is especially usefull for an easy to use proof of concept, when trying out new data.

### From Pypi
```pip install tryppy```

Here you find the [official pypi website](TODO).

### From Github
```pip install git+https://github.com/himmiE/tryppy.git```

### Using Docker Hub (recommended)
An official Docker image is available to run `tryppy` without installing Python or any dependencies locally.
If you have Docker installed, you can pull the latest image with:

```docker build -t tryppy:latest .```

```docker run tryppy:latest```

This launches an interactive Python shell with `tryppy` preinstalled.
## How to use
No matter which approach you choose, you may want to look at the *config.json* file. (describe where it is to be found) This file is to be edited by you whenever you want to change something about the workflow. It is probably a good idea to save changes in a new config file, in case you break something. You can also change the filename to have different versions. You will be able to choose the right config file later in your code.

### The Config File
When you first run your code a basic config-file will be created for you. ...

| option        | description   | example_input |
| ------------- |:-------------:|---------------|
| left foo      | right foo     |![This is an alt text.](/image/sample.webp "This is a sample image.")|
| left bar      | right bar     ||
| left baz      | right baz     ||

### The Data

### Use In Code


```
import NAME_OF_PACKAGE

# define the path for your data.
# if you have a custom config.json file, it should go here.
data_path = directory_where_data_should_do

# if you are handling multiple tasks with this package or are experimenting
# with different setups, you can rename the config file and pass the filename
# (relative to your defined data_path).

obj_name = NAME_OF_PACKAGE(data_path)
# alt: 
# obj_name = NAME_OF_PACKAGE(data_path, config_filename = 'save_all_config.json')

obj_name.run()

```


## License

TODO

