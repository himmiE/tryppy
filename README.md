# NAME_OF_THE_PACKAGE

NAME_OF_THE_PACKAGE is an open source python package. It helps to easily perform segmentation, SOME_FEATURES and classification on microscopy data of trypanosoma brucei. The package has been build specifically on the tryptag dataset. The quality of the results on other microscopy data or even on other () has not been varified by us.

## How to install
We provided several options to make the functionality of NAME_OF_THE_PACKAGE available to you. You can choose to install the package via pip or github into your python environment. Alternatively you can choose to use our code via the provided Docker container. This is especially usefull for an easy to use proof of concept, when trying out new data.

### How to install with pip
show installation
Here you find the [official pip website](TODO).

### How to install with github download
show installation

### How to install using docker
show installation

## How to use
No matter which approach you choose, you may want to look at the *config.json* file. (describe where it is to be found) This file is to be edited by you whenever you want to change something about the workflow. It is probably a good idea to save changes in a new config file, in case you break something. You can also change the filename to have different versions. You will be able to choose the right config file later in your code.

### The Config File

| option        | description   | example_input |
| ------------- |:-------------:|---------------|
| left foo      | right foo     |![This is an alt text.](/image/sample.webp "This is a sample image.")|
| left bar      | right bar     ||
| left baz      | right baz     ||

### The Data

### Use In Code


```
import NAME_OF_PACKAGE

obj_name = NAME_OF_PACKAGE(path/to/config/file.json)
# can be kept empty when you haven't changed name or location of the config file

obj_name.run()

```


## License

TODO