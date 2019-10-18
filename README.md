# In-vivo optoacoustic sparse reconstruction artefact removal with U-Net 

This repository contains the code of sparse reconstruction artefact removal with convolutional neural network that was employed in our work:

## Requirements

See environment.yml for the conda requirements to run the test code. In order to install the required version of lasagne run the following:

```
pip install https://github.com/Lasagne/Lasagne/archive/master.zip
```

## Running the code


### Training


### Testing

You can download a trained model by downloading the following, and placing it in the folder "demo/model/", with the filename "16843907".

[https://ndownloader.figshare.com/files/16843907](https://ndownloader.figshare.com/files/16843907)

The requirements for "test.py" can be easily satisfied by installing conda ([conda.io](conda.io)), and creating a new environment which satisfies the requirements listed in "environment.yml".

```
conda env create -f environment.yml
conda activate unet3
```

Then by running "test.py".
```
python test.py
```
This will use the downloded trained model and provided sample test data to test the network. Sample test data includes the
network input as artefactual sparse recostruction images, "test_32.mat", and ground truth artefact-free full reconstruction, "test_GT.mat", to be compared with network 
output for performance evaluation.

## Some results from our paper
