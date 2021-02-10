#!/bin/bash

# Download ModelNet10 dataset (around 450MB)
wget --no-check-certificate http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
unzip ModelNet10.zip -d ./data
rm ModelNet10.zip

