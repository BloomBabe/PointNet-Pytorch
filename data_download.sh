#!/bin/bash

# Download original ShapeNetPart dataset (around 700MB)
wget --no-check-certificate https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
unzip shapenetcore_partanno_segmentation_benchmark_v0_normal.zip -d ./data
rm shapenetcore_partanno_segmentation_benchmark_v0_normal.zip

