#!/usr/bin/env bash

mkdir -p data/stanford_ds
curl -o data/stanford_ds/images.tar http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
tar xfz data/stanford_ds/images.tar -C data/stanford_ds/
curl -o data/stanford_ds/annotation.tar http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar
tar xfz data/stanford_ds/annotation.tar -C data/stanford_ds/
