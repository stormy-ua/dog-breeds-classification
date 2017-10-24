#!/usr/bin/env bash

mkdir -p frozen/inception
curl -o ./frozen/inception/inception.tgz http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
tar xfz ./frozen/inception/inception.tgz -C ./frozen/inception
