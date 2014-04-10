#!/bin/bash

rsync -avz --exclude-from=${HOME}/research/etc/rsync-excludes \
    -e ssh dk@hpc.msu.edu:src/evomodel/expr/$1/$2 var/$1/
