#!/bin/bash

name=$1
script_dir=$(realpath "$(dirname "$0")")

cd /media/$USER/ExtremePro/$USER

rosbag record -a -O $(date +"%Y-%m-%d-%H-%M-%S")_$name -x "$(grep -v -P '^#(.*)' $script_dir/blacklist.txt | xargs | sed -e 's/ /|/g')"
