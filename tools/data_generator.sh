#!/bin/bash

sizes=( 128 256 512 1024 1536 2048 2560 3072 3584 4096 )

for i in ${sizes[@]}
do
    echo $(./gen_input $i)
done

