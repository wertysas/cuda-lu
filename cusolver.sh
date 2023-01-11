#!/bin/bash

#clear output file
echo -n "TYPE: RODINIA LU" > cuda-lu.txt
echo "" >> cuda-lu.txt

sizes=( 128 256 512 1024 1536 2048 2560 3072 3584 4096 )

for i in ${sizes[@]}
do
    echo "SIZE:" >> cuda-lu.txt
    for iter in {1..10}
    do
        echo $iter 
        echo $(./cuda-lu -i ../$i.dat) >> cuda-lu.txt
    done
done

echo -n "TYPE: cuSOLVER LU" > cuda-lu-cusolver.txt
echo "" >> cuda-lu-cusolver.txt

sizes=( 128 256 512 1024 1536 2048 2560 3072 3584 4096 )

for i in ${sizes[@]}
do
    echo "SIZE:" >> cuda-lu-cusolver.txt
    for iter in {1..10}
    do
        echo $iter 
        echo $(./cuda-lu-cusolver -i ../$i.dat) >> cuda-lu-cusolver.txt
    done
done

echo -n "TYPE: cuSOLVER LU STREAMED" > cuda-lu-cusolver-streaming.txt
echo "" >> cuda-lu-cusolver-streaming.txt

sizes=( 128 256 512 1024 1536 2048 2560 3072 3584 4096 )

for i in ${sizes[@]}
do
    echo "SIZE:" >> cuda-lu-cusolver-streaming.txt
    for iter in {1..10}
    do
        echo $iter 
        echo $(./cuda-lu-cusolver-streaming -i ../$i.dat) >> cuda-lu-cusolver-streaming.txt
    done
done


