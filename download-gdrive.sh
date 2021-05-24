#!/bin/bash
fileid="1EyUHTaKV6n4F9tVzsuoFQbubPQJlR0E5"
filename="ucf101_features.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
