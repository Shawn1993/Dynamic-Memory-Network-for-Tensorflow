#!/bin/bash 
if [ ! -d 'data' ];then
    mkdir 'data'
fi
cd 'data'

echo 'Downloading babi dataset ...'
if [ ! -d 'babi' ];then
    wget 'http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz' 
    tar -zxvf 'tasks_1-20_v1-2.tar.gz'
    mv 'tasks_1-20_v1-2' 'babi'
    rm 'tasks_1-20_v1-2.tar.gz'
fi
echo 'Downloaded'

if [ ! -d 'glove' ]; then
    mkdir 'glove'
fi
cd 'glove'

echo 'Downloading glove.6B ...'
if [ ! -d 'glove.6B' ]; then
    wget 'http://nlp.stanford.edu/data/glove.6B.zip'
    unzip 'glove.6B.zip' -d 'glove.6B'
    rm 'glove.6B.zip'
fi
echo 'Downloaded'