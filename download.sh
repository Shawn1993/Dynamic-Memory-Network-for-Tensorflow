#!/bin/bash 
mkdir data
cd data
wget 'http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz' 
tar -zxvf 'tasks_1-20_v1-2.tar.gz'
mv 'tasks_1-20_v1-2' 'babi'
rm 'tasks_1-20_v1-2.tar.gz'