#!/bin/bash
source /home/*****/anaconda3/etc/profile.d/conda.sh # ***** 에 리눅스 사용자 이름이 들어감.
conda activate temp                                 # Anaconda 가상 환경 active
python Training_kor2.py
conda deactivate