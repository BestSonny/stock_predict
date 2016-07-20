# For stock prediction
Pan He
## Description

This is an experimental project for stock prediction

## Installation

```bash
# in a terminal, run the commands
sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git
sudo pip install Theano
sudo pip install keras
sudo pip install pandas
```

## Running
```bash
THEANO_FLAGS=mode=FAST_RUN,device=gpu5,floatX=float32 python train_test.py
```
