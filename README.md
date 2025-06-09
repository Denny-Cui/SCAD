# SCAD: A Lightweight Recommendation Model based on Multi-interest

[Xiaotong Cui](cxt20030114@gmail.com), [Nan Wang*](wangnan@hlju.edu.cn)

## Prerequisites

- Python 3
- torch>=2.0
- d2l

## Getting Started

### Installation

- Install pytorch 2.x
- Install the d2l package based on the instructions here: https://d2l.ai/chapter_installation/index.html
- Clone this repository `git clone https://github.com/Denny-Cui/SCAD.git`

### Dataset

The two datasets that we used in our experiments are included.

### Training

#### Train on the exisiting datasets

you can use `python3 main.py --p train --dataset {dataset_name}` to train SCAD on dataset. Other hyperparameters can be found in the code.

For example, you can use `python3 main.py --p train --dataset lfm` to train SCAD model on the Lfm-1b dataset.

#### Train on your own datasets

If you want to train models on your own datasets, you should prepare the following two properly formatting files.

- user_item_8: nine numbers in a line `<user_id>` `<item_id1> ... <item_id8>` (a user with 8 item that it interacted with)
- user_item_neigh: ten numbers in a line `<user_id>` `<neigh_id>` `<item_id1> ... <item_id8>` (a user and a neighbor with 8 items that the neighbor interact with)

### Testing

To test a well trained SCAD model, you should finish training it on the dataset that you are going to test, and save the model in local files.

#### Test on the exisiting datasets

you can use `python3 main.py --p test --dataset {dataset_name}` to train SCAD on dataset. Other hyperparameters can be found in the code.

For example, you can use `python3 main.py --p test --dataset lfm` to train SCAD model on the Lfm-1b dataset.

#### Train on your own datasets

If you want to test models on your own datasets, you should prepare the two properly formatting files mentioned above as well.
