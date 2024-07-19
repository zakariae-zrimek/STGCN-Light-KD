# Enhanced Knee Osteoarthritis Diagnosis with 3D Skeletons Data Analysis

This repository provides the implementation of our paper "Enhanced Knee Osteoarthritis Diagnosis with 3D Skeletons Data Analysis using Light STGCN and Knowledge Distillation" (under submission).

![Project Overview](IMAGES/floawchart.png)


Figure 1: Overview of our proposed method. We employ a lightweight STGCN-GRU model with knowledge distillation to analyze 3D skeleton data for knee osteoarthritis (KOA) assessment. Our approach captures complex spatio-temporal features and focuses on relevant joints for accurate diagnosis.



### Data Preparation and Augmentation

We experimented on the Gait Dataset for Knee Osteoarthritis and Parkinson's Disease Analysis With Severity Levels. Before training and testing, for the convenience of fast data loading, the datasets should be converted to the proper format. Additionally, data augmentation techniques were applied to enhance the training process.

#### Dataset Download:
- [Gait Dataset for Knee Osteoarthritis and Parkinson's Disease Analysis With Severity Levels](https://data.mendeley.com/datasets/44pfnysy89/1)

#### Pre-processed Data on Google Drive:
Please download the pre-processed data from [Google Drive](https://drive.google.com/drive/folders/1QkDyMNmjSoko5QswwAuhBIEKEgCb4bx2?usp=drive_link) and extract the files.

Before training and testing, for the convenience of fast data loading, the datasets should be converted to the proper format. Additionally, data augmentation techniques were applied to enhance the training process.
## Requirements

- Python3 (>3.5)
- Install Tensorflow 2.0 from https://www.tensorflow.org/install
- To install other libraries simply run `pip install -r requirements.txt`

## Files
*  `train.py`: to perform training on Knee Osteoarthritis (KOA) classification.
* `data_prep_augmentation.py` : Script for data preparation and augmentation techniques used in Knee Osteoarthritis (KOA) classification.
* `graph_adjacency_processor.py` : Script for processing graph adjacency matrices, possibly used for representing relationships or connectivity in data related to Knee Osteoarthritis (KOA) classification.
* `gru_lstm_hybrid_network.py: Teacher model script implementing a hybrid architecture combining GRU and LSTM networks, tailored for Knee Osteoarthritis (KOA) classification tasks. 
* `student_model.py: Script for implementing a knowledge distillation approach, where a simplified student model learns from a more complex teacher model, specifically designed for Knee Osteoarthritis (KOA) classification tasks.

## running

* train the teacher model with the command:
    ```shell
    python train.py --ex Data --epoch 150 --batch_size 4 --cp1 stgcn --cp2 lstm
  ```
* Then train the student model through knowledge distillation:
  ```shell
    python student_model.py

    ```
* Visualize the performance of the models:
  * For the teacher model:
    ```shell
    python teacher.py
    ```
  * For the student model:
    ```shell
    python student.py
    ```

















