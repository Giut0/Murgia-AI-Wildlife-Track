# Wildlife Classification on Video from Phototraps in the Alta Murgia Park.
Final project for Computer Vision 24-25 course 

## Goal 🎯
Improve the efficiency of wildlife monitoring in Alta Murgia National Park by automating the process of species identification and classification and providing useful data for biodiversity management.

## Project description

This project aims to classify the fauna in videos captured by phototraps placed in the Alta Murgia National Park. 
The videos were processed automatically using an approach combining data augmentation and classification techniques. In particular, two main approaches were evaluated for the training phase:

   1. **Vision Transformer fine-tuning**: fine-tuning of the `vit_base_patch16_224` model was performed, initially keeping the backbone frozen and later allowing it to be modified during training;

   2. **Hybrid approach**: vitDet was employed as a feature extractor for images, then using these representations to train traditional classifiers.

Careful evaluation of the performance of the models followed, with the goal of identifying the one most effective in classifying the animal represented in each image.

## Origine dei dati 🗃️
I video usati per questo progetto provendono da...

## Pipeline

The wildlife classification pipeline consists of the following steps:

1.  **Video Acquisition:**
    * Videos are collected from **phototraps** placed in the Alta Murgia National Park.

2.  **Frame Extraction and Bounding Box (MegaDetector):**
    * Use of **MegaDetector** to identify events and select the representative **best frame**.
    * The **bounding boxes (bboxes)** of the animals in the extracted frames are identified.

3.  **Dataset Generation:**
    * The images and their annotations (bboxes) derived from the extracted frames are used to construct the training dataset.

4.  **Data Augmentation:**
    * Data augmentation techniques are applied to **balance the distribution of classes** in the dataset, with emphasis on underrepresented species.

5.  **Classification:**
    * Features are extracted using **ViTDet (Vision Transformer Detector)**.
    * Then, **classifiers** (Logistic regression, Decision tree and K-Nearest Neighbors) are trained on the extracted features.

6.  **Fine-tuning ViTDet:**
    * **Frozen Backbone:** Only the final levels of the ViTDet model are trained.
    * **Full Training:** The entire model, including the backbone undergoes fine-tuning.

## Valutazione dei Risultati

_Sezione in corso di completamento._

