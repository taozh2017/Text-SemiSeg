# Text-SemiSeg
Official code for "[Text-driven Multiplanar Visual Interaction for Semi-supervised Medical Image Segmentation](https://arxiv.org/pdf/2507.12382)", which has been **early accepted in MICCAI 2025**.

![The proposed Text-SemiSeg framework](./Figure/model.jpg)

## Installation

To set up the environment and install dependencies, run:

```bash
pip install -r requirements.txt
```

## Dataset
The data can be obtained from [Pancreas-CT](https://drive.google.com/drive/folders/1kQX8z34kF62ZF_1-DqFpIosB4zDThvPz?usp=sharing) and [BraTS](https://github.com/HiLab-git/SSL4MIS/tree/master/data/BraTS2019)

## Training
To train the model on a dataset, execute:
```bash
python train.py
```

## Prediction
After training, you can make predictions using:
```bash
python prediction.py
```

## Acknowledgements
Our code is based on [SSL4MIS](https://github.com/HiLab-git/SSL4MIS).

## Questions
If you have any questions, welcome contact me at 'taozhou.dreams@gmail.com'
