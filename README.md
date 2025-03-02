# HSLabeling: Towards Efficient Labeling for Large-scale Remote Sensing Image Segmentation with Hybrid Sparse Labeling (HSLabeling)
Code for TIP 2025 paper, [**"HSLabeling: Towards Efficient Labeling for Large-scale Remote Sensing Image Segmentation with Hybrid Sparse Labeling"**]
Authors: Jiaxing Lin, <a href="https://scholar.google.com.hk/citations?user=y0r9pGQAAAAJ">Zhen Yang</a>, <a href="https://scholar.google.com/citations?user=U3dQBVMAAAAJ">Qiang Liu</a>, Yinglong Yan, <a href="https://scholar.google.com/citations?user=Gr9afd0AAAAJ&hl=en">Pedram Ghamisi</a>, <a href="https://scholar.google.com/citations?user=y0ha5lMAAAAJ">Weiying Xie</a>, and <a href="https://scholar.google.com/citations?hl=en&user=Gfa4nasAAAAJ">Leyuan Fang</a>

## Getting Started
### Prepare Dataset
Download the Potsdam and Vaihingen [<b>datasets</b>](https://drive.google.com/drive/folders/1CiYzJyBn1rV-xsrsYQ6o2HDQjdfnadHl) after processing.

you can download the datasets from the official [<b>website</b>](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx) and [<b>link</b>](https://github.com/Junjue-Wang/LoveDA). Then, crop the original images and create labels following our code in [<b>Dataprocess</b>](https://github.com/linjiaxing99/HSLabeling/tree/master/DataProcess).

If your want to run our code on your own datasets, the pre-process code is also available in [<b>Dataprocess</b>](https://github.com/linjiaxing99/HSLabeling/tree/master/DataProcess).

## Evaluate
### 1. Download the [<b>Potsdam and Vaihingen datasets</b>](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx) and [<b>LoveDA datasets</b>](https://github.com/Junjue-Wang/LoveDA)
### 2. Download our weight
### 3. Run our code
```bash
python predict.py
```

## Train 
### 1. Generate SAM labels
```bash 
python segment-anything/notebooks/automatic_mask_generator.py
```
### 2. Generate prospective scribble and block labels base on SAM labels
```bash 
python DataProcess/sparse_label_generator.py
```
### 3. Generate hybrid sparse labels and Train segmentation model
```bash 
python run/point/generate_train.py.py
```

## Acknowledgement
We thank [<b>DBFNet</b>](https://github.com/Luffy03/DBFNet) and [<b>Segment Anything</b>](https://github.com/facebookresearch/segment-anything) for part of their codes, processed datasets, data partitions, and pretrained models.


