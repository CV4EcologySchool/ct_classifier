# Animal Classification in Camera Trap Images

Example repository for the CV4Ecology 2022 Summer School to train and test deep
learning models on camera trap images, using the [CCT20 dataset](https://lila.science/datasets/caltech-camera-traps).

## Installation instructions

1. Install [Conda](http://conda.io/)

2. Create environment and install requirements

You will need to install the proper version of PyTorch and torchvision for your CUDA version. The 2023 cohort should use the following. If you are using a different machine than the 2023 cohort, replace the `pip install torch=.....` command below with the appropriate one from [here](https://pytorch.org/get-started/locally/).

```bash
conda create -n cv4ecology python=3.9 -y
conda activate cv4ecology
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

3. Download dataset

**NOTE:** Requires the [azcopy CLI](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10) to be installed and set up on your machine.

```bash
sh scripts/download_dataset.sh 
```

This downloads the [CCT20](https://lila.science/datasets/caltech-camera-traps) subset to the `datasets/CaltechCT` folder.


## Reproduce results

1. Train

```bash
python ct_classifier/train.py --config configs/exp_resnet18.yaml
```

2. Test/inference

@CV4Ecology participants: Up to you to figure that one out. :)

See also ct_classifier/visualize_predictions.ipynb
