# Animal Classification in Camera Trap Images

Example repository for the CV4Ecology 2022 Summer School to train and test deep
learning models on camera trap images, using the [CCT20 dataset](https://lila.science/datasets/caltech-camera-traps).

## Installation instructions

1. Install [Conda](http://conda.io/)

2. Create environment and install requirements

```bash
conda create -n cv4ecology python=3.8 -y
conda activate cv4ecology
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
