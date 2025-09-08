# Patch Classifier Training, Validation

## Environment
```
# install packages
$ pip install -r requirements.txt
```

## Training
```
# Example of data
.
├── train
│   ├── background
│   │    ├── image1
│   │    ├── ...
│   │    └── imageN
│   ├── foreground
│   │    ├── image1
│   │    ├── ...
│   │    └── imageN
└── val
    ├── background
    │    ├── image1
    │    ├── ...
    │    └── imageN
    └── foreground
         ├── image1
         ├── ...
         └── imageN

# run training
$ python3 run_train.py --patch_size 28 --n_features 256 --gpu_index 0 --dir_train_data <path/to/train/dir> --dir_val_data <path/to/val/dir>
```