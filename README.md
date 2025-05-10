# Unsignalized intersection

This repo is published and maintained by [Zhenlin (Gavin) Xu](https://github.com/Zhenlin-Xu) and [Saeed Rahmani](https://github.com/SaeedRahmani).
<!-- New datasets (including Argoverse 2) are on the way.  -->

## Installation Setup

The versions of critical package dependencies are specified here. 

### Lyft

- `conda create -n lyft python=3.8 --yes`
- `pip install l5kit`

### Waymo

- `conda create -n waymo python=3.10 --yes`
- `pip install waymo-open-dataset-tf-2-12-0`

## Raw Dataset Downloading

### Lyft

Download the raw datasets including:

- [Sample dataset](https://woven.toyota/common/assets/data/prediction-sample.tar)
- [Training part 2/2 dataset](https://woven.toyota/common/assets/data/prediction-train_full.tar) (ignore training part 1 dataset since it is included in train part 2)
- [Validation dataset](https://woven.toyota/common/assets/data/prediction-validate.tar)

with [aerial map](https://woven.toyota/common/assets/data/prediction-aerial_map.tar) and [semantic map](https://woven.toyota/common/assets/data/prediction-semantic_map.tar) via https://woven.toyota/en/prediction-dataset.

Expected dataset storage structure

```
raw_data/lyft/
      +- scenes/
            +- sample.zarr
            +- train.zarr
            +- validate.zarr
      +- aerial_map/
            +- aerial_map.png
      +- semantic_map/
            +- semantic_map.pb
      +- meta.json
```

### Waymo

## Filter Unsignalized Intersections

The first step of pre-processing the trajectory datasets and extract the conflicts
is to filter out the unsignalized intersections.

### Lyft

For lyft dataset, two unsignalized intersections are found via google map,
since the AV fleet's route was fixed.

```shell
$ python ./filter_intersection_LYFT.py --help
usage: filter_intersection_LYFT.py [-h] --type {sample,train,validate} --id {0,1}

Filter intersection scenes from Lyft dataset

optional arguments:
  -h, --help            show this help message and exit
  --type {sample,train,validate}
                        Type of dataset to load, choose from 'sample', 'train', 'validate'
  --id {0,1}            ID of dataset to load, 0 for 'WTgZ' and 1 for 'sGK1'
  
$ python ./filter_intersection_LYFT.py --type sample --id 0
```

The IDs of the filtered scenes including the expected intersections,
are saved under the `./processed/lyft` folder.

`FIXME:` in future, multiprocessing should be added to accelerate this step for larger datasets such as,
train and validation.

### Waymo

## Identify Conflicts (cross and merge)

## Metrics calculation

## References

```bibtex
@misc{Woven Planet Holdings, Inc. 2020,
    title = {One Thousand and One Hours: Self-driving Motion Prediction Dataset},
    author = {Houston, J. and Zuidhof, G. and Bergamini, L. and Ye, Y. and Jain, A. and Omari, S. and Iglovikov, V. and Ondruska, P.},
    year = {2020},
    howpublished = {\url{https://woven.toyota/en/prediction-dataset}}
}
```