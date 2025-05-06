# Unsignalized intersection

This repo is published and maintained by [Zhenlin (Gavin) Xu](https://github.com/Zhenlin-Xu) and [Saeed Rahmani](https://github.com/SaeedRahmani).
<!-- New datasets (including Argoverse 2) are on the way.  -->

## Installation Setup

The verisions of critical package dependencies are specified here. 

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

```
raw_data/lyft/
      +- scenes/
            +- sample.zarr
                  +- train.zarr
                  +- train_full.zarr
      +- aerial_map/
            +- aerial_map.png
      +- semantic_map/
            +- semantic_map.pb
      +- meta.json
```

### Waymo

## References

```bibtex
@misc{Woven Planet Holdings, Inc. 2020,
    title = {One Thousand and One Hours: Self-driving Motion Prediction Dataset},
    author = {Houston, J. and Zuidhof, G. and Bergamini, L. and Ye, Y. and Jain, A. and Omari, S. and Iglovikov, V. and Ondruska, P.},
    year = {2020},
    howpublished = {\url{https://woven.toyota/en/prediction-dataset}}
}
```