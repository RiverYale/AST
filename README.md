# AST
[_AST: An Attention-guided Segment Transformer for Drone-based Cross-view Geo-localization_](https://www.github.com/riveryale/AST) (under reviewing)

This repository provides the trained model link and the source code for our paper. Thank you for your kind attention.

## Requirements
- Python 3.7
- GPU Memory >= 8G
- Pytorch >= 1.1.0

You can learn more about dependencies in ``requirements.txt``.

## Dataset
Please prepare [University-1652](https://github.com/layumi/University1652-Baseline) and modify the ``root`` and ``save_path`` in ``config.yaml``.

## Training & Evaluation
```
python train.py -c config.yaml
python train.py -c config.yaml --evaluate
```
You can learn more about configurations in ``config.yaml`` (default configurations) and ``train.py`` (configuration introduction)

## Trained Model
You could download the trained model at [GoogleDrive](https://drive.google.com/file/d/10yKlOG1ZnZIwRIqLDTrpbpp_dH5BVMRz/view?usp=sharing). After download, please modify the ``checkpoint`` in ``config.yaml``. Then you can evaluate the trained model by adding ``--evaluate``
<!-- 
## Citation
```
Not yet
``` -->
