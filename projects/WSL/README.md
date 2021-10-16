# Toward Joint Thing-and-Stuff Mining for Weakly Supervised Panoptic Segmentation

By [Yunhang Shen](), [Liujuan Cao](), [Zhiwei Chen](), [Feihong Lian](), [Baochang Zhang](), [Chi Su](), [Yongjian Wu](), [Feiyue Huang](), [Rongrong Ji]().

CVPR 2021 Paper.

This project is based on [Detectron2](https://github.com/facebookresearch/detectron2).

## License

JTSM is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement
We borrowed code from [Detectron2](https://github.com/facebookresearch/detectron2), [DRN-WSOD-pytorch](https://github.com/shenyunhang/DRN-WSOD-pytorch), and [Detectron](https://github.com/facebookresearch/Detectron).

## Citing JTSM

If you find JTSM useful in your research, please consider citing:

```
@InProceedings{JTSM_2021_CVPR,
	author = {Shen, Yunhang and Cao, Liujuan and Chen, Zhiwei and Lian, Feihong and Zhang, Baochang and Su, Chi and Wu, Yongjian and Huang, Feiyue and Ji, Rongrong},
	title = {Toward Joint Thing-and-Stuff Mining for Weakly Supervised Panoptic Segmentation},
	booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	year = {2021},
	pages = {16694-16705}
}   
```


## Installation

Install our forked Detectron2:
```
git clone https://github.com/shenyunhang/JTSM.git JTSM
cd JTSM
python3 -m pip install -e .
```
If you have problem of installing Detectron2, please checking [the instructions](https://detectron2.readthedocs.io/tutorials/install.html).

Install JTSM project:
```
cd projects/WSL
pip3 install -r requirements.txt
git submodule update --init --recursive
python3 -m pip install -e .
cd ../../
```


## Dataset Preparation

#### PASCAL VOC 2012:
Please follow [this](https://github.com/shenyunhang/JTSM/blob/JTSM/datasets/README.md#expected-dataset-structure-for-pascal-voc) to creating symlinks for PASCAL VOC.

Also download SBD data:
```
cd datasets/
wget --no-check-certificate http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz
tar xvzf benchmark.tgz
benchmark_RELEASE/dataset/ SBD
```

Convert VOC 2012 and SBD to our format:
```
python3 projects/WSL/tools/convert_voc2012_and_sbd_instance.py
python3 projects/WSL/tools/convert_voc2012_and_sbd_panoptic.py
python3 projects/WSL/tools/prepare_panoptic_fpn_voc2012_and_sbd.py
```

Download MCG segmentation proposal from [here](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg/) to detectron/datasets/data, and transform it to pickle serialization format:
```
cd datasets/proposals
tar xvzf MCG-Pascal-Segmentation_trainvaltest_2012-proposals.tgz
tar xvzf MCG-SBD-trainval-proposals.tgz
cd ../../
python3 projects/WSL/tools/proposal_convert.py voc_2012_train_instance datasets/proposals/MCG-Pascal-Segmentation_trainvaltest_2012-proposals datasets/proposals/mcg_voc_2012_train_instance_segmentation_d2
python3 projects/WSL/tools/proposal_convert.py voc_2012_val_instance datasets/proposals/MCG-Pascal-Segmentation_trainvaltest_2012-proposals datasets/proposals/mcg_voc_2012_val_instance_segmentation_d2
python3 projects/WSL/tools/proposal_convert.py sbd_9118_instance datasets/proposals/MCG-SBD-trainval-proposals/ datasets/proposals/mcg_sbd_9118_instance_segmentation_d2
```

#### COCO:
Please follow [this](https://github.com/shenyunhang/JTSM/blob/JTSM/datasets/README.md#expected-dataset-structure-for-coco-instancekeypoint-detection) to creating symlinks for MS COCO.

Please follow [this](https://github.com/facebookresearch/Detectron/blob/main/detectron/datasets/data/README.md#coco-minival-annotations) to download `minival` and `valminusminival` annotations.

Download MCG segmentation proposal from [here](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/mcg/) to detectron/datasets/data, and transform it to pickle serialization format:
```
cd datasets/proposals
tar xvzf MCG-COCO-train2014-proposals.tgz
tar xvzf MCG-COCO-val2014-proposals.tgz
cd ../../
python3 projects/WSL/tools/proposal_convert.py coco_2014_train datasets/proposals/MCG-COCO-train2014-proposals datasets/proposals/mcg_coco_instance_segmentation_d2
python3 projects/WSL/tools/proposal_convert.py coco_2014_val datasets/proposals/MCG-COCO-val2014-proposals datasets/proposals/mcg_coco_instance_segmentation_d2
```

Noted that above proposal conversion will take some time, which depends on the CPU performance.


## Model Preparation

Download Resnet-WS models from this [here](https://1drv.ms/f/s!Am1oWgo9554dgRQ8RE1SRGvK7HW2):
```
mv models JTSM/
```

Then we have the following directory structure:
```
JTSM
|_ models
|  |_ DRN-WSOD
|     |_ resnet18_ws_model_120.pkl
|     |_ resnet150_ws_model_120.pkl
|     |_ resnet101_ws_model_120.pkl
|_ ...
```


## Quick Start: Using JTSM

### PASCAL VOC 2012 Panoptic

#### ResNet18-WS
```
python3.8 projects/WSL/tools/train_net.py --dist-url "tcp://127.0.0.1:52044" --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-PanopticSegmentation/jtsm_WSR_18_DC5_1x.yaml OUTPUT_DIR output/jtsm_WSR_18_DC5_voc2012_sbd_`date +'%Y-%m-%d_%H-%M-%S'`
```

#### ResNet50-WS
```
python3.8 projects/WSL/tools/train_net.py --dist-url "tcp://127.0.0.1:52044" --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-PanopticSegmentation/jtsm_WSR_50_DC5_1x.yaml OUTPUT_DIR output/jtsm_WSR_50_DC5_voc2012_sbd_`date +'%Y-%m-%d_%H-%M-%S'`
```

### MS COCO Panoptic

#### ResNet18-WS
```
python3.8 projects/WSL/tools/train_net.py --dist-url "tcp://127.0.0.1:52044" --num-gpus 4 --config-file projects/WSL/configs/COCO-PanopticSegmentation/jtsm_WSR_18_DC5_1x.yaml OUTPUT_DIR output/jtsm_WSR_18_DC5_voc2012_sbd_`date +'%Y-%m-%d_%H-%M-%S'`
```

We can also run JTSM on deeper backbone, e.g., ResNet50-WS and ResNet101-WS, by modifying the configures for ResNet18-WS backbone.

### PASCAL VOC 2007 Detection

#### ResNet18-WS
```
python3.8 projects/WSL/tools/train_net.py --dist-url "tcp://127.0.0.1:52044" --num-gpus 4 --config-file projects/WSL/configs/PascalVOC-PanopticSegmentation/jtsm_WSR_18_DC5_1x_VOC2007.yaml OUTPUT_DIR output/jtsm_WSR_18_DC5_voc2007_`date +'%Y-%m-%d_%H-%M-%S'`
```
