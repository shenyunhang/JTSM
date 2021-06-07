# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import MetadataCatalog
from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.coco_panoptic import register_coco_panoptic, register_coco_panoptic_separated

from .builtin_meta import _get_builtin_metadata

# fmt: off
CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
# fmt: on

# ==== Predefined datasets and splits for Flickr ==========

_PREDEFINED_SPLITS_WEB = {}
_PREDEFINED_SPLITS_WEB["flickr"] = {
    "flickr_voc": ("flickr_voc/images", "flickr_voc/images_d2.json"),
    "flickr_coco": ("flickr_coco/images", "flickr_coco/images_d2.json"),
}


def register_all_web(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_WEB.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(key),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# ==== Predefined datasets and splits for VOC_PGT ==========

_PREDEFINED_SPLITS_VOC_PGT = {}
_PREDEFINED_SPLITS_VOC_PGT["voc_2007_pgt"] = {
    "voc_2007_train_pgt": (
        "VOC2007/JPEGImages",
        "VOC2007/../results/VOC2007/Main/voc_2007_train_pgt.json",
    ),
    "voc_2007_val_pgt": (
        "VOC2007/JPEGImages",
        "VOC2007/../results/VOC2007/Main/voc_2007_val_pgt.json",
    ),
}


def register_all_voc_pgt(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_VOC_PGT.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(key),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# ==== Predefined datasets and splits for VOC_SBD ==========

_PREDEFINED_SPLITS_VOC_SBD = {
    "voc_2012_train_instance": (
        "VOC_SBD/images",
        "VOC_SBD/annotations/voc_2012_train_instance.json",
    ),
    "voc_2012_val_instance": ("VOC_SBD/images", "VOC_SBD/annotations/voc_2012_val_instance.json"),
    "sbd_9118_instance": ("VOC_SBD/images", "VOC_SBD/annotations/sbd_9118_instance.json"),
    "voc_2012_train_instance_pgt": (
        "VOC_SBD/images",
        "VOC_SBD/annotations/voc_2012_train_instance_pgt.json",
    ),
    "sbd_9118_instance_pgt": ("VOC_SBD/images", "VOC_SBD/annotations/sbd_9118_instance_pgt.json"),
}

_PREDEFINED_SPLITS_VOC_SBD_PANOPTIC = {
    "voc_2012_train_panoptic": (
        "VOC_SBD/annotations/panoptic",
        "VOC_SBD/annotations/voc_2012_train_panoptic.json",
        "VOC_SBD/annotations/panoptic_stuff",
    ),
    "voc_2012_val_panoptic": (
        "VOC_SBD/annotations/panoptic",
        "VOC_SBD/annotations/voc_2012_val_panoptic.json",
        "VOC_SBD/annotations/panoptic_stuff",
    ),
    "sbd_9118_panoptic": (
        "VOC_SBD/annotations/panoptic",
        "VOC_SBD/annotations/sbd_9118_panoptic.json",
        "VOC_SBD/annotations/panoptic_stuff",
    ),
}


def register_all_voc_sbd(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_VOC_SBD.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_builtin_metadata("voc_sbd"),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_VOC_SBD_PANOPTIC.items():
        prefix_instances = prefix.replace("_panoptic", "_instance")
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        # The "separated" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic FPN
        register_coco_panoptic_separated(
            prefix,
            _get_builtin_metadata("voc_sbd_panoptic_separated"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_coco_panoptic(
            prefix,
            _get_builtin_metadata("voc_sbd_panoptic_standard"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            instances_json,
        )


# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("WSL_DATASETS", "datasets")
    register_all_web(_root)
    register_all_voc_pgt(_root)
    register_all_voc_sbd(_root)
