# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.meta_arch.panoptic_fpn import combine_semantic_and_instance_outputs
from detectron2.modeling.meta_arch.semantic_seg import build_sem_seg_head
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage

from ..postprocessing import detector_postprocess, sem_seg_postprocess

__all__ = ["GeneralizedMCNNWSL"]


@META_ARCH_REGISTRY.register()
class GeneralizedMCNNWSL(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        sem_seg_head: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        has_cpg: bool,
        combine_on: bool,
        combine_overlap_threshold: int,
        combine_stuff_area_limit: int,
        combine_instances_confidence_threshold: int,
        ps_on: bool,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.sem_seg_head = sem_seg_head

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        self.has_cpg = has_cpg

        # options when combining instance & semantic outputs
        self.combine_on = combine_on
        self.combine_overlap_threshold = combine_overlap_threshold
        self.combine_stuff_area_limit = combine_stuff_area_limit
        self.combine_instances_confidence_threshold = combine_instances_confidence_threshold

        self.ps_on = ps_on

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "sem_seg_head": build_sem_seg_head(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "has_cpg": True
            if "CSC" in cfg.MODEL.ROI_HEADS.NAME or "WSJDS" in cfg.MODEL.ROI_HEADS.NAME
            # or "UWSODROIHeads" in cfg.MODEL.ROI_HEADS.NAME
            else False,
            # options when combining instance & semantic outputs
            "combine_on": cfg.MODEL.PANOPTIC_FPN.COMBINE.ENABLED,
            "combine_overlap_threshold": cfg.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH,
            "combine_stuff_area_limit": cfg.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT,
            "combine_instances_confidence_threshold": (
                cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH
            ),
            "ps_on": cfg.WSL.PS_ON,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.

                For now, each item in the list is a dict that contains:

                * "image": Tensor, image in (C, H, W) format.
                * "instances": Instances
                * "sem_seg": semantic segmentation ground truth.
                * Other information that's included in the original dicts, such as:
                  "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                each dict is the results for one image. The dict contains the following keys:

                * "instances": see :meth:`GeneralizedRCNN.forward` for its format.
                * "sem_seg": see :meth:`SemanticSegmentor.forward` for its format.
                * "panoptic_seg": available when `PANOPTIC_FPN.COMBINE.ENABLED`.
                  See the return value of
                  :func:`combine_semantic_and_instance_outputs` for its format.
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        if self.has_cpg:
            images.tensor.requires_grad = True

        if "sem_seg" in batched_inputs[0]:
            gt_sem_seg = [x["sem_seg"].to(self.device) for x in batched_inputs]
            gt_sem_seg = ImageList.from_tensors(
                gt_sem_seg, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
        else:
            gt_sem_seg = None

        features = self.backbone(images.tensor)

        assert "superpixels" in batched_inputs[0]
        superpixels = self.preprocess_superpixel(batched_inputs)
        assert "proposals" in batched_inputs[0]
        proposals = [x["proposals"].to(self.device) for x in batched_inputs]

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances, gt_sem_seg, superpixels
        )
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        if self.has_cpg:
            images.tensor.requires_grad = False
            images.tensor.detach()

        _, sem_seg_losses = self.sem_seg_head(features, self.roi_heads.pgt_sem_seg)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        losses.update(sem_seg_losses)
        return losses

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
        only_sem_seg: bool = False,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        assert "superpixels" in batched_inputs[0]
        superpixels = self.preprocess_superpixel(batched_inputs)
        assert "proposals" in batched_inputs[0]
        proposals = [x["proposals"].to(self.device) for x in batched_inputs]

        if only_sem_seg:
            sem_seg_results, _ = self.sem_seg_head(features, None)
            return sem_seg_results, _, _

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _, all_scores, all_boxes = self.roi_heads(
                images, features, proposals, None, None, superpixels
            )
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results, all_scores, all_boxes = self.roi_heads.forward_with_given_boxes(
                features, detected_instances
            )

        sem_seg_results, _ = self.sem_seg_head(features, None)

        if do_postprocess and self.ps_on:
            return self._postprocess_ps(
                sem_seg_results, results, batched_inputs, images.image_sizes
            )

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedMCNNWSL._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results, all_scores, all_boxes

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def preprocess_superpixel(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["superpixels"].to(self.device) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: Tuple[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def _postprocess_ps(self, sem_seg_results, detector_results, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        processed_results = []
        for sem_seg_result, detector_result, input_per_image, image_size in zip(
            sem_seg_results, detector_results, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            sem_seg_r = sem_seg_postprocess(sem_seg_result, image_size, height, width)
            detector_r = detector_postprocess(detector_result, height, width)

            processed_results.append({"sem_seg": sem_seg_r, "instances": detector_r})

            if self.combine_on:
                panoptic_r = combine_semantic_and_instance_outputs(
                    detector_r,
                    sem_seg_r.argmax(dim=0),
                    self.combine_overlap_threshold,
                    self.combine_stuff_area_limit,
                    self.combine_instances_confidence_threshold,
                )
                processed_results[-1]["panoptic_seg"] = panoptic_r
        return processed_results
