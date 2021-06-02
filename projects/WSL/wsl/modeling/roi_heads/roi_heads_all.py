# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import torch
from torch import nn

from detectron2.layers import ShapeSpec, batched_nms
from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler, convert_boxes_to_pooler_format
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from detectron2.modeling.roi_heads.keypoint_head import (
    build_keypoint_head,
    keypoint_rcnn_inference,
    keypoint_rcnn_loss,
)
from detectron2.modeling.roi_heads.mask_head import (
    build_mask_head,
    mask_rcnn_inference,
    mask_rcnn_loss,
)
from detectron2.modeling.sampling import subsample_labels
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from wsl.layers import CSC
from wsl.modeling.mrrp_poolers import MRRPROIPooler
from wsl.modeling.roi_heads.fast_rcnn import (
    ATTOutputLayers,
    ATTOutputs,
    CSCOutputs,
    GAMOutputLayers,
    GAMOutputs,
    OICROutputLayers,
    OICROutputs,
    PCLOutputs,
    WSDDNOutputLayers,
    WSDDNOutputs,
)
from wsl.modeling.roi_heads.third_party.cpg_stats import Statistic

logger = logging.getLogger(__name__)


def pairwise_iou_wsl(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    boxes1, boxes2 = boxes1.tensor, boxes2.tensor

    width_height_outer = torch.max(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.min(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height_inner = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height_outer.abs_()  # [N,M,2]
    outer = width_height_outer.prod(dim=2)  # [N,M]

    sign = width_height_inner.clone()
    sign[sign > 0] = 1
    sign[sign < 0] = 0
    sign = sign.prod(dim=2)
    sign[sign == 0] = -1

    width_height_inner.abs_()  # [N,M,2]
    inter = width_height_inner.prod(dim=2)  # [N,M]

    # handle empty boxes
    iou = torch.where(
        outer > 0, inter / outer * sign, torch.zeros(1, dtype=inter.dtype, device=inter.device)
    )
    return iou


def select_foreground_proposals(
    proposals: List[Instances], bg_label: int
) -> Tuple[List[Instances], List[torch.Tensor]]:
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


def select_proposals_with_visible_keypoints(proposals: List[Instances]) -> List[Instances]:
    """
    Args:
        proposals (list[Instances]): a list of N Instances, where N is the
            number of images.

    Returns:
        proposals: only contains proposals with at least one visible keypoint.

    Note that this is still slightly different from Detectron.
    In Detectron, proposals for training keypoint head are re-sampled from
    all the proposals with IOU>threshold & >=1 visible keypoint.

    Here, the proposals are first sampled from all proposals with
    IOU>threshold, then proposals with no visible keypoint are filtered out.
    This strategy seems to make no difference on Detectron and is easier to implement.
    """
    ret = []
    all_num_fg = []
    for proposals_per_image in proposals:
        # If empty/unannotated image (hard negatives), skip filtering for train
        if len(proposals_per_image) == 0:
            ret.append(proposals_per_image)
            continue
        gt_keypoints = proposals_per_image.gt_keypoints.tensor
        # #fg x K x 3
        vis_mask = gt_keypoints[:, :, 2] >= 1
        xs, ys = gt_keypoints[:, :, 0], gt_keypoints[:, :, 1]
        proposal_boxes = proposals_per_image.proposal_boxes.tensor.unsqueeze(dim=1)  # #fg x 1 x 4
        kp_in_box = (
            (xs >= proposal_boxes[:, :, 0])
            & (xs <= proposal_boxes[:, :, 2])
            & (ys >= proposal_boxes[:, :, 1])
            & (ys <= proposal_boxes[:, :, 3])
        )
        selection = (kp_in_box & vis_mask).any(dim=1)
        selection_idxs = torch.nonzero(selection, as_tuple=True)[0]
        all_num_fg.append(selection_idxs.numel())
        ret.append(proposals_per_image[selection_idxs])

    storage = get_event_storage()
    storage.put_scalar("keypoint_head/num_fg_samples", np.mean(all_num_fg))
    return ret


@torch.no_grad()
def get_image_level_gt(targets, num_classes):
    if targets is None:
        return None, None, None
    gt_classes_img = [torch.unique(t.gt_classes, sorted=True) for t in targets]
    gt_classes_img_int = [gt.to(torch.int64) for gt in gt_classes_img]
    gt_classes_img_oh = torch.cat(
        [
            torch.zeros(
                (1, num_classes), dtype=torch.float, device=gt_classes_img[0].device
            ).scatter_(1, torch.unsqueeze(gt, dim=0), 1)
            for gt in gt_classes_img_int
        ],
        dim=0,
    )

    return gt_classes_img, gt_classes_img_int, gt_classes_img_oh


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(ROIHeads, self).__init__()
        # fmt: off
        self.batch_size_per_image     = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt       = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides          = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels         = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg    = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta           = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        # Box2BoxTransform for bounding box regression
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_idxs = torch.arange(gt_classes.shape[0])
        return sampled_idxs, gt_classes[sampled_idxs]

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_sample_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], ret_MI=False, suffix=""
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_sample_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []
        matched_idxs_ = []

        num_fg_samples = []
        num_bg_samples = []
        num_ig_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_ig_samples.append((gt_classes == -1).sum().item())
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1] - num_ig_samples[-1])
            proposals_with_gt.append(proposals_per_image)

            matched_idxs_.append(matched_idxs[sampled_idxs])

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples" + suffix, np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples" + suffix, np.mean(num_bg_samples))
        storage.put_scalar("roi_head/num_ig_samples" + suffix, np.mean(num_ig_samples))

        if ret_MI:
            return proposals_with_gt, matched_idxs_
        return proposals_with_gt

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        Args:
            images (ImageList):
            features (dict[str,Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

        Returns:
            list[Instances]: length `N` list of `Instances` containing the
            detected instances. Returned during inference only; may be [] during training.

            dict[str->Tensor]:
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsWSL(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[self.in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.mask_on      = cfg.MODEL.MASK_ON
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        self.box_predictor = FastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

        if self.mask_on:
            self.mask_head = build_mask_head(
                cfg,
                ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
            )

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def forward(self, images, features, proposals, targets=None):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))

        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            features = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(features, [x.pred_boxes for x in instances])
            return self.mask_head(x, instances)


@ROI_HEADS_REGISTRY.register()
class WSDDNROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(WSDDNROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg, input_shape)
        self._init_mask_head(cfg, input_shape)
        self._init_keypoint_head(cfg, input_shape)

    def _init_box_head(self, cfg, input_shape):
        # fmt: off
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = WSDDNOutputLayers(cfg, self.box_head.output_shape)

    def _init_mask_head(self, cfg, input_shape):
        # fmt: off
        self.mask_on           = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _init_keypoint_head(self, cfg, input_shape):
        # fmt: off
        self.keypoint_on  = cfg.MODEL.KEYPOINT_ON
        if not self.keypoint_on:
            return
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.keypoint_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.keypoint_head = build_keypoint_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        self.gt_classes_img, self.gt_classes_img_int, self.gt_classes_img_oh = get_image_level_gt(
            targets, self.num_classes
        )

        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances, all_scores, all_boxes = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}, all_scores, all_boxes

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])

        objectness_logits = torch.cat([x.objectness_logits + 1 for x in proposals], dim=0)
        # objectness_logits = torch.cat([x.objectness_logits for x in proposals], dim=0)
        # objectness_logits[objectness_logits > 1] = 0
        # objectness_logits[objectness_logits < -1] = 0
        # objectness_logits = objectness_logits + 1
        box_features = box_features * objectness_logits.view(-1, 1, 1, 1)

        torch.cuda.empty_cache()

        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features, proposals)
        del box_features

        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return self.box_predictor.losses(predictions, proposals, self.gt_classes_img_oh)
        else:
            pred_instances, _, all_scores, all_boxes = self.box_predictor.inference(
                predictions, proposals
            )
            return pred_instances, all_scores, all_boxes

    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            return self.mask_head(mask_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            return self.mask_head(mask_features, instances)

    def _forward_keypoint(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            keypoint_features = self.keypoint_pooler(features, proposal_boxes)
            return self.keypoint_head(keypoint_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.keypoint_pooler(features, pred_boxes)
            return self.keypoint_head(keypoint_features, instances)


@ROI_HEADS_REGISTRY.register()
class MRRPWSDDNROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(MRRPWSDDNROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg, input_shape)
        self._init_mask_head(cfg, input_shape)
        self._init_keypoint_head(cfg, input_shape)

    def _init_box_head(self, cfg, input_shape):
        # fmt: off
        self.num_branch          = cfg.MODEL.MRRP.NUM_BRANCH
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = MRRPROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = WSDDNOutputLayers(cfg, self.box_head.output_shape)

    def _init_mask_head(self, cfg, input_shape):
        # fmt: off
        self.mask_on           = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _init_keypoint_head(self, cfg, input_shape):
        # fmt: off
        self.keypoint_on  = cfg.MODEL.KEYPOINT_ON
        if not self.keypoint_on:
            return
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.keypoint_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.keypoint_head = build_keypoint_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        self.gt_classes_img, self.gt_classes_img_int, self.gt_classes_img_oh = get_image_level_gt(
            targets, self.num_classes
        )

        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances, all_scores, all_boxes = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}, all_scores, all_boxes

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.in_features]
        features = [torch.chunk(f, self.num_branch) for f in features]
        features = [ff for f in features for ff in f]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])

        objectness_logits = torch.cat([x.objectness_logits + 1 for x in proposals], dim=0)
        # objectness_logits = torch.cat([x.objectness_logits for x in proposals], dim=0)
        # objectness_logits[objectness_logits > 1] = 0
        # objectness_logits[objectness_logits < -1] = 0
        # objectness_logits = objectness_logits + 1
        box_features = box_features * objectness_logits.view(-1, 1, 1, 1)

        torch.cuda.empty_cache()

        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features, proposals)
        del box_features

        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return self.box_predictor.losses(predictions, proposals, self.gt_classes_img_oh)
        else:
            pred_instances, _, all_scores, all_boxes = self.box_predictor.inference(
                predictions, proposals
            )
            return pred_instances, all_scores, all_boxes

    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            return self.mask_head(mask_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            return self.mask_head(mask_features, instances)

    def _forward_keypoint(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            keypoint_features = self.keypoint_pooler(features, proposal_boxes)
            return self.keypoint_head(keypoint_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.keypoint_pooler(features, pred_boxes)
            return self.keypoint_head(keypoint_features, instances)


@ROI_HEADS_REGISTRY.register()
class CSCROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(CSCROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg, input_shape)
        self._init_mask_head(cfg, input_shape)
        self._init_keypoint_head(cfg, input_shape)

    def _init_box_head(self, cfg, input_shape):
        # fmt: off
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = WSDDNOutputLayers(cfg, self.box_head.output_shape)

        # CSC
        self.output_dir = cfg.OUTPUT_DIR
        self.csc_max_iter = cfg.WSL.CSC_MAX_ITER
        self.iter = 0
        self.tau = 0.7
        self.fg_threshold = 0.1
        self.bg_threshold = 0.005
        self.csc = CSC(
            tau=self.tau,
            debug_info=False,
            fg_threshold=self.fg_threshold,
            mass_threshold=0.2,
            density_threshold=0.0,
            area_sqrt=True,
            context_scale=1.8,
        )

        self.csc_stats = Statistic(
            cfg.WSL.CSC_MAX_ITER, self.tau, 4, self.num_classes, cfg.OUTPUT_DIR, ""
        )

    def _init_mask_head(self, cfg, input_shape):
        # fmt: off
        self.mask_on           = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _init_keypoint_head(self, cfg, input_shape):
        # fmt: off
        self.keypoint_on  = cfg.MODEL.KEYPOINT_ON
        if not self.keypoint_on:
            return
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.keypoint_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.keypoint_head = build_keypoint_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        self.gt_classes_img, self.gt_classes_img_int, self.gt_classes_img_oh = get_image_level_gt(
            targets, self.num_classes
        )

        # del images
        self.images = images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances, all_scores, all_boxes = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}, all_scores, all_boxes

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])

        objectness_logits = torch.cat([x.objectness_logits + 1 for x in proposals], dim=0)
        # objectness_logits = torch.cat([x.objectness_logits for x in proposals], dim=0)
        # objectness_logits[objectness_logits > 1] = 0
        # objectness_logits[objectness_logits < -1] = 0
        # objectness_logits = objectness_logits + 1
        box_features = box_features * objectness_logits.view(-1, 1, 1, 1)

        torch.cuda.empty_cache()

        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        pred_class_logits, _ = predictions
        del box_features

        cpgs = self._forward_cpg(pred_class_logits, proposals)
        W_pos, W_neg, PL, NL = self._forward_csc(cpgs, pred_class_logits, proposals)
        self._save_mask(cpgs, "cpgs", "cpg")

        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            self.iter = self.iter + 1
            return self.box_predictor.losses_csc(
                predictions, proposals, W_pos, W_neg, PL, NL, self.csc_stats
            )
        else:
            pred_instances, _, all_scores, all_boxes = self.box_predictor.inference(
                predictions, proposals
            )
            return pred_instances, all_scores, all_boxes

    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            return self.mask_head(mask_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            return self.mask_head(mask_features, instances)

    def _forward_keypoint(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            keypoint_features = self.keypoint_pooler(features, proposal_boxes)
            return self.keypoint_head(keypoint_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.keypoint_pooler(features, pred_boxes)
            return self.keypoint_head(keypoint_features, instances)

    @torch.no_grad()
    def _forward_cpg(self, pred_class_logits, proposals):
        if not self.training:
            return None

        if self.iter > self.csc_max_iter:
            return None

        pred_class_img_logits = torch.sum(pred_class_logits, dim=0, keepdim=True)
        # pred_class_img_logits = torch.clamp(pred_class_img_logits, min=1e-6, max=1.0 - 1e-6)

        image_sizes = self.images.image_sizes[0]
        cpgs = torch.zeros(
            (1, self.num_classes, image_sizes[0], image_sizes[1]),
            dtype=pred_class_img_logits.dtype,
            device=pred_class_img_logits.device,
        )
        for c in range(self.num_classes):
            if self.gt_classes_img_oh[0, c] < 0.5:
                continue
            if pred_class_img_logits[0, c] < self.tau:
                continue

            grad_outputs = torch.zeros(
                pred_class_logits.size(),
                dtype=pred_class_logits.dtype,
                device=pred_class_logits.device,
            )
            grad_outputs[:, c] = 1.0
            (cpg,) = torch.autograd.grad(  # grad_outputs[0, c] = self.pred_class_img_logits[0, c]
                outputs=pred_class_logits,
                inputs=self.images.tensor,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False,
            )
            cpg.abs_()
            cpg, _ = torch.max(cpg, dim=1)

            # cpg_scale_op
            max_value = torch.max(cpg)
            cpg = cpg / max_value

            cpgs[0, c, :, :] = cpg[0, :, :]
            del cpg
            del grad_outputs
            torch.cuda.empty_cache()

        self.images.tensor.requires_grad = False
        self.images.tensor.detach()

        return cpgs

    @torch.no_grad()
    def _forward_csc(self, masks, pred_class_logits, proposals):
        if not self.training:
            return None, None, None, None

        if self.iter > self.csc_max_iter:
            PL = self.gt_classes_img_oh
            NL = torch.zeros(
                self.gt_classes_img_oh.size(),
                dtype=self.gt_classes_img_oh.dtype,
                device=self.gt_classes_img_oh.device,
            )
            W_pos = torch.ones(
                pred_class_logits.size(),
                dtype=pred_class_logits.dtype,
                device=pred_class_logits.device,
            )
            W_neg = torch.zeros(
                pred_class_logits.size(),
                dtype=pred_class_logits.dtype,
                device=pred_class_logits.device,
            )
            return W_pos, W_neg, PL, NL

        pred_class_img_logits = torch.sum(pred_class_logits, dim=0, keepdim=True)

        pooler_fmt_boxes = convert_boxes_to_pooler_format([x.proposal_boxes for x in proposals])

        W, PL, NL = self.csc(masks, self.gt_classes_img_oh, pred_class_img_logits, pooler_fmt_boxes)

        W_pos = torch.clamp(W, min=0.0)
        W_neg = torch.clamp(W, max=0.0)

        W_pos.abs_()
        W_neg.abs_()

        return W_pos, W_neg, PL, NL

    @torch.no_grad()
    def _save_mask(self, masks, prefix, suffix):
        if masks is None:
            return
        if self.iter % 1280 > 0:
            return

        output_dir = os.path.join(self.output_dir, prefix)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        id_str = "iter" + str(self.iter) + "_gpu" + str(masks.device.index)

        for c in range(self.num_classes):
            if self.gt_classes_img_oh[0, c] < 0.5:
                continue
            # if self.pred_class_img_logits[0, c] < self.tau:
            # continue

            mask = masks[0, c, ...].clone().detach().cpu().numpy()
            max_value = np.max(mask)
            if max_value > 0:
                max_value = max_value * 0.1
                mask = np.clip(mask, 0, max_value)
                mask = mask / max_value * 255
            mask = mask.astype(np.uint8)
            im_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

            save_path = os.path.join(output_dir, id_str + "_c" + str(c) + "_" + suffix + ".png")
            cv2.imwrite(save_path, im_color)

        im = self.images.tensor[0, ...].clone().detach().cpu().numpy()
        channel_swap = (1, 2, 0)
        im = im.transpose(channel_swap)
        pixel_means = [103.939, 116.779, 123.68]
        im += pixel_means
        im = im.astype(np.uint8)
        save_path = os.path.join(output_dir, id_str + ".png")
        cv2.imwrite(save_path, im)


@ROI_HEADS_REGISTRY.register()
class CSCOICRROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(CSCROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)
        self._init_mask_head(cfg)
        self._init_keypoint_head(cfg)

        self.mean_loss = cfg.WSL.MEAN_LOSS

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = WSDDNOutputLayers(
            self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg
        )

        # CSC
        self.output_dir = cfg.OUTPUT_DIR
        self.csc_max_iter = cfg.WSL.CSC_MAX_ITER
        self.iter = 0
        self.tau = 0.7
        self.fg_threshold = 0.1
        self.bg_threshold = 0.005
        self.csc = CSC(
            tau=self.tau,
            debug_info=False,
            fg_threshold=self.fg_threshold,
            mass_threshold=0.2,
            density_threshold=0.0,
            area_sqrt=True,
            context_scale=1.8,
        )

        self.csc_stats = Statistic(
            cfg.WSL.CSC_MAX_ITER, self.tau, 4, self.num_classes, cfg.OUTPUT_DIR, ""
        )

        # OICR
        self.K = cfg.WSL.REFINE_NUM
        self.has_reg = cfg.WSL.HAS_REG
        self.has_gam = cfg.WSL.HAS_GAM
        self.box_refinery = []
        for k in range(self.K):
            box_refinery = OICROutputLayers(
                self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg
            )
            self.add_module("box_refinery_{}".format(k), box_refinery)
            self.box_refinery.append(box_refinery)

        if self.has_reg:
            self.box_refinery_reg = FastRCNNOutputLayers(
                self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg
            )

        if self.has_gam:
            self.box_gam = GAMOutputLayers(in_channels, self.num_classes)

    def _init_mask_head(self, cfg):
        # fmt: off
        self.mask_on           = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _init_keypoint_head(self, cfg):
        # fmt: off
        self.keypoint_on                         = cfg.MODEL.KEYPOINT_ON
        if not self.keypoint_on:
            return
        pooler_resolution                        = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales                            = tuple(1.0 / self.feature_strides[k] for k in self.in_features)  # noqa
        sampling_ratio                           = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type                              = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        self.normalize_loss_by_visible_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS  # noqa
        self.keypoint_loss_weight                = cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.keypoint_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.keypoint_head = build_keypoint_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        self.gt_classes_img, self.gt_classes_img_int, self.gt_classes_img_oh = get_image_level_gt(
            targets, self.num_classes
        )
        # del images
        self.images = images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        features_list = [features[f] for f in self.in_features]

        if self.training:
            losses = self._forward_box(features_list, proposals)
            # During training the proposals used by the box head are
            # used by the mask, keypoint (and densepose) heads.
            losses.update(self._forward_mask(features_list, proposals))
            losses.update(self._forward_keypoint(features_list, proposals))
            return proposals, losses
        else:
            pred_instances, all_scores, all_boxes = self._forward_box(features_list, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}, all_scores, all_boxes

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        features = [features[f] for f in self.in_features]

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        # GAM guided attention module
        if self.has_gam:
            gam = [self.box_gam(feature) for feature in features]
            assert len(features) == 1
            features = [gam[0][0]]
            pred_class_gam_logits = gam[0][1]

        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])

        objectness_logits = torch.cat([x.objectness_logits + 1 for x in proposals], dim=0)
        # objectness_logits = torch.cat([x.objectness_logits for x in proposals], dim=0)
        # objectness_logits[objectness_logits > 1] = 0
        # objectness_logits[objectness_logits < -1] = 0
        # objectness_logits = objectness_logits + 1
        box_features = box_features * objectness_logits.view(-1, 1, 1, 1)

        torch.cuda.empty_cache()

        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)

        # OICR
        pred_class_logits_K = []
        pred_proposal_deltas_K = []
        for box_refinery in self.box_refinery:
            pred_class_logits_k, pred_proposal_deltas_k = box_refinery(box_features)
            pred_class_logits_K.append(pred_class_logits_k)
            pred_proposal_deltas_K.append(pred_proposal_deltas_k)
        if self.has_reg:
            pred_class_logits_reg, pred_proposal_deltas_reg = self.box_refinery_reg(box_features)

        del box_features

        # GAM guided attention module
        if not self.training and self.has_reg:
            outputs = OICROutputs(
                self.box2box_transform,
                [pred_class_logits_reg],
                [pred_proposal_deltas_reg],
                proposals,
                self.smooth_l1_beta,
                self.mean_loss,
                None,
                "_reg",
                has_reg=True,
            )
            pred_instances, _, all_scores, all_boxes = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances, all_scores, all_boxes

        # OICR
        if not self.training:
            outputs = OICROutputs(
                self.box2box_transform,
                pred_class_logits_K,
                pred_proposal_deltas_K,
                proposals,
                self.smooth_l1_beta,
                self.mean_loss,
                None,
                "_mean",
            )

            pred_instances, _, all_scores, all_boxes = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances, all_scores, all_boxes

        cpgs = self._forward_cpg(pred_class_logits, proposals)
        W_pos, W_neg, PL, NL = self._forward_csc(cpgs, pred_class_logits, proposals)
        self._save_mask(cpgs, "cpgs", "cpg")

        outputs = CSCOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.mean_loss,
            W_pos,
            W_neg,
            PL,
            NL,
            self.csc_stats,
        )

        losses = outputs.losses()

        # OICR
        pred_class_img_logits = outputs.predict_probs_img().clone().detach()
        pred_class_img_logits = torch.index_select(
            pred_class_img_logits, 1, self.gt_classes_img_int
        ).view(-1)

        prev_pred_score = pred_class_logits.clone().detach()
        for k in range(self.K):
            prev_pred_score = torch.index_select(prev_pred_score, 1, self.gt_classes_img_int)
            pgt_idx = torch.argmax(prev_pred_score, dim=0)
            pgt_boxes = proposals[0].proposal_boxes[pgt_idx]
            pgt_classes = self.gt_classes_img

            target = Instances(proposals[0].image_size)
            target.gt_boxes = pgt_boxes
            target.gt_classes = pgt_classes
            targets = [target]

            proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals_k, matched_idxs = self.label_and_sample_proposals(
                proposals, targets, ret_MI=True, suffix="_r" + str(k)
            )
            self.proposal_append_gt = proposal_append_gt

            proposal_weights = torch.index_select(pred_class_img_logits, 0, matched_idxs[0])

            outputs = OICROutputs(
                self.box2box_transform,
                pred_class_logits_K[k],
                pred_proposal_deltas_K[k],
                proposals_k,
                self.smooth_l1_beta,
                self.mean_loss,
                proposal_weights,
                "_r" + str(k),
            )

            prev_pred_score = outputs.predict_probs()
            prev_pred_score = prev_pred_score[0].clone().detach()

            losses.update(outputs.losses())

        # GAM guided attention module
        if self.has_reg:
            prev_pred_score = torch.index_select(prev_pred_score, 1, self.gt_classes_img_int)
            pgt_idx = torch.argmax(prev_pred_score, dim=0)
            pgt_boxes = proposals[0].proposal_boxes[pgt_idx]
            pgt_classes = self.gt_classes_img

            target = Instances(proposals[0].image_size)
            target.gt_boxes = pgt_boxes
            target.gt_classes = pgt_classes
            targets = [target]

            proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals_reg, matched_idxs = self.label_and_sample_proposals(
                proposals, targets, ret_MI=True, suffix="_reg"
            )
            self.proposal_append_gt = proposal_append_gt

            proposal_weights = torch.index_select(pred_class_img_logits, 0, matched_idxs[0])

            outputs = OICROutputs(
                self.box2box_transform,
                pred_class_logits_reg,
                pred_proposal_deltas_reg,
                proposals_reg,
                self.smooth_l1_beta,
                self.mean_loss,
                proposal_weights,
                "_reg",
                has_reg=True,
            )

            prev_pred_score = outputs.predict_probs()
            prev_pred_score = prev_pred_score[0].clone().detach()

            losses.update(outputs.losses())

        if self.has_gam:
            outputs = GAMOutputs(pred_class_gam_logits, self.mean_loss, self.gt_classes_img_oh)
            losses.update(outputs.losses())

        self.iter = self.iter + 1
        return losses

        if self.training:
            return outputs.losses()
        else:
            pred_instances, _, all_scores, all_boxes = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances, all_scores, all_boxes

    def _forward_mask(self, features, instances):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            mask_logits = self.mask_head(mask_features)
            return {"loss_mask": mask_rcnn_loss(mask_logits, proposals)}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            mask_logits = self.mask_head(mask_features)
            mask_rcnn_inference(mask_logits, instances)
            return instances

    def _forward_keypoint(self, features, instances):
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (list[Tensor]): #level input features for keypoint prediction
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        num_images = len(instances)

        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            keypoint_features = self.keypoint_pooler(features, proposal_boxes)
            keypoint_logits = self.keypoint_head(keypoint_features)

            normalizer = (
                num_images
                * self.batch_size_per_image
                * self.positive_sample_fraction
                * keypoint_logits.shape[1]
            )
            loss = keypoint_rcnn_loss(
                keypoint_logits,
                proposals,
                normalizer=None if self.normalize_loss_by_visible_keypoints else normalizer,
            )
            return {"loss_keypoint": loss * self.keypoint_loss_weight}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.keypoint_pooler(features, pred_boxes)
            keypoint_logits = self.keypoint_head(keypoint_features)
            keypoint_rcnn_inference(keypoint_logits, instances)
            return instances

    @torch.no_grad()
    def _forward_cpg(self, pred_class_logits, proposals):
        if not self.training:
            return None

        if self.iter > self.csc_max_iter:
            return None

        pred_class_img_logits = torch.sum(pred_class_logits, dim=0, keepdim=True)
        # pred_class_img_logits = torch.clamp(pred_class_img_logits, min=1e-6, max=1.0 - 1e-6)

        image_sizes = self.images.image_sizes[0]
        cpgs = torch.zeros(
            (1, self.num_classes, image_sizes[0], image_sizes[1]),
            dtype=pred_class_img_logits.dtype,
            device=pred_class_img_logits.device,
        )
        for c in range(self.num_classes):
            if self.gt_classes_img_oh[0, c] < 0.5:
                continue
            if pred_class_img_logits[0, c] < self.tau:
                continue

            grad_outputs = torch.zeros(
                pred_class_logits.size(),
                dtype=pred_class_logits.dtype,
                device=pred_class_logits.device,
            )
            grad_outputs[:, c] = 1.0
            (cpg,) = torch.autograd.grad(  # grad_outputs[0, c] = self.pred_class_img_logits[0, c]
                outputs=pred_class_logits,
                inputs=self.images.tensor,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False,
            )
            cpg.abs_()
            cpg, _ = torch.max(cpg, dim=1)

            # cpg_scale_op
            max_value = torch.max(cpg)
            cpg = cpg / max_value

            cpgs[0, c, :, :] = cpg[0, :, :]
            del cpg
            del grad_outputs
            torch.cuda.empty_cache()

        self.images.tensor.requires_grad = False
        self.images.tensor.detach()

        return cpgs

    @torch.no_grad()
    def _forward_csc(self, masks, pred_class_logits, proposals):
        if not self.training:
            return None, None, None, None

        if self.iter > self.csc_max_iter:
            PL = self.gt_classes_img_oh
            NL = torch.zeros(
                self.gt_classes_img_oh.size(),
                dtype=self.gt_classes_img_oh.dtype,
                device=self.gt_classes_img_oh.device,
            )
            W_pos = torch.ones(
                pred_class_logits.size(),
                dtype=pred_class_logits.dtype,
                device=pred_class_logits.device,
            )
            W_neg = torch.zeros(
                pred_class_logits.size(),
                dtype=pred_class_logits.dtype,
                device=pred_class_logits.device,
            )
            return W_pos, W_neg, PL, NL

        pred_class_img_logits = torch.sum(pred_class_logits, dim=0, keepdim=True)

        pooler_fmt_boxes = convert_boxes_to_pooler_format([x.proposal_boxes for x in proposals])

        W, PL, NL = self.csc(masks, self.gt_classes_img_oh, pred_class_img_logits, pooler_fmt_boxes)

        W_pos = torch.clamp(W, min=0.0)
        W_neg = torch.clamp(W, max=0.0)

        W_pos.abs_()
        W_neg.abs_()

        return W_pos, W_neg, PL, NL

    @torch.no_grad()
    def _save_mask(self, masks, prefix, suffix):
        if self.iter % 1280 == 0:
            output_dir = os.path.join(self.output_dir, prefix)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            id_str = "iter" + str(self.iter) + "_gpu" + str(masks.device.index)

            for c in range(self.num_classes):
                if self.gt_classes_img_oh[0, c] < 0.5:
                    continue
                # if self.pred_class_img_logits[0, c] < self.tau:
                # continue

                mask = masks[0, c, ...].clone().detach().cpu().numpy()
                max_value = np.max(mask)
                if max_value > 0:
                    max_value = max_value * 0.1
                    mask = np.clip(mask, 0, max_value)
                    mask = mask / max_value * 255
                mask = mask.astype(np.uint8)
                im_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

                save_path = os.path.join(output_dir, id_str + "_c" + str(c) + "_" + suffix + ".png")
                cv2.imwrite(save_path, im_color)

            im = self.images.tensor[0, ...].clone().detach().cpu().numpy()
            channel_swap = (1, 2, 0)
            im = im.transpose(channel_swap)
            pixel_means = [103.939, 116.779, 123.68]
            im += pixel_means
            im = im.astype(np.uint8)
            save_path = os.path.join(output_dir, id_str + ".png")
            cv2.imwrite(save_path, im)


@ROI_HEADS_REGISTRY.register()
class OICRROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(OICRROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg, input_shape)
        self._init_mask_head(cfg, input_shape)
        self._init_keypoint_head(cfg, input_shape)

    def _init_box_head(self, cfg, input_shape):
        # fmt: off
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = WSDDNOutputLayers(cfg, self.box_head.output_shape)

        # OICR
        self.output_dir = cfg.OUTPUT_DIR
        self.iter = 0
        self.refine_K = cfg.WSL.REFINE_NUM
        self.refine_mist = cfg.WSL.REFINE_MIST
        self.refine_reg = cfg.WSL.REFINE_REG
        self.has_gam = cfg.WSL.HAS_GAM
        self.box_refinery = []
        for k in range(self.refine_K):
            box_refinery = OICROutputLayers(cfg, self.box_head.output_shape, k)
            self.add_module("box_refinery_{}".format(k), box_refinery)
            self.box_refinery.append(box_refinery)

        # GAM guided attention module
        if self.has_gam:
            self.box_gam = GAMOutputLayers(in_channels, self.num_classes)

        self.sampling_on = cfg.WSL.SAMPLING.SAMPLING_ON
        if self.sampling_on:
            self.proposal_matchers = []

            for k in range(self.refine_K):
                # Matcher to assign box proposals to gt boxes
                self.proposal_matchers.append(
                    Matcher(
                        cfg.WSL.SAMPLING.IOU_THRESHOLDS[k],
                        cfg.WSL.SAMPLING.IOU_LABELS[k],
                        allow_low_quality_matches=False,
                    )
                )

    def _init_mask_head(self, cfg, input_shape):
        # fmt: off
        self.mask_on           = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _init_keypoint_head(self, cfg, input_shape):
        # fmt: off
        self.keypoint_on  = cfg.MODEL.KEYPOINT_ON
        if not self.keypoint_on:
            return
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.keypoint_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.keypoint_head = build_keypoint_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        self.gt_classes_img, self.gt_classes_img_int, self.gt_classes_img_oh = get_image_level_gt(
            targets, self.num_classes
        )
        self.num_preds_per_image = [len(p) for p in proposals]

        # del images
        self.images = images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances, all_scores, all_boxes = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}, all_scores, all_boxes

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.in_features]

        # GAM guided attention module
        if self.has_gam:
            gam = [self.box_gam(feature) for feature in features]
            assert len(features) == 1
            features = [gam[0][0]]
            pred_class_gam_logits = gam[0][1]

        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])

        objectness_logits = torch.cat([x.objectness_logits + 1 for x in proposals], dim=0)
        # objectness_logits = torch.cat([x.objectness_logits for x in proposals], dim=0)
        # objectness_logits[objectness_logits > 1] = 0
        # objectness_logits[objectness_logits < -1] = 0
        # objectness_logits = objectness_logits + 1
        box_features = box_features * objectness_logits.view(-1, 1, 1, 1)

        torch.cuda.empty_cache()

        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features, proposals)

        # OICR
        predictions_K = []
        for k in range(self.refine_K):
            predictions_k = self.box_refinery[k](box_features)
            predictions_K.append(predictions_k)

        del box_features

        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            losses = self.box_predictor.losses(predictions, proposals, self.gt_classes_img_oh)

            # OICR
            self.pred_class_img_logits = (
                self.box_predictor.predict_probs_img(predictions, proposals).clone().detach()
            )

            prev_pred_scores = predictions[0].clone().detach()
            prev_pred_boxes = [p.proposal_boxes for p in proposals]
            for k in range(self.refine_K):
                suffix = "_r" + str(k)
                targets, target_weights = self.get_pgt(
                    prev_pred_boxes, prev_pred_scores, proposals, suffix
                )

                proposal_append_gt = self.proposal_append_gt
                self.proposal_append_gt = False
                if self.sampling_on:
                    proposals_k, matched_idxs = self.label_and_sample_proposals_wsl(
                        k, proposals, targets, ret_MI=True, suffix=suffix
                    )
                else:
                    proposals_k, matched_idxs = self.label_and_sample_proposals(
                        proposals, targets, ret_MI=True, suffix=suffix
                    )
                self.proposal_append_gt = proposal_append_gt

                # proposal_weights = torch.index_select(pgt_weight, 0, matched_idxs[0])
                proposal_weights = torch.cat(
                    [
                        torch.index_select(target_weight, 0, matched_idx)
                        for target_weight, matched_idx in zip(target_weights, matched_idxs)
                    ],
                    dim=0,
                )

                losses_k = self.box_refinery[k].losses(
                    predictions_K[k], proposals_k, proposal_weights
                )

                prev_pred_scores = self.box_refinery[k].predict_probs(predictions_K[k], proposals_k)
                prev_pred_boxes = self.box_refinery[k].predict_boxes(predictions_K[k], proposals_k)
                prev_pred_scores = [
                    prev_pred_score.clone().detach() for prev_pred_score in prev_pred_scores
                ]
                prev_pred_boxes = [
                    prev_pred_box.clone().detach() for prev_pred_box in prev_pred_boxes
                ]

                losses.update(losses_k)

            # GAM guided attention module
            if self.has_gam:
                outputs = GAMOutputs(pred_class_gam_logits, self.mean_loss, self.gt_classes_img_oh)
                losses.update(outputs.losses())

            self.iter = self.iter + 1
            return losses
        else:
            if self.refine_reg[-1]:
                pred_instances, _, all_scores, all_boxes = self.box_refinery[-1].inference(
                    predictions_K[-1], proposals
                )
            else:
                pred_instances, _, all_scores, all_boxes = self.box_refinery[-1].inference(
                    predictions_K, proposals
                )
            return pred_instances, all_scores, all_boxes

    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            return self.mask_head(mask_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            return self.mask_head(mask_features, instances)

    def _forward_keypoint(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            keypoint_features = self.keypoint_pooler(features, proposal_boxes)
            return self.keypoint_head(keypoint_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.keypoint_pooler(features, pred_boxes)
            return self.keypoint_head(keypoint_features, instances)

    @torch.no_grad()
    def get_pgt_mist(self, prev_pred_boxes, prev_pred_scores, proposals, suffix):
        assert isinstance(prev_pred_boxes, tuple) or isinstance(prev_pred_boxes, list)
        if isinstance(prev_pred_boxes[0], Boxes):
            num_preds = [len(prev_pred_box) for prev_pred_box in prev_pred_boxes]
            prev_pred_boxes = [
                prev_pred_box.tensor.unsqueeze(1).expand(num_pred, self.num_classes, 4)
                for num_pred, prev_pred_box in zip(num_preds, prev_pred_boxes)
            ]
        else:
            assert isinstance(prev_pred_boxes[0], torch.Tensor)
        prev_pred_boxes = [
            prev_pred_box.view(-1, self.num_classes, 4) for prev_pred_box in prev_pred_boxes
        ]

        if isinstance(prev_pred_scores, torch.Tensor):
            prev_pred_scores = prev_pred_scores.split(self.num_preds_per_image, dim=0)
        else:
            assert isinstance(prev_pred_scores, list)
            assert isinstance(prev_pred_scores[0], torch.Tensor)

        prev_pred_scores = [
            torch.index_select(prev_pred_score, 1, gt_int)
            for prev_pred_score, gt_int in zip(prev_pred_scores, self.gt_classes_img_int)
        ]
        prev_pred_boxes = [
            torch.index_select(prev_pred_box, 1, gt_int)
            for prev_pred_box, gt_int in zip(prev_pred_boxes, self.gt_classes_img_int)
        ]

        # get top k
        num_preds = [prev_pred_score.size(0) for prev_pred_score in prev_pred_scores]
        top_ks = [max(int(num_pred * 0.10), 1) for num_pred in num_preds]
        pgt_scores_idxs = [
            torch.topk(prev_pred_score, top_k, dim=0)
            for prev_pred_score, top_k in zip(prev_pred_scores, top_ks)
        ]
        pgt_scores = [item[0] for item in pgt_scores_idxs]
        pgt_idxs = [item[1] for item in pgt_scores_idxs]
        pgt_idxs = [
            torch.unsqueeze(pgt_idx, 2).expand(top_k, gt_int.numel(), 4)
            for pgt_idx, top_k, gt_int in zip(pgt_idxs, top_ks, self.gt_classes_img_int)
        ]
        pgt_boxes = [
            torch.gather(prev_pred_box, 0, pgt_idx)
            for prev_pred_box, pgt_idx in zip(prev_pred_boxes, pgt_idxs)
        ]
        pgt_classes = [
            torch.unsqueeze(gt_int, 0).expand(top_k, gt_int.numel())
            for gt_int, top_k in zip(self.gt_classes_img_int, top_ks)
        ]
        pgt_weights = [
            torch.index_select(pred_logits, 1, gt_int).expand(top_k, gt_int.numel())
            for pred_logits, gt_int, top_k in zip(
                self.pred_class_img_logits.split(1, dim=0), self.gt_classes_img_int, top_ks
            )
        ]

        # get large scores
        masks = [pgt_score.ge(0.05) for pgt_score in pgt_scores]
        # mask[0, :] = True
        masks = [
            torch.cat([torch.full_like(mask[0:1, :], True), mask[1:, :]], dim=0) for mask in masks
        ]
        pgt_scores = [
            torch.masked_select(pgt_score, mask) for pgt_score, mask in zip(pgt_scores, masks)
        ]
        pgt_boxes = [
            torch.masked_select(pgt_box, torch.unsqueeze(mask, 2).expand(top_k, gt_int.numel(), 4))
            for pgt_box, mask, top_k, gt_int in zip(
                pgt_boxes, masks, top_ks, self.gt_classes_img_int
            )
        ]
        pgt_classes = [
            torch.masked_select(pgt_class, mask) for pgt_class, mask in zip(pgt_classes, masks)
        ]
        pgt_weights = [
            torch.masked_select(pgt_weight, mask) for pgt_weight, mask in zip(pgt_weights, masks)
        ]

        pgt_scores = [pgt_score.reshape(-1) for pgt_score in pgt_scores]
        pgt_boxes = [pgt_box.reshape(-1, 4) for pgt_box in pgt_boxes]
        pgt_classes = [pgt_class.reshape(-1) for pgt_class in pgt_classes]
        pgt_weights = [pgt_weight.reshape(-1) for pgt_weight in pgt_weights]

        # NMS
        keeps = [
            batched_nms(pgt_box, pgt_score, pgt_class, 0.10)
            for pgt_box, pgt_score, pgt_class in zip(pgt_boxes, pgt_scores, pgt_classes)
        ]
        pgt_scores = [pgt_score[keep] for pgt_score, keep in zip(pgt_scores, keeps)]
        pgt_boxes = [pgt_box[keep] for pgt_box, keep in zip(pgt_boxes, keeps)]
        pgt_classes = [pgt_class[keep] for pgt_class, keep in zip(pgt_classes, keeps)]
        pgt_weights = [pgt_weight[keep] for pgt_weight, keep in zip(pgt_weights, keeps)]

        pgt_boxes = [Boxes(pgt_box) for pgt_box in pgt_boxes]

        targets = [
            Instances(proposals[i].image_size, gt_boxes=pgt_box, gt_classes=pgt_class)
            for i, (pgt_box, pgt_class) in enumerate(zip(pgt_boxes, pgt_classes))
        ]

        self._save_pgt(pgt_boxes, pgt_classes, pgt_scores, "pgt_mist", suffix)

        return targets, pgt_weights

    @torch.no_grad()
    def get_pgt(self, prev_pred_boxes, prev_pred_scores, proposals, suffix):
        if isinstance(prev_pred_scores, torch.Tensor):
            prev_pred_scores = prev_pred_scores.split(self.num_preds_per_image, dim=0)
        else:
            assert isinstance(prev_pred_scores, list)
            assert isinstance(prev_pred_scores[0], torch.Tensor)

        prev_pred_scores = [
            torch.index_select(prev_pred_score, 1, gt_int)
            for prev_pred_score, gt_int in zip(prev_pred_scores, self.gt_classes_img_int)
        ]
        pgt_scores_idxs = [
            torch.max(prev_pred_score, dim=0) for prev_pred_score in prev_pred_scores
        ]
        pgt_scores = [item[0] for item in pgt_scores_idxs]
        pgt_idxs = [item[1] for item in pgt_scores_idxs]

        assert isinstance(prev_pred_boxes, tuple) or isinstance(prev_pred_boxes, list)
        if isinstance(prev_pred_boxes[0], Boxes):
            pgt_boxes = [
                prev_pred_box[pgt_idx] for prev_pred_box, pgt_idx in zip(prev_pred_boxes, pgt_idxs)
            ]
        else:
            assert isinstance(prev_pred_boxes[0], torch.Tensor)
            prev_pred_boxes = [
                prev_pred_box.view(-1, self.num_classes, 4) for prev_pred_box in prev_pred_boxes
            ]
            prev_pred_boxes = [
                torch.index_select(prev_pred_box, 1, gt_int)
                for prev_pred_box, gt_int in zip(prev_pred_boxes, self.gt_classes_img_int)
            ]
            pgt_boxes = [
                torch.index_select(prev_pred_box, 0, pgt_idx)
                for prev_pred_box, pgt_idx in zip(prev_pred_boxes, pgt_idxs)
            ]
            pgt_boxes = [pgt_box.view(-1, 4) for pgt_box in pgt_boxes]
            diags = [
                torch.tensor(
                    [i * gt_split.numel() + i for i in range(gt_split.numel())],
                    dtype=torch.int64,
                    device=pgt_boxes[0].device,
                )
                for gt_split in self.gt_classes_img
            ]
            pgt_boxes = [
                torch.index_select(pgt_box, 0, diag) for pgt_box, diag in zip(pgt_boxes, diags)
            ]
            pgt_boxes = [Boxes(pgt_box) for pgt_box in pgt_boxes]

        pgt_classes = self.gt_classes_img_int
        pgt_weights = [
            torch.index_select(pred_logits, 1, pgt_class).reshape(-1)
            for pred_logits, pgt_class in zip(
                self.pred_class_img_logits.split(1, dim=0), pgt_classes
            )
        ]

        targets = [
            Instances(proposals[i].image_size, gt_boxes=pgt_box, gt_classes=pgt_class)
            for i, (pgt_box, pgt_class) in enumerate(zip(pgt_boxes, pgt_classes))
        ]

        self._save_pgt(pgt_boxes, pgt_classes, pgt_scores, "pgt", suffix)

        return targets, pgt_weights

    @torch.no_grad()
    def _save_pgt(self, pgt_boxes, pgt_classes, pgt_scores, prefix, suffix):
        if self.iter % 1280 > 0:
            return
        output_dir = os.path.join(self.output_dir, prefix)
        if self.iter == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        for b, (pgt_box, pgt_class, pgt_score) in enumerate(
            zip(pgt_boxes, pgt_classes, pgt_scores)
        ):
            im = self.images.tensor[b, ...].clone().detach().cpu().numpy()
            channel_swap = (1, 2, 0)
            im = im.transpose(channel_swap)
            pixel_means = [103.939, 116.779, 123.68]
            im += pixel_means
            im = im.astype(np.uint8)
            h, w = im.shape[:2]
            im_pgt = im.copy()

            id_str = "i" + str(self.iter) + "_g" + str(pgt_box.device.index) + "_b" + str(b)
            pgt_box = pgt_box.tensor.clone().detach().cpu().numpy()
            pgt_class = pgt_class.clone().detach().cpu().numpy()
            pgt_score = pgt_score.clone().detach().cpu().numpy()
            for i in range(pgt_box.shape[0]):
                c = pgt_class[i]
                s = pgt_score[i]
                x0, y0, x1, y1 = pgt_box[i, :]
                x0 = int(max(x0, 0))
                y0 = int(max(y0, 0))
                x1 = int(min(x1, w))
                y1 = int(min(y1, h))
                cv2.rectangle(im_pgt, (x0, y0), (x1, y1), (0, 0, 255), 4)
                (tw, th), bl = cv2.getTextSize(str(c), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                cv2.putText(
                    im_pgt, str(c), (x0, y0 + th), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2
                )
                (_, t_h), bl = cv2.getTextSize(str(s), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv2.putText(
                    im_pgt, str(s), (x0 + tw, y0 + th), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1
                )

            save_path = os.path.join(output_dir, id_str + "_" + suffix + ".png")
            cv2.imwrite(save_path, im_pgt)

    @torch.no_grad()
    def label_and_sample_proposals_wsl(
        self, k: int, proposals: List[Instances], targets: List[Instances], ret_MI=False, suffix=""
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_sample_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []
        matched_idxs_ = []

        num_fg_samples = []
        num_bg_samples = []
        num_ig_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matchers[k](match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_ig_samples.append((gt_classes == -1).sum().item())
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1] - num_ig_samples[-1])
            proposals_with_gt.append(proposals_per_image)

            matched_idxs_.append(matched_idxs[sampled_idxs])

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples" + suffix, np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples" + suffix, np.mean(num_bg_samples))
        storage.put_scalar("roi_head/num_ig_samples" + suffix, np.mean(num_ig_samples))

        if ret_MI:
            return proposals_with_gt, matched_idxs_
        return proposals_with_gt


@ROI_HEADS_REGISTRY.register()
class CascadeOICRROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(CascadeOICRROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg, input_shape)
        self._init_mask_head(cfg, input_shape)
        self._init_keypoint_head(cfg, input_shape)

    def _init_box_head(self, cfg, input_shape):
        # fmt: off
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = WSDDNOutputLayers(cfg, self.box_head.output_shape)

        # OICR
        self.cls_agnostic_bbox_reg = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.output_dir = cfg.OUTPUT_DIR
        self.iter = 0
        self.refine_K = cfg.WSL.REFINE_NUM
        self.refine_mist = cfg.WSL.REFINE_MIST
        self.refine_reg = cfg.WSL.REFINE_REG
        self.has_gam = cfg.WSL.HAS_GAM
        self.box_refinery = []
        for k in range(self.refine_K):
            box_refinery = OICROutputLayers(cfg, self.box_head.output_shape, k)
            self.add_module("box_refinery_{}".format(k), box_refinery)
            self.box_refinery.append(box_refinery)

        # GAM guided attention module
        if self.has_gam:
            self.box_gam = GAMOutputLayers(in_channels, self.num_classes)

        self.cascade_on = cfg.WSL.CASCADE_ON
        if self.cascade_on and False:
            self.box_head_refinery = []
            for k in range(1, self.refine_K):
                box_head_refinery = build_box_head(
                    cfg,
                    ShapeSpec(
                        channels=in_channels, height=pooler_resolution, width=pooler_resolution
                    ),
                )
                self.add_module("box_head_refinery_{}".format(k), box_head_refinery)
                self.box_head_refinery.append(box_head_refinery)

        self.sampling_on = cfg.WSL.SAMPLING.SAMPLING_ON
        if self.sampling_on:
            self.proposal_matchers = []

            for k in range(self.refine_K):
                # Matcher to assign box proposals to gt boxes
                self.proposal_matchers.append(
                    Matcher(
                        cfg.WSL.SAMPLING.IOU_THRESHOLDS[k],
                        cfg.WSL.SAMPLING.IOU_LABELS[k],
                        allow_low_quality_matches=False,
                    )
                )

    def _init_mask_head(self, cfg, input_shape):
        # fmt: off
        self.mask_on           = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _init_keypoint_head(self, cfg, input_shape):
        # fmt: off
        self.keypoint_on  = cfg.MODEL.KEYPOINT_ON
        if not self.keypoint_on:
            return
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.keypoint_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.keypoint_head = build_keypoint_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        self.gt_classes_img, self.gt_classes_img_int, self.gt_classes_img_oh = get_image_level_gt(
            targets, self.num_classes
        )

        # del images
        self.images = images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances, all_scores, all_boxes = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}, all_scores, all_boxes

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.in_features]
        image_sizes = [x.image_size for x in proposals]

        # GAM guided attention module
        if self.has_gam:
            gam = [self.box_gam(feature) for feature in features]
            assert len(features) == 1
            features = [gam[0][0]]
            pred_class_gam_logits = gam[0][1]

        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])

        objectness_logits = torch.cat([x.objectness_logits + 1 for x in proposals], dim=0)
        box_features = box_features * objectness_logits.view(-1, 1, 1, 1)

        # torch.cuda.empty_cache()

        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features, proposals)
        # del box_features

        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            losses = self.box_predictor.losses(predictions, proposals, self.gt_classes_img_oh)

            self.pred_class_img_logits = (
                self.box_predictor.predict_probs_img(predictions, proposals).clone().detach()
            )

            prev_pred_scores = predictions[0].detach()
            prev_pred_boxes = [p.proposal_boxes for p in proposals]
            for k in range(self.refine_K):
                suffix = "_r" + str(k)
                if k > 0 and self.cascade_on:
                    # proposals_topk = self._create_proposals_from_boxes(prev_pred_boxes, prev_pred_scores, image_sizes)
                    targets, target_weights = self.get_pgt_mist(
                        prev_pred_boxes, prev_pred_scores, proposals_k, suffix + "_cascade"
                    )
                    proposals_topk = []
                    for target, target_weight in zip(targets, target_weights):
                        prop = Instances(target.image_size)
                        prop.objectness_logits = target_weight[target.gt_boxes.nonempty()]
                        prop.proposal_boxes = target.gt_boxes[target.gt_boxes.nonempty()]
                        proposals_topk.append(prop)
                    proposals_k = [
                        Boxes.cat(p_k, p_topk) for p_k, p_topk in zip(proposals_k, proposals_topk)
                    ]
                else:
                    proposals_k = proposals
                targets, target_weights = self.get_pgt(
                    prev_pred_boxes, prev_pred_scores, proposals_k, suffix
                )

                proposal_append_gt = self.proposal_append_gt
                self.proposal_append_gt = False
                if self.sampling_on:
                    proposals_k, matched_idxs = self.label_and_sample_proposals_wsl(
                        k, proposals_k, targets, ret_MI=True, suffix=suffix
                    )
                else:
                    proposals_k, matched_idxs = self.label_and_sample_proposals(
                        proposals_k, targets, ret_MI=True, suffix=suffix
                    )
                self.proposal_append_gt = proposal_append_gt

                proposal_weights = torch.cat(
                    [
                        torch.index_select(target_weight, 0, matched_idx)
                        for target_weight, matched_idx in zip(target_weights, matched_idxs)
                    ],
                    dim=0,
                )

                if k > 0 and self.cascade_on:
                    box_features_topk = self.box_pooler(
                        features, [x.proposal_boxes for x in proposals_topk]
                    )

                    objectness_logits = torch.cat(
                        [x.objectness_logits + 1 for x in proposals_topk], dim=0
                    )
                    box_features_topk = box_features * objectness_logits.view(-1, 1, 1, 1)

                    box_features_topk = self.box_head(box_features_topk)
                    # box_features = self.box_head_refinery[k](box_features)

                predictions_k = self.box_refinery[k](box_features)

                losses_k = self.box_refinery[k].losses(predictions_k, proposals_k, proposal_weights)

                prev_pred_scores = self.box_refinery[k].predict_probs(predictions_k, proposals_k)
                prev_pred_boxes = self.box_refinery[k].predict_boxes(predictions_k, proposals_k)
                prev_pred_scores = [
                    prev_pred_score.detach() for prev_pred_score in prev_pred_scores
                ]
                prev_pred_boxes = [prev_pred_box.detach() for prev_pred_box in prev_pred_boxes]

                losses.update(losses_k)

            # GAM guided attention module
            if self.has_gam:
                outputs = GAMOutputs(pred_class_gam_logits, self.mean_loss, self.gt_classes_img_oh)
                losses.update(outputs.losses())

            self.iter = self.iter + 1
            return losses
        else:
            predictions_K = []
            prev_pred_scores = predictions[0].detach()
            prev_pred_boxes = [p.proposal_boxes for p in proposals]
            for k in range(self.refine_K):
                if k > 0 and self.cascade_on:
                    proposals_topk = self._create_proposals_from_boxes(
                        prev_pred_boxes, prev_pred_scores, image_sizes
                    )
                    proposals_k = proposals_topk
                else:
                    proposals_k = proposals

                if k > 0 and self.cascade_on:
                    box_features = self.box_pooler(
                        features, [x.proposal_boxes for x in proposals_k]
                    )

                    objectness_logits = torch.cat(
                        [x.objectness_logits + 1 for x in proposals_k], dim=0
                    )
                    box_features = box_features * objectness_logits.view(-1, 1, 1, 1)

                    box_features = self.box_head(box_features)
                    # box_features = self.box_head_refinery[k](box_features)

                predictions_k = self.box_refinery[k](box_features)
                predictions_K.append(predictions_k)

                prev_pred_scores = self.box_refinery[k].predict_probs(predictions_k, proposals_k)
                prev_pred_boxes = self.box_refinery[k].predict_boxes(predictions_k, proposals_k)
                prev_pred_scores = [
                    prev_pred_score.detach() for prev_pred_score in prev_pred_scores
                ]
                prev_pred_boxes = [prev_pred_box.detach() for prev_pred_box in prev_pred_boxes]

            if self.refine_reg[-1]:
                pred_instances, _, all_scores, all_boxes = self.box_refinery[-1].inference(
                    predictions_K[-1], proposals
                )
            else:
                pred_instances, _, all_scores, all_boxes = self.box_refinery[-1].inference(
                    predictions_K, proposals
                )
            return pred_instances, all_scores, all_boxes

    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            return self.mask_head(mask_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            return self.mask_head(mask_features, instances)

    def _forward_keypoint(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            keypoint_features = self.keypoint_pooler(features, proposal_boxes)
            return self.keypoint_head(keypoint_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.keypoint_pooler(features, pred_boxes)
            return self.keypoint_head(keypoint_features, instances)

    @torch.no_grad()
    def get_pgt_mist(self, prev_pred_boxes, prev_pred_scores, proposals, suffix, top_pro=0.10):
        assert isinstance(prev_pred_boxes, tuple) or isinstance(prev_pred_boxes, list)
        if isinstance(prev_pred_boxes[0], Boxes):
            num_preds = [len(prev_pred_box) for prev_pred_box in prev_pred_boxes]
            prev_pred_boxes = [
                prev_pred_box.tensor.unsqueeze(1).expand(num_pred, self.num_classes, 4)
                for num_pred, prev_pred_box in zip(num_preds, prev_pred_boxes)
            ]
        else:
            assert isinstance(prev_pred_boxes[0], torch.Tensor)
            if self.cls_agnostic_bbox_reg:
                num_preds = [prev_pred_box.size(0) for prev_pred_box in prev_pred_boxes]
                prev_pred_boxes = [
                    prev_pred_box.unsqueeze(1).expand(num_pred, self.num_classes, 4)
                    for num_pred, prev_pred_box in zip(num_preds, prev_pred_boxes)
                ]
        prev_pred_boxes = [
            prev_pred_box.view(-1, self.num_classes, 4) for prev_pred_box in prev_pred_boxes
        ]

        if isinstance(prev_pred_scores, torch.Tensor):
            prev_pred_scores = prev_pred_scores.split(self.num_preds_per_image, dim=0)
        else:
            assert isinstance(prev_pred_scores, list)
            assert isinstance(prev_pred_scores[0], torch.Tensor)

        prev_pred_scores = [
            torch.index_select(prev_pred_score, 1, gt_int)
            for prev_pred_score, gt_int in zip(prev_pred_scores, self.gt_classes_img_int)
        ]
        prev_pred_boxes = [
            torch.index_select(prev_pred_box, 1, gt_int)
            for prev_pred_box, gt_int in zip(prev_pred_boxes, self.gt_classes_img_int)
        ]

        # get top k
        num_preds = [prev_pred_score.size(0) for prev_pred_score in prev_pred_scores]
        top_ks = [max(int(num_pred * top_pro), 1) for num_pred in num_preds]
        pgt_scores_idxs = [
            torch.topk(prev_pred_score, top_k, dim=0)
            for prev_pred_score, top_k in zip(prev_pred_scores, top_ks)
        ]
        pgt_scores = [item[0] for item in pgt_scores_idxs]
        pgt_idxs = [item[1] for item in pgt_scores_idxs]
        pgt_idxs = [
            torch.unsqueeze(pgt_idx, 2).expand(top_k, gt_int.numel(), 4)
            for pgt_idx, top_k, gt_int in zip(pgt_idxs, top_ks, self.gt_classes_img_int)
        ]
        pgt_boxes = [
            torch.gather(prev_pred_box, 0, pgt_idx)
            for prev_pred_box, pgt_idx in zip(prev_pred_boxes, pgt_idxs)
        ]
        pgt_classes = [
            torch.unsqueeze(gt_int, 0).expand(top_k, gt_int.numel())
            for gt_int, top_k in zip(self.gt_classes_img_int, top_ks)
        ]
        pgt_weights = [
            torch.index_select(pred_logits, 1, gt_int).expand(top_k, gt_int.numel())
            for pred_logits, gt_int, top_k in zip(
                self.pred_class_img_logits.split(1, dim=0), self.gt_classes_img_int, top_ks
            )
        ]

        # get large scores
        masks = [pgt_score.ge(0.05) for pgt_score in pgt_scores]
        # mask[0, :] = True
        masks = [
            torch.cat([torch.full_like(mask[0:1, :], True), mask[1:, :]], dim=0) for mask in masks
        ]
        pgt_scores = [
            torch.masked_select(pgt_score, mask) for pgt_score, mask in zip(pgt_scores, masks)
        ]
        pgt_boxes = [
            torch.masked_select(pgt_box, torch.unsqueeze(mask, 2).expand(top_k, gt_int.numel(), 4))
            for pgt_box, mask, top_k, gt_int in zip(
                pgt_boxes, masks, top_ks, self.gt_classes_img_int
            )
        ]
        pgt_classes = [
            torch.masked_select(pgt_class, mask) for pgt_class, mask in zip(pgt_classes, masks)
        ]
        pgt_weights = [
            torch.masked_select(pgt_weight, mask) for pgt_weight, mask in zip(pgt_weights, masks)
        ]

        pgt_scores = [pgt_score.reshape(-1) for pgt_score in pgt_scores]
        pgt_boxes = [pgt_box.reshape(-1, 4) for pgt_box in pgt_boxes]
        pgt_classes = [pgt_class.reshape(-1) for pgt_class in pgt_classes]
        pgt_weights = [pgt_weight.reshape(-1) for pgt_weight in pgt_weights]

        # NMS
        keeps = [
            batched_nms(pgt_box, pgt_score, pgt_class, 0.10)
            for pgt_box, pgt_score, pgt_class in zip(pgt_boxes, pgt_scores, pgt_classes)
        ]
        pgt_scores = [pgt_score[keep] for pgt_score, keep in zip(pgt_scores, keeps)]
        pgt_boxes = [pgt_box[keep] for pgt_box, keep in zip(pgt_boxes, keeps)]
        pgt_classes = [pgt_class[keep] for pgt_class, keep in zip(pgt_classes, keeps)]
        pgt_weights = [pgt_weight[keep] for pgt_weight, keep in zip(pgt_weights, keeps)]

        pgt_boxes = [Boxes(pgt_box) for pgt_box in pgt_boxes]

        targets = [
            Instances(proposals[i].image_size, gt_boxes=pgt_box, gt_classes=pgt_class)
            for i, (pgt_box, pgt_class) in enumerate(zip(pgt_boxes, pgt_classes))
        ]

        self._save_pgt(pgt_boxes, pgt_classes, pgt_scores, "pgt_mist", suffix)

        return targets, pgt_weights

    @torch.no_grad()
    def get_pgt(self, prev_pred_boxes, prev_pred_scores, proposals, suffix):
        if isinstance(prev_pred_scores, torch.Tensor):
            prev_pred_scores = prev_pred_scores.split(self.num_preds_per_image, dim=0)
        else:
            assert isinstance(prev_pred_scores, list)
            assert isinstance(prev_pred_scores[0], torch.Tensor)

        prev_pred_scores = [
            torch.index_select(prev_pred_score, 1, gt_int)
            for prev_pred_score, gt_int in zip(prev_pred_scores, self.gt_classes_img_int)
        ]
        pgt_scores_idxs = [
            torch.max(prev_pred_score, dim=0) for prev_pred_score in prev_pred_scores
        ]
        pgt_scores = [item[0] for item in pgt_scores_idxs]
        pgt_idxs = [item[1] for item in pgt_scores_idxs]

        assert isinstance(prev_pred_boxes, tuple) or isinstance(prev_pred_boxes, list)
        if isinstance(prev_pred_boxes[0], Boxes):
            pgt_boxes = [
                prev_pred_box[pgt_idx] for prev_pred_box, pgt_idx in zip(prev_pred_boxes, pgt_idxs)
            ]
        else:
            assert isinstance(prev_pred_boxes[0], torch.Tensor)
            if self.cls_agnostic_bbox_reg:
                num_preds = [prev_pred_box.size(0) for prev_pred_box in prev_pred_boxes]
                prev_pred_boxes = [
                    prev_pred_box.unsqueeze(1).expand(num_pred, self.num_classes, 4)
                    for num_pred, prev_pred_box in zip(num_preds, prev_pred_boxes)
                ]
            prev_pred_boxes = [
                prev_pred_box.view(-1, self.num_classes, 4) for prev_pred_box in prev_pred_boxes
            ]
            prev_pred_boxes = [
                torch.index_select(prev_pred_box, 1, gt_int)
                for prev_pred_box, gt_int in zip(prev_pred_boxes, self.gt_classes_img_int)
            ]
            pgt_boxes = [
                torch.index_select(prev_pred_box, 0, pgt_idx)
                for prev_pred_box, pgt_idx in zip(prev_pred_boxes, pgt_idxs)
            ]
            pgt_boxes = [pgt_box.view(-1, 4) for pgt_box in pgt_boxes]
            diags = [
                torch.tensor(
                    [i * gt_split.numel() + i for i in range(gt_split.numel())],
                    dtype=torch.int64,
                    device=pgt_boxes[0].device,
                )
                for gt_split in self.gt_classes_img
            ]
            pgt_boxes = [
                torch.index_select(pgt_box, 0, diag) for pgt_box, diag in zip(pgt_boxes, diags)
            ]
            pgt_boxes = [Boxes(pgt_box) for pgt_box in pgt_boxes]

        pgt_classes = self.gt_classes_img_int
        pgt_weights = [
            torch.index_select(pred_logits, 1, pgt_class).reshape(-1)
            for pred_logits, pgt_class in zip(
                self.pred_class_img_logits.split(1, dim=0), pgt_classes
            )
        ]

        targets = [
            Instances(proposals[i].image_size, gt_boxes=pgt_box, gt_classes=pgt_class)
            for i, (pgt_box, pgt_class) in enumerate(zip(pgt_boxes, pgt_classes))
        ]

        self._save_pgt(pgt_boxes, pgt_classes, pgt_scores, "pgt", suffix)

        return targets, pgt_weights

    @torch.no_grad()
    def _save_pgt(self, pgt_boxes, pgt_classes, pgt_scores, prefix, suffix):
        if self.iter % 1280 > 0:
            return
        output_dir = os.path.join(self.output_dir, prefix)
        if self.iter == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        for b, (pgt_box, pgt_class, pgt_score) in enumerate(
            zip(pgt_boxes, pgt_classes, pgt_scores)
        ):
            im = self.images.tensor[b, ...].clone().detach().cpu().numpy()
            channel_swap = (1, 2, 0)
            im = im.transpose(channel_swap)
            pixel_means = [103.939, 116.779, 123.68]
            im += pixel_means
            im = im.astype(np.uint8)
            h, w = im.shape[:2]
            im_pgt = im.copy()

            id_str = "i" + str(self.iter) + "_g" + str(pgt_box.device.index) + "_b" + str(b)
            pgt_box = pgt_box.tensor.clone().detach().cpu().numpy()
            pgt_class = pgt_class.clone().detach().cpu().numpy()
            pgt_score = pgt_score.clone().detach().cpu().numpy()
            for i in range(pgt_box.shape[0]):
                c = pgt_class[i]
                s = pgt_score[i]
                x0, y0, x1, y1 = pgt_box[i, :]
                x0 = int(max(x0, 0))
                y0 = int(max(y0, 0))
                x1 = int(min(x1, w))
                y1 = int(min(y1, h))
                cv2.rectangle(im_pgt, (x0, y0), (x1, y1), (0, 0, 255), 4)
                (tw, th), bl = cv2.getTextSize(str(c), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                cv2.putText(
                    im_pgt, str(c), (x0, y0 + th), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2
                )
                (_, t_h), bl = cv2.getTextSize(str(s), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv2.putText(
                    im_pgt, str(s), (x0 + tw, y0 + th), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1
                )

            save_path = os.path.join(output_dir, id_str + "_" + suffix + ".png")
            cv2.imwrite(save_path, im_pgt)

    @torch.no_grad()
    def label_and_sample_proposals_wsl(
        self, k: int, proposals: List[Instances], targets: List[Instances], ret_MI=False, suffix=""
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_sample_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []
        matched_idxs_ = []

        num_fg_samples = []
        num_bg_samples = []
        num_ig_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matchers[k](match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_ig_samples.append((gt_classes == -1).sum().item())
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1] - num_ig_samples[-1])
            proposals_with_gt.append(proposals_per_image)

            matched_idxs_.append(matched_idxs[sampled_idxs])

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples" + suffix, np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples" + suffix, np.mean(num_bg_samples))
        storage.put_scalar("roi_head/num_ig_samples" + suffix, np.mean(num_ig_samples))

        if ret_MI:
            return proposals_with_gt, matched_idxs_
        return proposals_with_gt

    def _create_proposals_from_boxes(self, boxes, scores, image_sizes):
        """
        Args:
            boxes (list[Tensor]): per-image predicted boxes, each of shape Ri x 4
            image_sizes (list[tuple]): list of image shapes in (h, w)

        Returns:
            list[Instances]: per-image proposals with the given boxes.
        """
        topk_scores_idxs = [torch.topk(score.view(-1), 512, dim=0) for score in scores]
        topk_scores = [item[0] for item in topk_scores_idxs]
        topk_idxs = [item[1] for item in topk_scores_idxs]

        topk_boxes = [
            Boxes(torch.index_select(b.view(-1, 4), 0, topk_idx))
            for b, topk_idx in zip(boxes, topk_idxs)
        ]

        proposals = []
        for boxes_per_image, scores_per_image, image_size in zip(
            topk_boxes, topk_scores, image_sizes
        ):
            boxes_per_image.clip(image_size)
            # if self.training:
            # # do not filter empty boxes at inference time,
            # # because the scores from each stage need to be aligned and added later
            # boxes_per_image = boxes_per_image[boxes_per_image.nonempty()]
            scores_per_image = scores_per_image[boxes_per_image.nonempty()]
            boxes_per_image = boxes_per_image[boxes_per_image.nonempty()]

            prop = Instances(image_size)
            prop.proposal_boxes = boxes_per_image
            prop.objectness_logits = scores_per_image
            proposals.append(prop)
        return proposals


@ROI_HEADS_REGISTRY.register()
class XROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(XROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg, input_shape)
        self._init_mask_head(cfg, input_shape)
        self._init_keypoint_head(cfg, input_shape)

    def _init_box_head(self, cfg, input_shape):
        # fmt: off
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.mrrp_on = cfg.MODEL.MRRP.MRRP_ON
        if self.mrrp_on:
            self.mrrp_num_branch = cfg.MODEL.MRRP.NUM_BRANCH
            self.mrrp_fast = cfg.MODEL.MRRP.TEST_BRANCH_IDX != -1
            # if self.mrrp_on and False:
            pooler_scales = tuple(
                1.0 / input_shape[k].stride for k in self.in_features * self.mrrp_num_branch
            )
            self.box_pooler = MRRPROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
        else:
            self.box_pooler = ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = WSDDNOutputLayers(cfg, self.box_head.output_shape)

        self.output_dir = cfg.OUTPUT_DIR
        self.iter = 0
        self.vis_test = cfg.WSL.VIS_TEST
        self.iter_test = 0
        self.epoch_test = 0
        self.refine_K = cfg.WSL.REFINE_NUM
        self.refine_mist = cfg.WSL.REFINE_MIST
        self.refine_reg = cfg.WSL.REFINE_REG
        self.has_gam = cfg.WSL.HAS_GAM
        self.box_refinery = []
        for k in range(self.refine_K):
            box_refinery = OICROutputLayers(cfg, self.box_head.output_shape, k)
            self.add_module("box_refinery_{}".format(k), box_refinery)
            self.box_refinery.append(box_refinery)

        # GAM guided attention module
        if self.has_gam:
            self.box_gam = GAMOutputLayers(in_channels, self.num_classes)

        self.rpn_on = cfg.WSL.RPN.RPN_ON
        if self.rpn_on:
            self.proposal_generator = build_proposal_generator(cfg, input_shape)
            self.rpn_warmup = cfg.WSL.RPN.WARMUP
            self.rpn_warmup_iters = cfg.WSL.RPN.WARMUP_ITERS

        self.sampling_on = cfg.WSL.SAMPLING.SAMPLING_ON
        if self.sampling_on:
            self.proposal_matchers = []
            self.batch_size_per_images = cfg.WSL.SAMPLING.BATCH_SIZE_PER_IMAGE
            self.positive_sample_fractions = cfg.WSL.SAMPLING.POSITIVE_FRACTION

            for k in range(self.refine_K):
                # Matcher to assign box proposals to gt boxes
                self.proposal_matchers.append(
                    Matcher(
                        cfg.WSL.SAMPLING.IOU_THRESHOLDS[k],
                        cfg.WSL.SAMPLING.IOU_LABELS[k],
                        allow_low_quality_matches=False,
                    )
                )

    def _init_mask_head(self, cfg, input_shape):
        # fmt: off
        self.mask_on           = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _init_keypoint_head(self, cfg, input_shape):
        # fmt: off
        self.keypoint_on  = cfg.MODEL.KEYPOINT_ON
        if not self.keypoint_on:
            return
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.keypoint_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.keypoint_head = build_keypoint_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        if self.mrrp_on and False:
            # Use 1 branch if using mrrp_fast during inference.
            num_branch = self.mrrp_num_branch if self.training or not self.mrrp_fast else 1
            # Duplicate images for all branches in MRRPNet.
            all_images = ImageList(
                torch.cat([images.tensor] * num_branch), images.image_sizes * num_branch
            )
            # Duplicate targets for all branches in MRRPNet.
            all_targets = targets * num_branch if targets is not None else None

            images = all_images
            targets = all_targets

        self.gt_classes_img, self.gt_classes_img_int, self.gt_classes_img_oh = get_image_level_gt(
            targets, self.num_classes
        )

        # del images
        self.images = images
        if self.training:
            if proposals:
                proposals = self.label_and_sample_proposals(proposals, targets)

        if self.rpn_on:
            proposals_rpn = self.proposal_generator(images, features, targets)
            for i, p in enumerate(proposals_rpn):
                if len(p) == 0:
                    device = p.objectness_logits.device
                    image_size = p.image_size

                    b_new = Boxes(
                        torch.tensor(
                            [[0.0, 0.0, image_size[1] - 1, image_size[0] - 1]],
                            device=device,
                            dtype=torch.float,
                        )
                    )
                    s_new = torch.tensor([1.0], device=device, dtype=torch.float)

                    p = Instances(image_size)
                    p.proposal_boxes = b_new
                    p.objectness_logits = s_new
                    proposals_rpn[i] = p
            if self.training:
                proposals_rpn = self.label_and_sample_proposals(
                    proposals_rpn, targets, suffix="_rpn"
                )

            if self.training and self.rpn_warmup and self.iter < self.rpn_warmup_iters:
                perms = [torch.randperm(len(p)) for p in proposals]
                perms_rpn = [torch.randperm(len(p_rpn)) for p_rpn in proposals_rpn]
                if self.iter % 1280 == 0:
                    print(
                        [
                            [
                                int(len(p) * (1 - self.iter / self.rpn_warmup_iters)),
                                int(len(p_rpn) * self.iter / self.rpn_warmup_iters),
                            ]
                            for p, perm, p_rpn, perm_rpn in zip(
                                proposals, perms, proposals_rpn, perms_rpn
                            )
                        ]
                    )
                # proposals_mix = [torch.cat([p[perm[:int(len(p) * (1-self.iter / self.rpn_warmup_iters))],...], p_rpn[perm_rpn[:int(len(p_rpn)*self.iter/self.rpn_warmup_iters)],...]], dim=0) for p,perm,p_rpn,perm_rpn in zip(proposals, perms,proposals_rpn, perms_rpn)]
                proposals_mix = [
                    Instances.cat(
                        [
                            p[perm[: int(len(p) * (1 - self.iter / self.rpn_warmup_iters))], ...],
                            p_rpn[: int(len(p_rpn) * self.iter / self.rpn_warmup_iters), ...],
                        ]
                    )
                    for p, perm, p_rpn, perm_rpn in zip(proposals, perms, proposals_rpn, perms_rpn)
                ]
                proposals = proposals_mix
            else:
                proposals = proposals_rpn

        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances, all_scores, all_boxes = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}, all_scores, all_boxes

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.in_features]

        # GAM guided attention module
        if self.has_gam:
            gam = [self.box_gam(feature) for feature in features]
            assert len(features) == 1
            features = [gam[0][0]]
            pred_class_gam_logits = gam[0][1]

        if self.mrrp_on:
            features = [torch.chunk(f, self.mrrp_num_branch) for f in features]
            features = [ff for f in features for ff in f]

        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])

        objectness_logits = torch.cat([x.objectness_logits + 1 for x in proposals], dim=0)
        box_features = box_features * torch.sigmoid(objectness_logits.view(-1, 1, 1, 1))
        if self.iter % 1280 == 0 and self.training:
            print(objectness_logits.max(), objectness_logits.min(), objectness_logits.mean())

        # torch.cuda.empty_cache()

        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features, proposals)
        # del box_features

        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            losses = self.box_predictor.losses(predictions, proposals, self.gt_classes_img_oh)

            self.pred_class_img_logits = (
                self.box_predictor.predict_probs_img(predictions, proposals).clone().detach()
            )

            prev_pred_scores = predictions[0].detach()
            prev_pred_boxes = [p.proposal_boxes for p in proposals]
            for k in range(self.refine_K):
                suffix = "_r" + str(k)
                targets, target_weights = self.get_pgt(
                    prev_pred_boxes, prev_pred_scores, proposals, suffix
                )

                proposal_append_gt = self.proposal_append_gt
                self.proposal_append_gt = False
                if self.sampling_on:
                    proposals_k, matched_idxs = self.label_and_sample_proposals_wsl(
                        k, proposals, targets, ret_MI=True, suffix=suffix
                    )
                else:
                    proposals_k, matched_idxs = self.label_and_sample_proposals(
                        proposals, targets, ret_MI=True, suffix=suffix
                    )
                self.proposal_append_gt = proposal_append_gt

                proposal_weights = torch.cat(
                    [
                        torch.index_select(target_weight, 0, matched_idx)
                        for target_weight, matched_idx in zip(target_weights, matched_idxs)
                    ],
                    dim=0,
                )

                predictions_k = self.box_refinery[k](box_features)

                losses_k = self.box_refinery[k].losses(predictions_k, proposals_k, proposal_weights)

                prev_pred_scores = self.box_refinery[k].predict_probs(predictions_k, proposals_k)
                prev_pred_boxes = self.box_refinery[k].predict_boxes(predictions_k, proposals_k)
                prev_pred_scores = [
                    prev_pred_score.detach() for prev_pred_score in prev_pred_scores
                ]
                prev_pred_boxes = [prev_pred_box.detach() for prev_pred_box in prev_pred_boxes]

                losses.update(losses_k)

            if self.rpn_on:
                targets, target_weights = self.get_pgt(
                    prev_pred_boxes, prev_pred_scores, proposals, "_rpn"
                )
                # targets, target_weights = self.get_pgt_mist(prev_pred_boxes, prev_pred_scores, proposals, '_rpn')
                losses_rpn = self.proposal_generator.get_losses(targets)
                losses.update(losses_rpn)

            # GAM guided attention module
            if self.has_gam:
                outputs = GAMOutputs(pred_class_gam_logits, self.mean_loss, self.gt_classes_img_oh)
                losses.update(outputs.losses())

            self.iter = self.iter + 1
            if self.iter_test > 0:
                self.epoch_test = self.epoch_test + 1
            self.iter_test = 0
            return losses
        else:
            if self.refine_reg[-1]:
                predictions_k = self.box_refinery[-1](box_features)
                if self.vis_test:
                    self._save_test(predictions_k, proposals)
                pred_instances, _, all_scores, all_boxes = self.box_refinery[-1].inference(
                    predictions_k, proposals
                )
            else:
                predictions_K = []
                for k in range(self.refine_K):
                    predictions_k = self.box_refinery[k](box_features)
                    predictions_K.append(predictions_k)
                pred_instances, _, all_scores, all_boxes = self.box_refinery[-1].inference(
                    predictions_K, proposals
                )
            self.iter_test = self.iter_test + 1
            return pred_instances, all_scores, all_boxes

    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            return self.mask_head(mask_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            return self.mask_head(mask_features, instances)

    def _forward_keypoint(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            keypoint_features = self.keypoint_pooler(features, proposal_boxes)
            return self.keypoint_head(keypoint_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.keypoint_pooler(features, pred_boxes)
            return self.keypoint_head(keypoint_features, instances)

    @torch.no_grad()
    def get_pgt_mist(self, prev_pred_boxes, prev_pred_scores, proposals, suffix, top_pro=0.10):
        assert isinstance(prev_pred_boxes, tuple) or isinstance(prev_pred_boxes, list)
        if isinstance(prev_pred_boxes[0], Boxes):
            num_preds = [len(prev_pred_box) for prev_pred_box in prev_pred_boxes]
            prev_pred_boxes = [
                prev_pred_box.tensor.unsqueeze(1).expand(num_pred, self.num_classes, 4)
                for num_pred, prev_pred_box in zip(num_preds, prev_pred_boxes)
            ]
        else:
            assert isinstance(prev_pred_boxes[0], torch.Tensor)
            if self.cls_agnostic_bbox_reg:
                num_preds = [prev_pred_box.size(0) for prev_pred_box in prev_pred_boxes]
                prev_pred_boxes = [
                    prev_pred_box.unsqueeze(1).expand(num_pred, self.num_classes, 4)
                    for num_pred, prev_pred_box in zip(num_preds, prev_pred_boxes)
                ]
        prev_pred_boxes = [
            prev_pred_box.view(-1, self.num_classes, 4) for prev_pred_box in prev_pred_boxes
        ]

        if isinstance(prev_pred_scores, torch.Tensor):
            num_preds_per_image = [len(p) for p in proposals]
            prev_pred_scores = prev_pred_scores.split(num_preds_per_image, dim=0)
        else:
            assert isinstance(prev_pred_scores, list)
            assert isinstance(prev_pred_scores[0], torch.Tensor)

        prev_pred_scores = [
            torch.index_select(prev_pred_score, 1, gt_int)
            for prev_pred_score, gt_int in zip(prev_pred_scores, self.gt_classes_img_int)
        ]
        prev_pred_boxes = [
            torch.index_select(prev_pred_box, 1, gt_int)
            for prev_pred_box, gt_int in zip(prev_pred_boxes, self.gt_classes_img_int)
        ]

        # get top k
        num_preds = [prev_pred_score.size(0) for prev_pred_score in prev_pred_scores]
        top_ks = [max(int(num_pred * top_pro), 1) for num_pred in num_preds]
        pgt_scores_idxs = [
            torch.topk(prev_pred_score, top_k, dim=0)
            for prev_pred_score, top_k in zip(prev_pred_scores, top_ks)
        ]
        pgt_scores = [item[0] for item in pgt_scores_idxs]
        pgt_idxs = [item[1] for item in pgt_scores_idxs]
        pgt_idxs = [
            torch.unsqueeze(pgt_idx, 2).expand(top_k, gt_int.numel(), 4)
            for pgt_idx, top_k, gt_int in zip(pgt_idxs, top_ks, self.gt_classes_img_int)
        ]
        pgt_boxes = [
            torch.gather(prev_pred_box, 0, pgt_idx)
            for prev_pred_box, pgt_idx in zip(prev_pred_boxes, pgt_idxs)
        ]
        pgt_classes = [
            torch.unsqueeze(gt_int, 0).expand(top_k, gt_int.numel())
            for gt_int, top_k in zip(self.gt_classes_img_int, top_ks)
        ]
        pgt_weights = [
            torch.index_select(pred_logits, 1, gt_int).expand(top_k, gt_int.numel())
            for pred_logits, gt_int, top_k in zip(
                self.pred_class_img_logits.split(1, dim=0), self.gt_classes_img_int, top_ks
            )
        ]

        # get large scores
        masks = [pgt_score.ge(0.05) for pgt_score in pgt_scores]
        masks = [
            torch.cat([torch.full_like(mask[0:1, :], True), mask[1:, :]], dim=0) for mask in masks
        ]
        pgt_scores = [
            torch.masked_select(pgt_score, mask) for pgt_score, mask in zip(pgt_scores, masks)
        ]
        pgt_boxes = [
            torch.masked_select(pgt_box, torch.unsqueeze(mask, 2).expand(top_k, gt_int.numel(), 4))
            for pgt_box, mask, top_k, gt_int in zip(
                pgt_boxes, masks, top_ks, self.gt_classes_img_int
            )
        ]
        pgt_classes = [
            torch.masked_select(pgt_class, mask) for pgt_class, mask in zip(pgt_classes, masks)
        ]
        pgt_weights = [
            torch.masked_select(pgt_weight, mask) for pgt_weight, mask in zip(pgt_weights, masks)
        ]

        pgt_scores = [pgt_score.reshape(-1) for pgt_score in pgt_scores]
        pgt_boxes = [pgt_box.reshape(-1, 4) for pgt_box in pgt_boxes]
        pgt_classes = [pgt_class.reshape(-1) for pgt_class in pgt_classes]
        pgt_weights = [pgt_weight.reshape(-1) for pgt_weight in pgt_weights]

        # NMS
        keeps = [
            batched_nms(pgt_box, pgt_score, pgt_class, 0.10)
            for pgt_box, pgt_score, pgt_class in zip(pgt_boxes, pgt_scores, pgt_classes)
        ]
        pgt_scores = [pgt_score[keep] for pgt_score, keep in zip(pgt_scores, keeps)]
        pgt_boxes = [pgt_box[keep] for pgt_box, keep in zip(pgt_boxes, keeps)]
        pgt_classes = [pgt_class[keep] for pgt_class, keep in zip(pgt_classes, keeps)]
        pgt_weights = [pgt_weight[keep] for pgt_weight, keep in zip(pgt_weights, keeps)]

        pgt_boxes = [Boxes(pgt_box) for pgt_box in pgt_boxes]

        targets = [
            Instances(proposals[i].image_size, gt_boxes=pgt_box, gt_classes=pgt_class)
            for i, (pgt_box, pgt_class) in enumerate(zip(pgt_boxes, pgt_classes))
        ]

        self._save_pgt(pgt_boxes, pgt_classes, pgt_scores, "pgt_mist", suffix)

        return targets, pgt_weights

    @torch.no_grad()
    def get_pgt(self, prev_pred_boxes, prev_pred_scores, proposals, suffix):
        if isinstance(prev_pred_scores, torch.Tensor):
            num_preds_per_image = [len(p) for p in proposals]
            prev_pred_scores = prev_pred_scores.split(num_preds_per_image, dim=0)
        else:
            assert isinstance(prev_pred_scores, list)
            assert isinstance(prev_pred_scores[0], torch.Tensor)

        prev_pred_scores = [
            torch.index_select(prev_pred_score, 1, gt_int)
            for prev_pred_score, gt_int in zip(prev_pred_scores, self.gt_classes_img_int)
        ]
        pgt_scores_idxs = [
            torch.max(prev_pred_score, dim=0) for prev_pred_score in prev_pred_scores
        ]
        pgt_scores = [item[0] for item in pgt_scores_idxs]
        pgt_idxs = [item[1] for item in pgt_scores_idxs]

        assert isinstance(prev_pred_boxes, tuple) or isinstance(prev_pred_boxes, list)
        if isinstance(prev_pred_boxes[0], Boxes):
            pgt_boxes = [
                prev_pred_box[pgt_idx] for prev_pred_box, pgt_idx in zip(prev_pred_boxes, pgt_idxs)
            ]
        else:
            assert isinstance(prev_pred_boxes[0], torch.Tensor)
            if self.cls_agnostic_bbox_reg:
                num_preds = [prev_pred_box.size(0) for prev_pred_box in prev_pred_boxes]
                prev_pred_boxes = [
                    prev_pred_box.unsqueeze(1).expand(num_pred, self.num_classes, 4)
                    for num_pred, prev_pred_box in zip(num_preds, prev_pred_boxes)
                ]
            prev_pred_boxes = [
                prev_pred_box.view(-1, self.num_classes, 4) for prev_pred_box in prev_pred_boxes
            ]
            prev_pred_boxes = [
                torch.index_select(prev_pred_box, 1, gt_int)
                for prev_pred_box, gt_int in zip(prev_pred_boxes, self.gt_classes_img_int)
            ]
            pgt_boxes = [
                torch.index_select(prev_pred_box, 0, pgt_idx)
                for prev_pred_box, pgt_idx in zip(prev_pred_boxes, pgt_idxs)
            ]
            pgt_boxes = [pgt_box.view(-1, 4) for pgt_box in pgt_boxes]
            diags = [
                torch.tensor(
                    [i * gt_split.numel() + i for i in range(gt_split.numel())],
                    dtype=torch.int64,
                    device=pgt_boxes[0].device,
                )
                for gt_split in self.gt_classes_img
            ]
            pgt_boxes = [
                torch.index_select(pgt_box, 0, diag) for pgt_box, diag in zip(pgt_boxes, diags)
            ]
            pgt_boxes = [Boxes(pgt_box) for pgt_box in pgt_boxes]

        pgt_classes = self.gt_classes_img_int
        pgt_weights = [
            torch.index_select(pred_logits, 1, pgt_class).reshape(-1)
            for pred_logits, pgt_class in zip(
                self.pred_class_img_logits.split(1, dim=0), pgt_classes
            )
        ]

        targets = [
            Instances(proposals[i].image_size, gt_boxes=pgt_box, gt_classes=pgt_class)
            for i, (pgt_box, pgt_class) in enumerate(zip(pgt_boxes, pgt_classes))
        ]

        self._save_pgt(pgt_boxes, pgt_classes, pgt_scores, "pgt", suffix)

        return targets, pgt_weights

    @torch.no_grad()
    def _save_pgt(self, pgt_boxes, pgt_classes, pgt_scores, prefix, suffix):
        if self.iter % 1280 > 0:
            return
        output_dir = os.path.join(self.output_dir, prefix)
        if self.iter == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        for b, (pgt_box, pgt_class, pgt_score) in enumerate(
            zip(pgt_boxes, pgt_classes, pgt_scores)
        ):
            im = self.images.tensor[b, ...].clone().detach().cpu().numpy()
            channel_swap = (1, 2, 0)
            im = im.transpose(channel_swap)
            pixel_means = [103.939, 116.779, 123.68]
            im += pixel_means
            im = im.astype(np.uint8)
            h, w = im.shape[:2]
            im_pgt = im.copy()

            id_str = "i" + str(self.iter) + "_g" + str(pgt_box.device.index) + "_b" + str(b)
            pgt_box = pgt_box.tensor.clone().detach().cpu().numpy()
            pgt_class = pgt_class.clone().detach().cpu().numpy()
            pgt_score = pgt_score.clone().detach().cpu().numpy()
            for i in range(pgt_box.shape[0]):
                c = pgt_class[i]
                s = pgt_score[i]
                x0, y0, x1, y1 = pgt_box[i, :]
                x0 = int(max(x0, 0))
                y0 = int(max(y0, 0))
                x1 = int(min(x1, w))
                y1 = int(min(y1, h))
                cv2.rectangle(im_pgt, (x0, y0), (x1, y1), (0, 0, 255), 4)
                (tw, th), bl = cv2.getTextSize(str(c), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                cv2.putText(
                    im_pgt, str(c), (x0, y0 + th), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2
                )
                (_, t_h), bl = cv2.getTextSize(str(s), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv2.putText(
                    im_pgt, str(s), (x0 + tw, y0 + th), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1
                )

            save_path = os.path.join(output_dir, id_str + suffix + ".png")
            cv2.imwrite(save_path, im_pgt)

    @torch.no_grad()
    def _save_test(self, predictions, proposals):

        prev_pred_scores = self.box_refinery[-1].predict_probs(predictions, proposals)
        prev_pred_boxes = self.box_refinery[-1].predict_boxes(predictions, proposals)
        prev_pred_scores = [prev_pred_score.detach() for prev_pred_score in prev_pred_scores]
        prev_pred_boxes = [prev_pred_box.detach() for prev_pred_box in prev_pred_boxes]

        self._save_box(prev_pred_boxes, prev_pred_scores, 1, 0.1, 8, "test", "")

        prev_pred_boxes = [p.proposal_boxes for p in proposals]
        num_preds = [len(prev_pred_box) for prev_pred_box in proposals]
        prev_pred_boxes = [
            prev_pred_box.tensor.unsqueeze(1).expand(num_pred, self.num_classes, 4)
            for num_pred, prev_pred_box in zip(num_preds, prev_pred_boxes)
        ]

        prev_pred_scores = [p.objectness_logits for p in proposals]
        prev_pred_scores = [
            prev_pred_score.unsqueeze(1).expand(num_pred, self.num_classes + 1)
            for num_pred, prev_pred_score in zip(num_preds, prev_pred_scores)
        ]

        self._save_box(prev_pred_boxes, prev_pred_scores, 50, -9999, 4, "test", "_rpn")

        self._save_proposal(proposals, "test", "_proposal")

    @torch.no_grad()
    def _save_proposal(self, proposals, prefix, suffix):
        output_dir = os.path.join(self.output_dir, prefix)
        for b, p in enumerate(proposals):
            box = p.proposal_boxes.tensor.clone().detach().cpu().numpy()
            logit = p.objectness_logits.clone().detach().cpu().numpy()
            level_ids = p.level_ids.clone().detach().cpu().numpy()

            gpu_id = p.objectness_logits.device.index
            id_str = "i" + str(self.iter_test) + "_g" + str(gpu_id) + "_b" + str(b)

            save_path = os.path.join(output_dir, id_str + "_box" + suffix + ".npy")
            np.save(save_path, box)

            save_path = os.path.join(output_dir, id_str + "_logit" + suffix + ".npy")
            np.save(save_path, logit)

            save_path = os.path.join(output_dir, id_str + "_level" + suffix + ".npy")
            np.save(save_path, level_ids)

    @torch.no_grad()
    def _save_box(self, prev_pred_boxes, prev_pred_scores, top_k, thres, thickness, prefix, suffix):
        prev_pred_boxes = [
            prev_pred_box.view(-1, self.num_classes, 4) for prev_pred_box in prev_pred_boxes
        ]

        prev_pred_scores = [
            torch.index_select(
                prev_pred_score,
                1,
                torch.arange(self.num_classes).to(prev_pred_score).to(torch.int64),
            )
            for prev_pred_score in prev_pred_scores
        ]

        num_preds = [prev_pred_score.size(0) for prev_pred_score in prev_pred_scores]
        for num_pred in num_preds:
            if num_pred > 0:
                continue
            return
        top_ks = [min(num_pred, top_k) for num_pred in num_preds]
        pgt_scores_idxs = [
            torch.topk(prev_pred_score, top_k, dim=0)
            for prev_pred_score, top_k in zip(prev_pred_scores, top_ks)
        ]
        pgt_scores = [item[0] for item in pgt_scores_idxs]
        pgt_idxs = [item[1] for item in pgt_scores_idxs]
        pgt_idxs = [
            torch.unsqueeze(pgt_idx, 2).expand(top_k, self.num_classes, 4)
            for pgt_idx, top_k in zip(pgt_idxs, top_ks)
        ]
        pgt_boxes = [
            torch.gather(prev_pred_box, 0, pgt_idx)
            for prev_pred_box, pgt_idx in zip(prev_pred_boxes, pgt_idxs)
        ]

        masks = [pgt_score.ge(thres) for pgt_score in pgt_scores]
        pgt_scores = [
            torch.masked_select(pgt_score, mask) for pgt_score, mask in zip(pgt_scores, masks)
        ]
        pgt_boxes = [
            torch.masked_select(
                pgt_box, torch.unsqueeze(mask, 2).expand(top_k, self.num_classes, 4)
            )
            for pgt_box, mask, top_k in zip(pgt_boxes, masks, top_ks)
        ]

        pgt_scores = [pgt_score.reshape(-1) for pgt_score in pgt_scores]
        pgt_boxes = [pgt_box.view(-1, 4) for pgt_box in pgt_boxes]

        output_dir = os.path.join(self.output_dir, prefix)
        if self.iter_test == 0 and self.epoch_test == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        for b, pgt_box in enumerate(pgt_boxes):
            im = self.images.tensor[b, ...].clone().detach().cpu().numpy()
            channel_swap = (1, 2, 0)
            im = im.transpose(channel_swap)
            pixel_means = [103.939, 116.779, 123.68]
            im += pixel_means
            im = im.astype(np.uint8)
            h, w = im.shape[:2]
            im_pgt = im.copy()

            id_str = "i" + str(self.iter_test) + "_g" + str(pgt_box.device.index) + "_b" + str(b)
            pgt_box = pgt_box.cpu().numpy()
            for i in range(pgt_box.shape[0]):
                x0, y0, x1, y1 = pgt_box[i, :]
                x0 = int(max(x0, 0))
                y0 = int(max(y0, 0))
                x1 = int(min(x1, w))
                y1 = int(min(y1, h))
                cv2.rectangle(im_pgt, (x0, y0), (x1, y1), (0, 0, 255), thickness)

            save_path = os.path.join(
                output_dir, id_str + suffix + "_e" + str(self.epoch_test) + ".png"
            )
            cv2.imwrite(save_path, im_pgt)

    def _sample_proposals_wsl(
        self, k, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes,
            self.batch_size_per_images[k],
            self.positive_sample_fractions[k],
            self.num_classes,
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        # return sampled_idxs, gt_classes[sampled_idxs]

        gt_classes_sp = torch.full_like(gt_classes, -1)
        gt_classes_sp[sampled_idxs] = gt_classes[sampled_idxs]
        sampled_idxs = torch.arange(gt_classes.shape[0])
        return sampled_idxs, gt_classes_sp[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals_wsl(
        self, k: int, proposals: List[Instances], targets: List[Instances], ret_MI=False, suffix=""
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_sample_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []
        matched_idxs_ = []

        num_fg_samples = []
        num_bg_samples = []
        num_ig_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # print(k, match_quality_matrix.max(), match_quality_matrix.min(), match_quality_matrix.mean())
            matched_idxs, matched_labels = self.proposal_matchers[k](match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals_wsl(
                k, matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_ig_samples.append((gt_classes == -1).sum().item())
            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1] - num_ig_samples[-1])
            proposals_with_gt.append(proposals_per_image)

            matched_idxs_.append(matched_idxs[sampled_idxs])

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples" + suffix, np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples" + suffix, np.mean(num_bg_samples))
        storage.put_scalar("roi_head/num_ig_samples" + suffix, np.mean(num_ig_samples))

        if ret_MI:
            return proposals_with_gt, matched_idxs_
        return proposals_with_gt


@ROI_HEADS_REGISTRY.register()
class MRRPOICRROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(MRRPOICRROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg, input_shape)
        self._init_mask_head(cfg, input_shape)
        self._init_keypoint_head(cfg, input_shape)

    def _init_box_head(self, cfg, input_shape):
        # fmt: off
        self.num_branch          = cfg.MODEL.MRRP.NUM_BRANCH
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in self.in_features * self.num_branch)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = MRRPROIPooler(
            # self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = WSDDNOutputLayers(cfg, self.box_head.output_shape)

        # OICR
        self.output_dir = cfg.OUTPUT_DIR
        self.iter = 0
        self.refine_K = cfg.WSL.REFINE_NUM
        self.refine_mist = cfg.WSL.REFINE_MIST
        self.refine_reg = cfg.WSL.REFINE_REG
        self.has_gam = cfg.WSL.HAS_GAM
        self.box_refinery = []
        for k in range(self.refine_K):
            box_refinery = OICROutputLayers(cfg, self.box_head.output_shape, k)
            self.add_module("box_refinery_{}".format(k), box_refinery)
            self.box_refinery.append(box_refinery)

        # GAM guided attention module
        if self.has_gam:
            self.box_gam = GAMOutputLayers(in_channels, self.num_classes)

    def _init_mask_head(self, cfg, input_shape):
        # fmt: off
        self.mask_on           = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _init_keypoint_head(self, cfg, input_shape):
        # fmt: off
        self.keypoint_on  = cfg.MODEL.KEYPOINT_ON
        if not self.keypoint_on:
            return
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.keypoint_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.keypoint_head = build_keypoint_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        if self.training and False:
            targets = targets * self.num_branch
            proposals = proposals * self.num_branch
            images.tensor = images.tensor.expand(self.num_branch, -1, -1, -1)
            images.image_sizes = images.image_sizes * self.num_branch

        self.gt_classes_img, self.gt_classes_img_int, self.gt_classes_img_oh = get_image_level_gt(
            targets, self.num_classes
        )
        self.num_preds_per_image = [len(p) for p in proposals]

        # del images
        self.images = images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances, all_scores, all_boxes = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}, all_scores, all_boxes

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.in_features]

        # GAM guided attention module
        if self.has_gam:
            gam = [self.box_gam(feature) for feature in features]
            assert len(features) == 1
            features = [gam[0][0]]
            pred_class_gam_logits = gam[0][1]

        features = [torch.chunk(f, self.num_branch) for f in features]
        features = [ff for f in features for ff in f]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])

        objectness_logits = torch.cat([x.objectness_logits + 1 for x in proposals], dim=0)
        # objectness_logits = torch.cat([x.objectness_logits for x in proposals], dim=0)
        # objectness_logits[objectness_logits > 1] = 0
        # objectness_logits[objectness_logits < -1] = 0
        # objectness_logits = objectness_logits + 1
        box_features = box_features * objectness_logits.view(-1, 1, 1, 1)

        torch.cuda.empty_cache()

        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features, proposals)

        # OICR
        predictions_K = []
        for k in range(self.refine_K):
            predictions_k = self.box_refinery[k](box_features)
            predictions_K.append(predictions_k)

        del box_features

        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            losses = self.box_predictor.losses(predictions, proposals, self.gt_classes_img_oh)

            # OICR
            self.pred_class_img_logits = (
                self.box_predictor.predict_probs_img(predictions, proposals).clone().detach()
            )

            prev_pred_scores = predictions[0].clone().detach()
            prev_pred_boxes = [p.proposal_boxes for p in proposals]
            for k in range(self.refine_K):
                suffix = "_r" + str(k)
                targets, target_weights = self.get_pgt(
                    prev_pred_boxes, prev_pred_scores, proposals, suffix
                )

                proposal_append_gt = self.proposal_append_gt
                self.proposal_append_gt = False
                proposals_k, matched_idxs = self.label_and_sample_proposals(
                    proposals, targets, ret_MI=True, suffix="_r" + str(k)
                )
                self.proposal_append_gt = proposal_append_gt

                # proposal_weights = torch.index_select(pgt_weight, 0, matched_idxs[0])
                proposal_weights = torch.cat(
                    [
                        torch.index_select(target_weight, 0, matched_idx)
                        for target_weight, matched_idx in zip(target_weights, matched_idxs)
                    ],
                    dim=0,
                )

                losses_k = self.box_refinery[k].losses(
                    predictions_K[k], proposals_k, proposal_weights
                )

                prev_pred_scores = self.box_refinery[k].predict_probs(predictions_K[k], proposals_k)
                prev_pred_boxes = self.box_refinery[k].predict_boxes(predictions_K[k], proposals_k)
                prev_pred_scores = [
                    prev_pred_score.clone().detach() for prev_pred_score in prev_pred_scores
                ]
                prev_pred_boxes = [
                    prev_pred_box.clone().detach() for prev_pred_box in prev_pred_boxes
                ]

                losses.update(losses_k)

            # GAM guided attention module
            if self.has_gam:
                outputs = GAMOutputs(pred_class_gam_logits, self.mean_loss, self.gt_classes_img_oh)
                losses.update(outputs.losses())

            self.iter = self.iter + 1
            return losses
        else:
            if self.refine_reg[-1]:
                pred_instances, _, all_scores, all_boxes = self.box_refinery[-1].inference(
                    predictions_K[-1], proposals
                )
            else:
                pred_instances, _, all_scores, all_boxes = self.box_refinery[-1].inference(
                    predictions_K, proposals
                )
            return pred_instances, all_scores, all_boxes

    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            return self.mask_head(mask_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            return self.mask_head(mask_features, instances)

    def _forward_keypoint(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            keypoint_features = self.keypoint_pooler(features, proposal_boxes)
            return self.keypoint_head(keypoint_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.keypoint_pooler(features, pred_boxes)
            return self.keypoint_head(keypoint_features, instances)

    @torch.no_grad()
    def get_pgt_mist(self, prev_pred_boxes, prev_pred_scores, proposals, suffix):
        assert isinstance(prev_pred_boxes, tuple) or isinstance(prev_pred_boxes, list)
        if isinstance(prev_pred_boxes[0], Boxes):
            num_preds = [len(prev_pred_box) for prev_pred_box in prev_pred_boxes]
            prev_pred_boxes = [
                prev_pred_box.tensor.unsqueeze(1).expand(num_pred, self.num_classes, 4)
                for num_pred, prev_pred_box in zip(num_preds, prev_pred_boxes)
            ]
        else:
            assert isinstance(prev_pred_boxes[0], torch.Tensor)
        prev_pred_boxes = [
            prev_pred_box.view(-1, self.num_classes, 4) for prev_pred_box in prev_pred_boxes
        ]

        if isinstance(prev_pred_scores, torch.Tensor):
            prev_pred_scores = prev_pred_scores.split(self.num_preds_per_image, dim=0)
        else:
            assert isinstance(prev_pred_scores, list)
            assert isinstance(prev_pred_scores[0], torch.Tensor)

        prev_pred_scores = [
            torch.index_select(prev_pred_score, 1, gt_int)
            for prev_pred_score, gt_int in zip(prev_pred_scores, self.gt_classes_img_int)
        ]
        prev_pred_boxes = [
            torch.index_select(prev_pred_box, 1, gt_int)
            for prev_pred_box, gt_int in zip(prev_pred_boxes, self.gt_classes_img_int)
        ]

        # get top k
        num_preds = [prev_pred_score.size(0) for prev_pred_score in prev_pred_scores]
        top_ks = [max(int(num_pred * 0.10), 1) for num_pred in num_preds]
        pgt_scores_idxs = [
            torch.topk(prev_pred_score, top_k, dim=0)
            for prev_pred_score, top_k in zip(prev_pred_scores, top_ks)
        ]
        pgt_scores = [item[0] for item in pgt_scores_idxs]
        pgt_idxs = [item[1] for item in pgt_scores_idxs]
        pgt_idxs = [
            torch.unsqueeze(pgt_idx, 2).expand(top_k, gt_int.numel(), 4)
            for pgt_idx, top_k, gt_int in zip(pgt_idxs, top_ks, self.gt_classes_img_int)
        ]
        pgt_boxes = [
            torch.gather(prev_pred_box, 0, pgt_idx)
            for prev_pred_box, pgt_idx in zip(prev_pred_boxes, pgt_idxs)
        ]
        pgt_classes = [
            torch.unsqueeze(gt_int, 0).expand(top_k, gt_int.numel())
            for gt_int, top_k in zip(self.gt_classes_img_int, top_ks)
        ]
        pgt_weights = [
            torch.index_select(pred_logits, 1, gt_int).expand(top_k, gt_int.numel())
            for pred_logits, gt_int, top_k in zip(
                self.pred_class_img_logits.split(1, dim=0), self.gt_classes_img_int, top_ks
            )
        ]

        # get large scores
        masks = [pgt_score.ge(0.05) for pgt_score in pgt_scores]
        # mask[0, :] = True
        masks = [
            torch.cat([torch.full_like(mask[0:1, :], True), mask[1:, :]], dim=0) for mask in masks
        ]
        pgt_scores = [
            torch.masked_select(pgt_score, mask) for pgt_score, mask in zip(pgt_scores, masks)
        ]
        pgt_boxes = [
            torch.masked_select(pgt_box, torch.unsqueeze(mask, 2).expand(top_k, gt_int.numel(), 4))
            for pgt_box, mask, top_k, gt_int in zip(
                pgt_boxes, masks, top_ks, self.gt_classes_img_int
            )
        ]
        pgt_classes = [
            torch.masked_select(pgt_class, mask) for pgt_class, mask in zip(pgt_classes, masks)
        ]
        pgt_weights = [
            torch.masked_select(pgt_weight, mask) for pgt_weight, mask in zip(pgt_weights, masks)
        ]

        pgt_scores = [pgt_score.reshape(-1) for pgt_score in pgt_scores]
        pgt_boxes = [pgt_box.reshape(-1, 4) for pgt_box in pgt_boxes]
        pgt_classes = [pgt_class.reshape(-1) for pgt_class in pgt_classes]
        pgt_weights = [pgt_weight.reshape(-1) for pgt_weight in pgt_weights]

        # NMS
        keeps = [
            batched_nms(pgt_box, pgt_score, pgt_class, 0.10)
            for pgt_box, pgt_score, pgt_class in zip(pgt_boxes, pgt_scores, pgt_classes)
        ]
        pgt_scores = [pgt_score[keep] for pgt_score, keep in zip(pgt_scores, keeps)]
        pgt_boxes = [pgt_box[keep] for pgt_box, keep in zip(pgt_boxes, keeps)]
        pgt_classes = [pgt_class[keep] for pgt_class, keep in zip(pgt_classes, keeps)]
        pgt_weights = [pgt_weight[keep] for pgt_weight, keep in zip(pgt_weights, keeps)]

        pgt_boxes = [Boxes(pgt_box) for pgt_box in pgt_boxes]

        targets = [
            Instances(proposals[i].image_size, gt_boxes=pgt_box, gt_classes=pgt_class)
            for i, (pgt_box, pgt_class) in enumerate(zip(pgt_boxes, pgt_classes))
        ]

        self._save_pgt(pgt_boxes, pgt_classes, pgt_scores, "pgt_mist", suffix)

        return targets, pgt_weights

    @torch.no_grad()
    def get_pgt(self, prev_pred_boxes, prev_pred_scores, proposals, suffix):
        if isinstance(prev_pred_scores, torch.Tensor):
            prev_pred_scores = prev_pred_scores.split(self.num_preds_per_image, dim=0)
        else:
            assert isinstance(prev_pred_scores, list)
            assert isinstance(prev_pred_scores[0], torch.Tensor)

        prev_pred_scores = [
            torch.index_select(prev_pred_score, 1, gt_int)
            for prev_pred_score, gt_int in zip(prev_pred_scores, self.gt_classes_img_int)
        ]
        pgt_scores_idxs = [
            torch.max(prev_pred_score, dim=0) for prev_pred_score in prev_pred_scores
        ]
        pgt_scores = [item[0] for item in pgt_scores_idxs]
        pgt_idxs = [item[1] for item in pgt_scores_idxs]

        assert isinstance(prev_pred_boxes, tuple) or isinstance(prev_pred_boxes, list)
        if isinstance(prev_pred_boxes[0], Boxes):
            pgt_boxes = [
                prev_pred_box[pgt_idx] for prev_pred_box, pgt_idx in zip(prev_pred_boxes, pgt_idxs)
            ]
        else:
            assert isinstance(prev_pred_boxes[0], torch.Tensor)
            prev_pred_boxes = [
                prev_pred_box.view(-1, self.num_classes, 4) for prev_pred_box in prev_pred_boxes
            ]
            prev_pred_boxes = [
                torch.index_select(prev_pred_box, 1, gt_int)
                for prev_pred_box, gt_int in zip(prev_pred_boxes, self.gt_classes_img_int)
            ]
            pgt_boxes = [
                torch.index_select(prev_pred_box, 0, pgt_idx)
                for prev_pred_box, pgt_idx in zip(prev_pred_boxes, pgt_idxs)
            ]
            pgt_boxes = [pgt_box.view(-1, 4) for pgt_box in pgt_boxes]
            diags = [
                torch.tensor(
                    [i * gt_split.numel() + i for i in range(gt_split.numel())],
                    dtype=torch.int64,
                    device=pgt_boxes[0].device,
                )
                for gt_split in self.gt_classes_img
            ]
            pgt_boxes = [
                torch.index_select(pgt_box, 0, diag) for pgt_box, diag in zip(pgt_boxes, diags)
            ]
            pgt_boxes = [Boxes(pgt_box) for pgt_box in pgt_boxes]

        pgt_classes = self.gt_classes_img_int
        pgt_weights = [
            torch.index_select(pred_logits, 1, pgt_class).reshape(-1)
            for pred_logits, pgt_class in zip(
                self.pred_class_img_logits.split(1, dim=0), pgt_classes
            )
        ]

        targets = [
            Instances(proposals[i].image_size, gt_boxes=pgt_box, gt_classes=pgt_class)
            for i, (pgt_box, pgt_class) in enumerate(zip(pgt_boxes, pgt_classes))
        ]

        self._save_pgt(pgt_boxes, pgt_classes, pgt_scores, "pgt", suffix)

        return targets, pgt_weights

    @torch.no_grad()
    def _save_pgt(self, pgt_boxes, pgt_classes, pgt_scores, prefix, suffix):
        if self.iter % 1280 > 0:
            return
        output_dir = os.path.join(self.output_dir, prefix)
        if self.iter == 0:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        for b, (pgt_box, pgt_class, pgt_score) in enumerate(
            zip(pgt_boxes, pgt_classes, pgt_scores)
        ):
            im = self.images.tensor[b, ...].clone().detach().cpu().numpy()
            channel_swap = (1, 2, 0)
            im = im.transpose(channel_swap)
            pixel_means = [103.939, 116.779, 123.68]
            im += pixel_means
            im = im.astype(np.uint8)
            h, w = im.shape[:2]
            im_pgt = im.copy()

            id_str = "i" + str(self.iter) + "_g" + str(pgt_box.device.index) + "_b" + str(b)
            pgt_box = pgt_box.tensor.clone().detach().cpu().numpy()
            pgt_class = pgt_class.clone().detach().cpu().numpy()
            pgt_score = pgt_score.clone().detach().cpu().numpy()
            for i in range(pgt_box.shape[0]):
                c = pgt_class[i]
                s = pgt_score[i]
                x0, y0, x1, y1 = pgt_box[i, :]
                x0 = int(max(x0, 0))
                y0 = int(max(y0, 0))
                x1 = int(min(x1, w))
                y1 = int(min(y1, h))
                cv2.rectangle(im_pgt, (x0, y0), (x1, y1), (0, 0, 255), 4)
                (tw, th), bl = cv2.getTextSize(str(c), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
                cv2.putText(
                    im_pgt, str(c), (x0, y0 + th), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2
                )
                (_, t_h), bl = cv2.getTextSize(str(s), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                cv2.putText(
                    im_pgt, str(s), (x0 + tw, y0 + th), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1
                )

            save_path = os.path.join(output_dir, id_str + "_" + suffix + ".png")
            cv2.imwrite(save_path, im_pgt)


@ROI_HEADS_REGISTRY.register()
class PCLROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(PCLROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg, input_shape)
        self._init_mask_head(cfg, input_shape)
        self._init_keypoint_head(cfg, input_shape)

    def _init_box_head(self, cfg, input_shape):
        # fmt: off
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = WSDDNOutputLayers(cfg, self.box_head.output_shape)

        # PCL
        self.refine_K = cfg.WSL.REFINE_NUM
        self.refine_reg = cfg.WSL.REFINE_REG
        self.box_refinery = []
        for k in range(self.refine_K):
            box_refinery = OICROutputLayers(cfg, self.box_head.output_shape, k)
            self.add_module("box_refinery_{}".format(k), box_refinery)
            self.box_refinery.append(box_refinery)

        # GAM guided attention module
        # if self.has_gam:
        # self.box_gam = GAMOutputLayers(in_channels, self.num_classes)

    def _init_mask_head(self, cfg, input_shape):
        # fmt: off
        self.mask_on           = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _init_keypoint_head(self, cfg, input_shape):
        # fmt: off
        self.keypoint_on  = cfg.MODEL.KEYPOINT_ON
        if not self.keypoint_on:
            return
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.keypoint_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.keypoint_head = build_keypoint_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        self.gt_classes_img, self.gt_classes_img_int, self.gt_classes_img_oh = get_image_level_gt(
            targets, self.num_classes
        )
        self.num_preds_per_image = [len(p) for p in proposals]

        # del images
        self.images = images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances, all_scores, all_boxes = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}, all_scores, all_boxes

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.in_features]

        # GAM guided attention module
        # if self.has_gam:
        # gam = [self.box_gam(feature) for feature in features]
        # assert len(features) == 1
        # features = [gam[0][0]]
        # pred_class_gam_logits = gam[0][1]

        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])

        objectness_logits = torch.cat([x.objectness_logits + 1 for x in proposals], dim=0)
        # objectness_logits = torch.cat([x.objectness_logits for x in proposals], dim=0)
        # objectness_logits[objectness_logits > 1] = 0
        # objectness_logits[objectness_logits < -1] = 0
        # objectness_logits = objectness_logits + 1
        box_features = box_features * objectness_logits.view(-1, 1, 1, 1)

        torch.cuda.empty_cache()

        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features, proposals)

        # PCL
        predictions_K = []
        for k in range(self.refine_K):
            predictions_k = self.box_refinery[k](box_features)
            predictions_K.append(predictions_k)

        del box_features

        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            losses = self.box_predictor.losses(predictions, proposals, self.gt_classes_img_oh)

            # PCL
            self.pred_class_img_logits = (
                self.box_predictor.predict_probs_img(predictions, proposals).clone().detach()
            )

            prev_pred_scores = predictions[0].clone().detach()
            for k in range(self.refine_K):
                losses_k = self.box_refinery[k].losses_pcl(
                    predictions_K[k], proposals, prev_pred_scores, self.gt_classes_img_oh
                )

                prev_pred_scores = self.box_refinery[k].predict_probs(predictions_K[k], proposals)
                prev_pred_scores = [
                    prev_pred_score.clone().detach() for prev_pred_score in prev_pred_scores
                ][0]

                losses.update(losses_k)

            # GAM guided attention module
            # if self.has_gam:
            # outputs = GAMOutputs(pred_class_gam_logits, self.mean_loss, self.gt_classes_img_oh)
            # losses.update(outputs.losses())

            return losses
            # GAM guided attention module
        else:
            if self.refine_reg[-1]:
                pred_instances, _, all_scores, all_boxes = self.box_refinery[-1].inference(
                    predictions_K[-1], proposals, pcl_bg=True
                )
            else:
                pred_instances, _, all_scores, all_boxes = self.box_refinery[-1].inference(
                    predictions_K, proposals, pcl_bg=True
                )
            return pred_instances, all_scores, all_boxes

    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            return self.mask_head(mask_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            return self.mask_head(mask_features, instances)

    def _forward_keypoint(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            keypoint_features = self.keypoint_pooler(features, proposal_boxes)
            return self.keypoint_head(keypoint_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.keypoint_pooler(features, pred_boxes)
            return self.keypoint_head(keypoint_features, instances)


@ROI_HEADS_REGISTRY.register()
class ATTROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(ATTROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)
        self._init_mask_head(cfg)
        self._init_keypoint_head(cfg)

        self.mean_loss = cfg.WSL.MEAN_LOSS

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = ATTOutputLayers(
            self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg
        )

    def _init_mask_head(self, cfg):
        # fmt: off
        self.mask_on           = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _init_keypoint_head(self, cfg):
        # fmt: off
        self.keypoint_on                         = cfg.MODEL.KEYPOINT_ON
        if not self.keypoint_on:
            return
        pooler_resolution                        = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales                            = tuple(1.0 / self.feature_strides[k] for k in self.in_features)  # noqa
        sampling_ratio                           = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type                              = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        self.normalize_loss_by_visible_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS  # noqa
        self.keypoint_loss_weight                = cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.keypoint_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.keypoint_head = build_keypoint_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        self.gt_classes_img, self.gt_classes_img_int, self.gt_classes_img_oh = get_image_level_gt(
            targets, self.num_classes
        )
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        features_list = [features[f] for f in self.in_features]

        if self.training:
            losses = self._forward_box(features_list, proposals)
            # During training the proposals used by the box head are
            # used by the mask, keypoint (and densepose) heads.
            losses.update(self._forward_mask(features_list, proposals))
            losses.update(self._forward_keypoint(features_list, proposals))
            return proposals, losses
        else:
            pred_instances, all_scores, all_boxes = self._forward_box(features_list, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}, all_scores, all_boxes

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        features = [features[f] for f in self.in_features]

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])

        objectness_logits = torch.cat([x.objectness_logits + 1 for x in proposals], dim=0)
        # objectness_logits = torch.cat([x.objectness_logits for x in proposals], dim=0)
        # objectness_logits[objectness_logits > 1] = 0
        # objectness_logits[objectness_logits < -1] = 0
        # objectness_logits = objectness_logits + 1
        box_features = box_features * objectness_logits.view(-1, 1, 1, 1)

        torch.cuda.empty_cache()

        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas, pred_class_img_logits = self.box_predictor(
            box_features
        )
        del box_features

        outputs = ATTOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
            self.mean_loss,
            self.gt_classes_img_oh,
            pred_class_img_logits,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _, all_scores, all_boxes = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances, all_scores, all_boxes

    def _forward_mask(self, features, instances):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (list[Tensor]): #level input features for mask prediction
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            mask_logits = self.mask_head(mask_features)
            return {"loss_mask": mask_rcnn_loss(mask_logits, proposals)}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            mask_logits = self.mask_head(mask_features)
            mask_rcnn_inference(mask_logits, instances)
            return instances

    def _forward_keypoint(self, features, instances):
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (list[Tensor]): #level input features for keypoint prediction
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        num_images = len(instances)

        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            keypoint_features = self.keypoint_pooler(features, proposal_boxes)
            keypoint_logits = self.keypoint_head(keypoint_features)

            normalizer = (
                num_images
                * self.batch_size_per_image
                * self.positive_sample_fraction
                * keypoint_logits.shape[1]
            )
            loss = keypoint_rcnn_loss(
                keypoint_logits,
                proposals,
                normalizer=None if self.normalize_loss_by_visible_keypoints else normalizer,
            )
            return {"loss_keypoint": loss * self.keypoint_loss_weight}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.keypoint_pooler(features, pred_boxes)
            keypoint_logits = self.keypoint_head(keypoint_features)
            keypoint_rcnn_inference(keypoint_logits, instances)
            return instances
