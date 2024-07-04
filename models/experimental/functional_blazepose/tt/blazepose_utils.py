import numpy as np
import torch
import ttnn
from models.experimental.functional_blazepose.tt.blazepose_model import blazepose


# Converting this method to ttnn creates infinite loop


def decode_boxes(raw_boxes, anchors, device):
    """Converts the predictions into actual coordinates using
    the anchor boxes. Processes the entire batch at once.
    """
    boxes = ttnn.zeros_like(raw_boxes)
    x_scale = 128.0
    y_scale = 128.0
    h_scale = 128.0
    w_scale = 128.0
    num_keypoints = 4

    raw_boxes = ttnn.from_device(raw_boxes)
    anchors = ttnn.from_device(anchors)

    raw_box = ttnn.to_layout(raw_boxes[:, :, 0:1], layout=ttnn.TILE_LAYOUT)
    raw_box = ttnn.to_device(raw_box, device=device)

    anchor_0 = ttnn.to_layout(anchors[:, 2:3], layout=ttnn.TILE_LAYOUT)
    anchor_0 = ttnn.to_device(anchor_0, device=device)

    anchor_1 = ttnn.to_layout(anchors[:, 0:1], layout=ttnn.TILE_LAYOUT)
    anchor_1 = ttnn.to_device(anchor_1, device=device)
    x_center = (raw_box * (1 / x_scale)) * anchor_0 + anchor_1

    raw_box = ttnn.to_layout(raw_boxes[:, :, 2:3], layout=ttnn.TILE_LAYOUT)
    raw_box = ttnn.to_device(raw_box, device=device)

    anchor_0 = ttnn.to_layout(anchors[:, 2:3], layout=ttnn.TILE_LAYOUT)
    anchor_0 = ttnn.to_device(anchor_0, device=device)

    w = (raw_box * (1 / w_scale)) * anchor_0

    raw_box = ttnn.to_layout(raw_boxes[:, :, 3:4], layout=ttnn.TILE_LAYOUT)
    raw_box = ttnn.to_device(raw_box, device=device)

    anchor_0 = ttnn.to_layout(anchors[:, 3:], layout=ttnn.TILE_LAYOUT)
    anchor_0 = ttnn.to_device(anchor_0, device=device)

    h = (raw_box * (1 / h_scale)) * anchor_0

    raw_boxes = ttnn.to_torch(raw_boxes)
    anchors = ttnn.to_torch(anchors)

    y_center = raw_boxes[..., 1] / y_scale * anchors[:, 3] + anchors[:, 1]

    y_center = ttnn.from_torch(y_center.unsqueeze(-1), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    raw_boxes = ttnn.from_torch(raw_boxes, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    anchors = ttnn.from_torch(anchors, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    box_1 = y_center - (h * (1 / 2.0))  # ymin
    box_2 = x_center - (w * (1 / 2.0))  # xmin
    box_3 = y_center + (h * (1 / 2.0))  # ymax
    box_4 = x_center + (w * (1 / 2.0))  # xmax

    box = boxes[:, :, 4:]

    box_1 = ttnn.to_torch(box_1)
    box_2 = ttnn.to_torch(box_2)
    box_3 = ttnn.to_torch(box_3)
    box_4 = ttnn.to_torch(box_4)

    box = ttnn.to_torch(box)
    # Current concat implementation requires aligned last dim when concatting on last dim
    boxes = torch.concat([box_1, box_2, box_3, box_4, box], dim=-1)
    boxes = ttnn.from_torch(boxes, dtype=ttnn.bfloat16, device=device)

    boxes = ttnn.to_torch(boxes)

    for k in range(num_keypoints):
        offset = 4 + k * 2
        raw_box = ttnn.to_layout(raw_boxes[:, :, offset : offset + 1], layout=ttnn.TILE_LAYOUT)
        raw_box = ttnn.to_device(raw_box, device=device)

        anchor_0 = ttnn.to_layout(anchors[:, 2:3], layout=ttnn.TILE_LAYOUT)
        anchor_0 = ttnn.to_device(anchor_0, device=device)

        anchor_1 = ttnn.to_layout(anchors[:, 0:1], layout=ttnn.TILE_LAYOUT)
        anchor_1 = ttnn.to_device(anchor_1, device=device)
        keypoint_x = (raw_box * (1 / x_scale)) * anchor_0 + anchor_1

        raw_boxes = ttnn.to_torch(raw_boxes)
        anchors = ttnn.to_torch(anchors)
        keypoint_y = raw_boxes[..., offset + 1] / y_scale * anchors[:, 3] + anchors[:, 1]

        keypoint_x = ttnn.to_torch(keypoint_x).squeeze(-1)
        boxes[..., offset] = keypoint_x
        boxes[..., offset + 1] = keypoint_y
        raw_boxes = ttnn.from_torch(raw_boxes, dtype=ttnn.bfloat16)
        anchors = ttnn.from_torch(anchors, dtype=ttnn.bfloat16)

    boxes = ttnn.from_torch(boxes, dtype=ttnn.bfloat16)

    return boxes


def tensors_to_detections(raw_box_tensor, raw_score_tensor, anchors, device):
    num_anchors = 896
    assert len(anchors.shape) == 2
    assert anchors.shape[0] == num_anchors
    assert anchors.shape[1] == 4

    num_coords = 12
    num_classes = 1

    assert raw_box_tensor.shape[1] == num_anchors
    assert raw_box_tensor.shape[2] == num_coords

    assert raw_score_tensor.shape[1] == num_anchors
    assert raw_score_tensor.shape[2] == num_classes

    assert raw_box_tensor.shape[0] == raw_score_tensor.shape[0]

    detection_boxes = decode_boxes(raw_box_tensor, anchors, device)
    score_clipping_thresh = 100.0
    thresh = score_clipping_thresh

    raw_score_tensor = ttnn.to_layout(raw_score_tensor, layout=ttnn.TILE_LAYOUT)
    raw_score_tensor = ttnn.to_device(raw_score_tensor, device=device)

    raw_score_tensor = ttnn.clamp(raw_score_tensor, -thresh, thresh)
    detection_scores = ttnn.sigmoid(raw_score_tensor)
    detection_scores = ttnn.to_torch(detection_scores)
    detection_scores = detection_scores.squeeze(dim=-1)

    # Note: we stripped off the last dimension from the scores tensor
    # because there is only has one class. Now we can simply use a mask
    # to filter out the boxes with too low confidence.
    min_score_thresh = 0.75
    mask = detection_scores >= min_score_thresh
    mask = torch.load("mask.pt")

    # ttnn tensor can not be used as index calue
    detection_boxes = ttnn.to_torch(detection_boxes)

    # Because each image from the batch can have a different number of
    # detections, process them one at a time using a loop.
    # mask is a bool tensor, ttnn doesnt support bool tensor
    output_detections = []

    for i in range(raw_box_tensor.shape[0]):
        boxes = detection_boxes[i, mask[i]]
        scores = detection_scores[i, mask[i]].unsqueeze(dim=-1)
        output_detections.append(ttnn.from_torch(torch.cat((boxes, scores), dim=-1), dtype=ttnn.bfloat16))
    return output_detections


def intersect(box_a, box_b, device):
    A = box_a.shape[0]
    B = box_b.shape[0]

    tensor_box_A = ttnn.to_torch(box_a)
    tensor_box_B = ttnn.to_torch(box_b)

    # ttnn do not have support for unsqueeze and expand
    tensor_A = tensor_box_A[:, 2:].unsqueeze(1).expand(A, B, 2)
    tensor_B = tensor_box_B[:, 2:].unsqueeze(0).expand(A, B, 2)
    tensor_A = ttnn.from_torch(tensor_A, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tensor_B = ttnn.from_torch(tensor_B, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    max_xy = ttnn.minimum(tensor_A, tensor_B)

    tensor_A = tensor_box_A[:, :2].unsqueeze(1).expand(A, B, 2)
    tensor_B = tensor_box_B[:, :2].unsqueeze(0).expand(A, B, 2)
    tensor_A = ttnn.from_torch(tensor_A, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    tensor_B = ttnn.from_torch(tensor_B, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    min_xy = ttnn.maximum(tensor_A, tensor_B)

    # High value is temporary
    inter = ttnn.clamp((max_xy - min_xy), low=0, high=500)
    return inter[:, :, 0:1] * inter[:, :, 1:2]


def jaccard(box_a, box_b, device):
    inter = intersect(box_a, box_b, device)
    inter = ttnn.to_torch(inter).squeeze(-1)
    box_a = ttnn.to_torch(box_a)
    box_b = ttnn.to_torch(box_b)

    # ttnn do not have support for unsqueeze and expand_as
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]

    area_a = ttnn.from_torch(area_a, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    area_b = ttnn.from_torch(area_b, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    inter = ttnn.from_torch(inter, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
    union = area_a + area_b - inter
    union = ttnn.reciprocal(union)

    return inter * union  # [A,B]


def overlap_similarity(box, other_boxes, device):
    """Computes the IOU between a bounding box and set of other boxes."""
    return jaccard(box, other_boxes, device)  # .squeeze(0)


def weighted_non_max_suppression(detections, device):
    output_detections = []
    num_coords = 12
    # Sort the detections from highest to lowest score.
    detections = ttnn.to_torch(detections)
    remaining = torch.argsort(detections[:, num_coords], descending=True)
    detections = ttnn.from_torch(detections, dtype=ttnn.bfloat16)  # , layout = ttnn.TILE_LAYOUT, device = device)

    # remaining is torch tensor as ttnn tensor cannot be used for indexing
    while len(remaining) > 0:
        if remaining[0] == 0:
            detection = detections[remaining[0] : 1, :]
        else:
            detection = detections[remaining[0] : remaining[0] + 1, :]

        # Compute the overlap between the first box and the other
        # remaining boxes. (Note that the other_boxes also include
        # the first_box.)

        first_box = detection[:, :4]

        i = 0
        other_boxes = []
        while i < len(remaining):
            if remaining[i] == 0:
                tensor = detections[0:1, 0:4]
            else:
                tensor = detections[remaining[i] - 1 : remaining[i], 0:4]
            i += 1

            other_boxes.append(tensor)

        other_boxes = ttnn.concat(other_boxes, dim=0)

        ious = overlap_similarity(first_box, other_boxes, device)

        # If two detections don't overlap enough, they are considered
        # to be from different faces.
        min_suppression_threshold = 0.3
        mask = ious > min_suppression_threshold
        mask = ttnn.to_torch(mask).squeeze(0)

        i = 0
        overlap = []
        mask_idx = []

        while i < mask.shape[-1]:
            mask_idx.append(mask[i].item())
            if mask[i] == 0:
                overlap.append(remaining[0:1])
            else:
                overlap.append(remaining[int(mask[i].item()) : int(mask[i].item()) + 1])
            i += 1

        overlapping = torch.concat(overlap, dim=-1)

        rem = []
        i = 0
        while i < mask.shape[-1]:
            if mask[i] != 1:
                if i == 0:
                    rem.append(remaining[0:1])
                else:
                    rem.append(remaining[i : i + 1])
            i += 1

        if rem:
            remaining = torch.concat(rem, dim=-1)
        else:
            remaining = []

        # Take an average of the coordinates from the overlapping
        # detections, weighted by their confidence scores.
        weighted_detection = detection
        weighted_detection = ttnn.to_torch(weighted_detection).squeeze(0)

        if overlapping.shape[0] > 1:
            score_detecion = detections[:, num_coords : num_coords + 1]

            i = 0
            coordinates = []
            scores = []
            while i < overlapping.shape[0]:
                coordinates.append(detections[overlapping[i] : overlapping[i] + 1, 0:num_coords])
                scores.append(ttnn.to_torch(score_detecion[overlapping[i] : overlapping[i] + 1, :]))
                i += 1
            coordinates = ttnn.concat(coordinates, dim=0)

            scores = torch.concat(scores, dim=0)
            scores = ttnn.to_device(
                ttnn.from_torch(scores, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), device=device
            )

            total_score = ttnn.reciprocal(ttnn.sum(scores))

            coordinates = ttnn.to_torch(coordinates)
            scores = ttnn.to_torch(scores)
            total_score = ttnn.to_torch(total_score)

            weighted = (coordinates * scores).sum(dim=0) * total_score
            weighted_detection[:num_coords] = weighted
            weighted_detection[num_coords] = total_score / (overlapping.shape[0])

        output_detections.append(weighted_detection)

    return output_detections


def predict_on_batch(x, anchors, parameters, device):
    x_scale = 128.0
    y_scale = 128.0
    assert x.shape[1] == 3
    assert x.shape[2] == y_scale
    assert x.shape[3] == x_scale

    x = ttnn.to_torch(x)

    x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)  # , layout = ttnn.TILE_LAYOUT)
    x = x * (1 / 255.0)

    x = ttnn.permute(x, (0, 2, 3, 1))
    x = ttnn.to_layout(ttnn.from_device(x), layout=ttnn.ROW_MAJOR_LAYOUT)

    with torch.no_grad():
        out = blazepose(x, parameters, device)
        out1 = ttnn.to_torch(out[0])
        out2 = ttnn.to_torch(out[1])

    out1 = ttnn.from_torch(out1, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    out2 = ttnn.from_torch(out2, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    detections = tensors_to_detections(out1, out2, anchors, device)
    num_coords = 12
    filtered_detections = []
    for i in range(len(detections)):
        faces = weighted_non_max_suppression(detections[i], device)
        faces = torch.stack(faces) if len(faces) > 0 else torch.zeros((0, num_coords + 1))
        filtered_detections.append(faces)
    return filtered_detections


def predict_on_image(img, parameters, anchors, device):
    return predict_on_batch(img, anchors, parameters, device)[0]
