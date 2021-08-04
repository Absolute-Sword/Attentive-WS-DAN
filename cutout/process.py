import random
import torch
from utils import intersect


def cutout_process(image, boxes, labels, fill_val=0, bbox_remove_thres=0.4):
    image = image[0]
    boxes = boxes[0]
    labels = labels[0]
    original_h = image.size(1)
    original_w = image.size(2)
    original_channel = image.size(0)

    new_image = image

    for _ in range(50):
        # Random cutout size: [0.15, 0.5] of original dimension
        cutout_size_h = random.uniform(0.15 * original_h, 0.5 * original_h)
        cutout_size_w = random.uniform(0.15 * original_w, 0.5 * original_w)

        # Random position for cutout
        left = random.uniform(0, original_w - cutout_size_w)
        right = left + cutout_size_w
        top = random.uniform(0, original_h - cutout_size_h)
        bottom = top + cutout_size_h
        cutout = torch.FloatTensor([int(left), int(top), int(right), int(bottom)])

        # Calculate intersect between cutout and bounding boxes
        overlap_size = intersect(cutout.unsqueeze(0), boxes)
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        ratio = overlap_size / area_boxes
        # If all boxes have Iou greater than bbox_remove_thres, try again
        if ratio.min().item() > bbox_remove_thres:
            continue

        cutout_arr = torch.full((original_channel, int(bottom) - int(top), int(right) - int(left)), fill_val)
        new_image[:, int(top):int(bottom), int(left):int(right)] = cutout_arr

        # Create new boxes and labels
        boolean = ratio < bbox_remove_thres

        new_boxes = boxes[boolean[0], :]

        new_labels = labels[boolean[0]]

        return new_image.unsqueeze(dim=0), new_boxes.unsqueeze(dim=0), new_labels.unsqueeze(dim=0)
