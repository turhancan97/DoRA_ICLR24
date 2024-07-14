import sys
sys.path.append('object_detection/torch_utils/')

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import json

@torch.inference_mode()
def evaluate(model, data_loader, args, device):
    model.eval()

    # Initialize metrics
    results = []
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    iou_sum = 0  # Sum of IoUs for mIoU calculation
    total_boxes = 0  # Total number of boxes for mIoU calculation
    iou_threshold = 0.5

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        with torch.no_grad():
            outputs = model(images)
        
        for target, output in zip(targets, outputs):
            image_id = target["image_id"].item()
            gt_labels = target['labels'].cpu().numpy()
            gt_boxes = target['boxes'].cpu().numpy()

            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            matched_gt = []

            for box, score, label in zip(boxes, scores, labels):
                x_min, y_min, x_max, y_max = box
                result = {
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)],
                    "score": float(score)
                }
                results.append(result)

                # Calculate IoU and determine if the detection is a true positive
                max_iou = 0
                for i, gt_box in enumerate(gt_boxes):
                    if label == gt_labels[i] and i not in matched_gt:
                        iou = calculate_iou(box, gt_box)
                        if iou > max_iou:
                            max_iou = iou
                            max_iou_idx = i

                if max_iou >= iou_threshold:
                    tp += 1
                    matched_gt.append(max_iou_idx)
                else:
                    fp += 1
                    
                iou_sum += max_iou
                total_boxes += 1

            # Any unmatched ground truth boxes are false negatives
            fn += len(gt_boxes) - len(matched_gt)

    # Calculate precision, recall, F1 score, and mIoU
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mIoU = iou_sum / total_boxes if total_boxes > 0 else 0

    # Save the results to a JSON file
    with open(f'{args.output_dir}/detections.json', "w") as f:
        json.dump(results, f)

    # Load the COCO validation ground truth
    coco_gt = COCO(f'{args.data_path}/annotations/instances_val2017.json')

    # Load the results
    coco_dt = coco_gt.loadRes(f'{args.output_dir}/detections.json')

    # Initialize COCOeval
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')

    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return precision, recall, f1_score, mIoU

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

# @torch.inference_mode()
# def evaluate(model, data_loader, device):
#     loss_val = []
#     with torch.no_grad():
#         for images, targets in data_loader:
#             images = list(img.to(device) for img in images)
#             targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]
#             loss_dict = model(images, targets)
#             losses = sum(loss for loss in loss_dict.values())
#             loss_val.append(losses.item())
#     return loss_val
