from eval_object_detect import object_detection_model, COCODataset
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

def data_loader(data_path):

    transform = transforms.Compose([
        # transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    val_dataset = COCODataset(root=f'{data_path}/val2017', annotation=f'{data_path}/annotations/instances_val2017.json', resize=False, transforms=transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)

    return val_loader

# Custom collate function
def collate_fn(batch):
    return tuple(zip(*batch))

def eval_func(args, detection_model, val_loader):
    # Set the model to evaluation mode
    detection_model.eval()
    detection_model.cuda()

    # Load the fine-tuned model weights
    detection_model.load_state_dict(torch.load(f'{args.output_dir}/{args.backbone_name}_fine_tuned_dora_coco.pth'))

    # Initialize metrics
    results = []
    tp = 0  # True positives
    fp = 0  # False positives
    fn = 0  # False negatives
    iou_sum = 0  # Sum of IoUs for mIoU calculation
    total_boxes = 0  # Total number of boxes for mIoU calculation
    iou_threshold = 0.5

    for images, targets in val_loader:
        images = list(image.to('cuda') for image in images)
        with torch.no_grad():
            outputs = detection_model(images)
        
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
                max_iou_idx = -1
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation of Object Detection Model on Validation Dqta')
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='../../model/venice/checkpoint_all_100.pth', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--data_path', default='../../dataset/pascal', type=str)
    parser.add_argument('--output_dir', default="../../output/all/detection", help='Path to save logs and checkpoints')
    parser.add_argument('--backbone_name', default="vit", help='choose backbone model (cnn or vit)')
    parser.add_argument('--num_labels', default=21, type=int, help='Number of labels for Object Detector')
    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    detection_model = object_detection_model(args, dora_backbone=True, num_classes=args.num_labels)
    val_loader = data_loader(args.data_path)
    precision, recall, f1_score, mIoU = eval_func(args, detection_model, val_loader)
    print(f"Precision: {precision:.4f}", f"Recall: {recall:.4f}", f"F1 Score: {f1_score:.4f}")
    print(f'Mean IoU: {mIoU:.4f}')