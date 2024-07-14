import sys
sys.path.append('../')
sys.path.append('../../')

import os
import argparse
from functools import partial

import torch
import torchvision
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms
import utils
import vision_transformer as vits

import warnings
warnings.filterwarnings("ignore")
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN

from object_detection.fasterrcnn_vitdet import ViT, SimpleFeaturePyramid
from object_detection.model.layers import LastLevelMaxPool
from object_detection.model.model_summary import summary
from object_detection.torch_utils.engine import evaluate

torch.cuda.empty_cache()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def create_dora_model(args, num_classes=81, pretrained=True):
    # Base
    embed_dim, depth, num_heads, dp = 384, 12, 6, 0.1
    # Load the pretrained backbone.
    net = ViT(  # Single-scale ViT backbone
        img_size=256,
        patch_size=args.patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=dp,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=[
            # 2, 5, 8 11 for global attention
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,
        ],
        residual_block_indexes=[],
        use_rel_pos=True,
        out_feature="last_feat",
    )

    if pretrained:
        print('Loading Pretrained ViT Small Dora Weights...')
        utils.load_pretrained_weights(net, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
        print(f"Model {args.arch} built.")

    backbone = SimpleFeaturePyramid(
        net,
        in_feature="last_feat",
        out_channels=256,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        top_block=LastLevelMaxPool(),
        norm="LN",
        square_pad=256,
    )

    backbone.out_channels = 256

    # Feature maps to perform RoI cropping.
    # If backbone returns a Tensor, `featmap_names` is expected to
    # be [0]. We can choose which feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=backbone._out_features,
        output_size=7,
        sampling_ratio=2
    )

    # Final Faster RCNN model.
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        box_roi_pool=roi_pooler
    )

    return model

def object_detection_model(args, dora_backbone=True, num_classes=81):
    if dora_backbone:
        detection_model = create_dora_model(args, num_classes, pretrained=True)
    else:
        # Load Faster RCNN pre-trained model
        detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
        # Get the number of input features 
        in_features = detection_model.roi_heads.box_predictor.cls_score.in_features
        # define a new head for the detector with required number of classes
        detection_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    detection_model.cuda()
    # detection_model.eval()
    return detection_model

def data_loader(data_path):

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset_train = COCODataset(root=f'{data_path}/train2017', annotation=f'{data_path}/annotations/instances_train2017.json', transforms=transform)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size_per_gpu, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)

    val_dataset = COCODataset(root=f'{data_path}/val2017', annotation=f'{data_path}/annotations/instances_val2017.json', resize = False, transforms=transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size_per_gpu, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # # DataLoader is iterable over Dataset
    # for imgs, annotations in val_loader:
    #     imgs = list(img.to(device) for img in imgs)
    #     annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
    #     print(annotations)
    return train_loader, val_loader

def train(args, detection_model, train_loader, val_loader):

    # Freeze DORA weights
    for param in detection_model.parameters():
        param.requires_grad = True

    params = [p for p in detection_model.parameters() if p.requires_grad]

    # Define the optimizer
    optimizer = optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    detection_model.to(device)
    # Training loop
    for epoch in range(args.epochs):
        loss_train = []
        detection_model.train()
        image_num = 0
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: torch.tensor(v).to(device) for k, v in t.items()} for t in targets]

            loss_dict = detection_model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            if image_num % 100 == 0:
                # print('Total Loss: ', sum(loss_train))
                print('Loss: ', losses.item())

            loss_train.append(losses.item())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            image_num = image_num + 1

        # update the learning rate
        lr_scheduler.step()
        precision, recall, f1_score, mIoU = evaluate(detection_model, val_loader, args, device)
        print(f"Epoch [{epoch+1}/{args.epochs}], Total Train Loss: {sum(loss_train)}")
        print(f"Precision: {precision:.4f}", f"Recall: {recall:.4f}", f"F1 Score: {f1_score:.4f}")
        print(f'Mean IoU: {mIoU:.4f}')

        torch.save(detection_model.state_dict(), f'{args.output_dir}/{args.backbone_name}_fine_tuned_dora_coco.pth')

def resize_image_and_boxes(image, boxes, size=(256, 256)):
    original_width, original_height = image.size
    new_width, new_height = size

    # Resize the image
    image = image.resize((new_width, new_height))

    # Calculate the scaling factor
    x_scale = new_width / original_width
    y_scale = new_height / original_height

    # Resize the bounding boxes
    boxes = boxes.clone()
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * x_scale
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * y_scale

    return image, boxes

class COCODataset(Dataset):
    def __init__(self, root, annotation, resize = True, transforms=None):
        self.root = root
        self.transforms = transforms
        self.resize = resize
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]["file_name"]
        # open the input image
        img = Image.open(os.path.join(self.root, path))
        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        for i in range(num_objs):
            xmin = coco_annotation[i]["bbox"][0]
            ymin = coco_annotation[i]["bbox"][1]
            xmax = xmin + coco_annotation[i]["bbox"][2]
            ymax = ymin + coco_annotation[i]["bbox"][3]

            if xmin >= xmax or ymin >= ymax: # min x > max x or min y > max y
                print('Problem with bbox')
                xmin = 0
                ymin = 0
                xmax = 1
                ymax = 1

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(coco_annotation[i]['category_id'])

        # # Ensure target contains valid boxes
        if not boxes:
            boxes.append([0, 0, 1, 1])
            labels.append(0)

        # boxes as tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels as tensor
        labels = torch.tensor(labels, dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]["area"])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.resize:
            img, my_annotation["boxes"] = resize_image_and_boxes(img, my_annotation["boxes"])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)

# Custom collate function
def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training Object Detection Model with Dora Backbone')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='../../model/venice/checkpoint_all_100.pth', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=12, type=int, help='Number of epochs of training.')
    parser.add_argument('--batch_size_per_gpu', default=4, type=int, help='Per-GPU batch-size')
    parser.add_argument('--data_path', default='../../dataset/pascal', type=str)
    parser.add_argument('--num_workers', default=2, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--output_dir', default="../../output/all/detection", help='Path to save logs and checkpoints')
    parser.add_argument('--backbone_name', default="vit", help='choose backbone model (cnn or vit)')
    parser.add_argument('--num_labels', default=21, type=int, help='Number of labels for Object Detector')
    args = parser.parse_args()
    ###############
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    ###############
    detection_model = object_detection_model(args, dora_backbone=True, num_classes=args.num_labels)
    summary(detection_model, args.batch_size_per_gpu)
    train_loader, val_loader = data_loader(args.data_path)
    train(args, detection_model, train_loader, val_loader)