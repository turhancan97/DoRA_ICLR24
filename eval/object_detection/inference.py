from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
import torch
from torchvision import transforms
from eval_object_detect import object_detection_model
import argparse
import random

def inference(args, detection_model):
    # Set the model to evaluation mode
    detection_model.eval()
    detection_model.cuda()

    # Load the fine-tuned model weights
    detection_model.load_state_dict(torch.load(f'{args.output_dir}/{args.backbone_name}_fine_tuned_dora_coco.pth'))

    # Load COCO category names
    coco = COCO(f'{args.data_path}/annotations/instances_val2017.json')  # Path to COCO validation annotations
    category_names = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}

    # Generate unique colors for each category
    random.seed(2345)  # For reproducibility
    category_colors = {cat_id: (random.random(), random.random(), random.random()) for cat_id in category_names}

    # Define prediction function
    def predict(image):
        # Transform the image to tensor
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        image = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Move to device if using GPU
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        detection_model.to(device)
        image = image.to(device)
        
        with torch.no_grad():
            predictions = detection_model(image)
        
        return predictions

    # Load an example image
    image_path = f'{args.image_input_path}'
    image = Image.open(image_path).convert('RGB')
    image = image.resize((256, 256))

    # Get predictions
    predictions = predict(image)
    threshold = 0.7
    predictions[0]['boxes'] = predictions[0]['boxes'][predictions[0]['scores'] > threshold]
    predictions[0]['labels'] = predictions[0]['labels'][predictions[0]['scores'] > threshold]
    predictions[0]['scores'] = predictions[0]['scores'][predictions[0]['scores'] > threshold]
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(12, 8))

    # Display the image
    ax.imshow(image)

    # Move predictions to CPU if on GPU
    for key in predictions[0]:
        if isinstance(predictions[0][key], torch.Tensor):
            predictions[0][key] = predictions[0][key].cpu()

    # Display the bounding boxes and labels
    for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
        x_min, y_min, x_max, y_max = box
        color = category_colors[int(label)]
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=4, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        label_name = category_names[int(label)]
        text = f'{label_name}: {score:.2f}'
        plt.text(x_min, y_min, text, fontsize=16, weight='bold', bbox=dict(facecolor=color, alpha=0.5))

    # Save the image
    plt.savefig(f'{args.image_output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Inference on the Object Detection Model')
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='../../model/venice/checkpoint_all_100.pth', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--data_path', default='../../dataset/pascal', type=str)
    parser.add_argument('--output_dir', default="../../output/all/detection", help='Path to save logs and checkpoints')
    parser.add_argument('--backbone_name', default="vit", help='choose backbone model (cnn or vit)')
    parser.add_argument('--num_labels', default=21, type=int, help='Number of labels for Object Detector')
    parser.add_argument('--image_input_path', default='input/images/pedestrian.jpg', type=str)
    parser.add_argument('--image_output_path', default='output/images/pedestrian.jpg', type=str)
    args = parser.parse_args()
    ###############
    detection_model = object_detection_model(args, dora_backbone=True, num_classes=args.num_labels)
    inference(args, detection_model)