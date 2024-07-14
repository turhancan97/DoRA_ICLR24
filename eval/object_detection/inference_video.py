from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
import torch
from torchvision import transforms
from eval_object_detect import object_detection_model
import cv2
import argparse
import random

def inference_video(args, detection_model):
    # Set the model to evaluation mode
    detection_model.eval()
    detection_model.cuda()

    # Load the fine-tuned model weights
    detection_model.load_state_dict(torch.load(f'{args.output_dir}/{args.backbone_name}_fine_tuned_dora_coco.pth'))

    # Load COCO category names
    coco = COCO(f'{args.data_path}/annotations/instances_val2017.json')  # Path to COCO validation annotations
    category_names = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}

    # Generate unique colors for each category
    random.seed(42)  # For reproducibility
    category_colors = {cat_id: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for cat_id in category_names}

    # Define prediction function
    def predict(image):
        # Transform the image to tensor
        transform = transforms.Compose([
            # transforms.Resize((256, 256)),
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

    # Video processing
    video_path = f'{args.video_input_path}'
    output_path = f'{args.video_output_path}'
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        original_width, original_height = image.size

        # Get predictions
        predictions = predict(image)
        threshold = 0.7
        predictions[0]['boxes'] = predictions[0]['boxes'][predictions[0]['scores'] > threshold]
        predictions[0]['labels'] = predictions[0]['labels'][predictions[0]['scores'] > threshold]
        predictions[0]['scores'] = predictions[0]['scores'][predictions[0]['scores'] > threshold]

        # Draw bounding boxes and labels on the frame
        for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
            x_min, y_min, x_max, y_max = box
            # x_min = int(x_min * frame_width / 256)
            # y_min = int(y_min * frame_height / 256)
            # x_max = int(x_max * frame_width / 256)
            # y_max = int(y_max * frame_height / 256)
            color = category_colors[int(label)]
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 3)
            label_name = category_names[int(label)]
            text = f'{label_name}: {score:.2f}'
            ((text_width, text_height), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (int(x_min), int(y_min) - int(1.3 * text_height)), (int(x_min) + text_width, int(y_min)), color, -1)
            cv2.putText(frame, text, (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Write the frame with detection boxes to the output video
        out.write(frame)

    # Release everything if job is finished
    cap.release()
    out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Video Inference on the Object Detection Model')
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='../../model/venice/checkpoint_all_100.pth', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--data_path', default='../../dataset/pascal', type=str)
    parser.add_argument('--output_dir', default="../../output/all/detection", help='Path to save logs and checkpoints')
    parser.add_argument('--backbone_name', default="vit", help='choose backbone model (cnn or vit)')
    parser.add_argument('--num_labels', default=21, type=int, help='Number of labels for Object Detector')
    parser.add_argument('--video_input_path', default='input/videos/venice4.mp4', type=str)
    parser.add_argument('--video_output_path', default='output/videos/venice4.mp4', type=str)
    args = parser.parse_args()
    ###############
    detection_model = object_detection_model(args, dora_backbone=True, num_classes=args.num_labels)
    inference_video(args, detection_model)