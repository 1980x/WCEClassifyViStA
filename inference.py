import os
import argparse

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from WCE_classification import CustomResNet18WithMask, CustomVGGWithMask, CustomDatasetWithMask

def test(pretrained_model_path, test_data_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs('./seg_inferences', exist_ok=True) # directory where predicted segmentation masks will be written

    
    model_dict = torch.load(pretrained_model_path)#'./checkpoints_custom/49_model.pth
    resnet_model = CustomResNet18WithMask()
    resnet_model.load_state_dict(model_dict['resnet18'])
    resnet_model.to(device)
    resnet_model.eval()

    vgg_model = CustomVGGWithMask()
    vgg_model.load_state_dict(model_dict['vgg'])
    vgg_model.to(device)
    vgg_model.eval()

    image_paths = [os.path.join(test_data_path, fname) for fname in sorted(os.listdir(test_data_path)) if fname.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    labels = [0] * len(image_paths) # dummy labels just to use the same dataset class; we will actually predict the labels

    test_dataset = CustomDatasetWithMask(image_paths, labels, train=False, inference=True) 

    predictions = []

    print("Inference in progress...................")
    for i, (image, mask, labels,img_name) in enumerate(test_loader):

        image = image.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            resnet_output, _, resnet_seg_output = resnet_model(image, mask)
            vgg_output, _, vgg_seg_output = vgg_model(image, mask)

        combined_probs = (torch.sigmoid(resnet_output) + torch.sigmoid(vgg_output)) / 2
        combined_probs = combined_probs.squeeze().item()

        # Normalize segmentation outputs
        combined_seg_output = (resnet_seg_output + vgg_seg_output) / 2
        combined_seg_output_normalized = torch.nn.functional.softmax(combined_seg_output, dim=1)

        # Save the segmentation mask
        seg_output_path = os.path.join('./seg_inferences', img_name[0].split('/')[-1])
        torchvision.utils.save_image(combined_seg_output_normalized, seg_output_path)

        # Store predictions
        class_prob = combined_probs
        class_idx = int(combined_probs > 0.5)
        predictions.append((os.path.basename(img_name[0]), class_idx, class_prob))
    
    c1 = 0
    c2 = 0
    # Write predictions to inferences.txt
    with open('inferences.csv', 'w') as f:
        for img_name, class_idx, class_prob in predictions:
            if class_idx == 1:
                label_name = 'bleeding'
            elif class_idx == 0:
                label_name = 'non-bleeding'
            f.write(f'{img_name}, {label_name}\n')
            if class_idx == 1:
                c1 += 1
            elif class_idx == 0:
                c2 += 1

            #f.write(f'{img_name} - Class: {class_idx}, Confidence: {class_prob:.4f}\n')

    # Compute metrics (assuming true_labels contains ground truth class indices for the test dataset)
    """true_labels = [1]*len(predictions)# Populate this with the ground truth class indices; currently all labels set to positive class
    predicted_labels = [class_idx for _, class_idx, _ in predictions]

    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    accuracy = accuracy_score(true_labels, predicted_labels)

    # Print metrics
    print(f'F1-Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Acuracy: {accuracy:.4f}')"""
    print(f"Sairam, len:{len(test_dataset)}, positives:{c1}, -ves: {c2}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Your script description here.')

    parser.add_argument('--model_path', type=str, default='../checkpoints_custom/49_model.pth', help='Pytorch pretrained model path.')
    parser.add_argument('--test_dir', type=str, default='', help='Test data path')

    args = parser.parse_args()
    
    test(args.model_path, args.test_dir)

