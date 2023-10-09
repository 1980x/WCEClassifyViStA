import os
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np

############################################### loss function for segmentation branch #########################################

def dice_coefficient(predicted, target):
    predicted = predicted.clip(-2, 2)
    intersection = torch.sum(predicted * target)
    union = torch.sum(predicted) + torch.sum(target)
    dice = (2.0 * intersection) / (union + 1e-5)  
    return dice

def hybrid_seg_loss(predicted, target, alpha=0.5, beta=0.5):
    dice = 1 - dice_coefficient(predicted, target)
    ce = F.cross_entropy(predicted.view(predicted.shape[0], 2, -1), target.view(predicted.shape[0], -1), reduction='mean')

    loss = alpha * dice + beta * ce
    return loss

#############################################################################

############################################# U-net styled decoder for segmentation branch ######################################

class Decoder(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(Decoder, self).__init__()

        self.conv = nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1) #keep ratio
        self.conv_trans = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(mid_channel, out_channel, 1)

    def forward(self, x):
        x = F.relu(self.conv(x), inplace=True)
        x = F.relu(self.conv_trans(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        return x
############################################################################################

############################################## Our ClassifyViStA network comprising of ensemble of encoder-decoder styled resnet18 and vgg16

class CustomResNet18WithMask(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.resnet18 = torchvision.models.resnet18()
        
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_features, num_classes)
        
        # center
        self.center = nn.Conv2d(512, 256, 3, stride=1, padding=1) #Decoder(512, 312, 256)

        #decoder for resnet18
        self.decoder5 = Decoder(256+512, 256, 256)
        self.decoder4 = Decoder(256+256, 128, 128)
        self.decoder3 = Decoder(128+128, 64, 64)
        self.decoder2 = Decoder(64+64, 32, 32)
        self.decoder1 = Decoder(32, 16, 16)
        self.decoder0 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)

        self.seg = nn.Conv2d(8, num_classes+1, kernel_size=1)

    def forward(self, img, mask):
        cache_for_seg = []

        # Downsample mask to match the size of layer 4 output (7x7)
        mask_downsampled = F.interpolate(mask, size=7, mode='bilinear', align_corners=False)
        
        # Forward pass through the ResNet-18 model for the image
        img_output = self.resnet18.maxpool(self.resnet18.relu(self.resnet18.bn1(self.resnet18.conv1(img))))
        img_output = self.resnet18.layer1(img_output)
        cache_for_seg.append(img_output)
        img_output = self.resnet18.layer2(img_output)
        cache_for_seg.append(img_output)
        img_output = self.resnet18.layer3(img_output)
        cache_for_seg.append(img_output)
        img_output = self.resnet18.layer4(img_output)
        cache_for_seg.append(img_output)
                
        # Element-wise multiplication of layer 4 output and downsampled mask
        img_output_masked = img_output * mask_downsampled
        
        # Forward pass through the modified layer (average pooling and final fc layer)
        masked_output = self.resnet18.avgpool(img_output_masked)
        masked_output = masked_output.view(masked_output.size(0), -1)
        masked_output = self.resnet18.fc(masked_output)
        
        img_output = self.resnet18.avgpool(img_output)
        img_output = img_output.view(img_output.size(0), -1)
        img_output = self.resnet18.fc(img_output)
       
        #decoding for seg
        center = F.relu(self.center(cache_for_seg[-1])) #7x7
        dec5 = self.decoder5(torch.cat([center, cache_for_seg[-1]], 1)) #14x14
        dec4 = self.decoder4(torch.cat([dec5, cache_for_seg[-2]], 1)) #28x28
        dec3 = self.decoder3(torch.cat([dec4, cache_for_seg[-3]], 1)) #56x56
        dec2 = self.decoder2(torch.cat([dec3, cache_for_seg[-4]], 1)) #112x112
        dec1 = self.decoder1(dec2) #224x224
        dec0 = F.relu(self.decoder0(dec1))
        seg = self.seg(dec0)

        return img_output, masked_output, seg

class CustomVGGWithMask(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.vgg = torchvision.models.vgg16_bn()
        
        num_features = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(num_features, num_classes)
        
        #center
        self.center = nn.Conv2d(512, 256, 3, stride=1, padding=1) 

        #decoder for vgg16_bn
        self.decoder5 = Decoder(256+512, 256, 256)
        self.decoder4 = Decoder(256+512, 128, 128)
        self.decoder3 = Decoder(128+256, 64, 64)
        self.decoder2 = Decoder(64+128, 32, 32)
        self.decoder1 = Decoder(32, 16, 16)
        self.decoder0 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)

        self.seg = nn.Conv2d(8, num_classes+1, kernel_size=1)

    def forward(self, img, mask):
        cache_for_seg = []

        # Downsample mask to match the size of layer 4 output (7x7)
        mask_downsampled = F.interpolate(mask, size=7, mode='bilinear', align_corners=False)
        
        # Forward pass through the vgg  model for the image
        y1 = self.vgg.features[2](self.vgg.features[1](self.vgg.features[0](img)))
        y1 = self.vgg.features[5](self.vgg.features[4](self.vgg.features[3](y1)))
        y1 = self.vgg.features[6](y1)
        y2 = self.vgg.features[9](self.vgg.features[8](self.vgg.features[7](y1)))
        y2 = self.vgg.features[12](self.vgg.features[11](self.vgg.features[10](y2)))
        y2 = self.vgg.features[13](y2)
        y3 = self.vgg.features[16](self.vgg.features[15](self.vgg.features[14](y2)))
        y3 = self.vgg.features[19](self.vgg.features[18](self.vgg.features[17](y3)))
        y3 = self.vgg.features[22](self.vgg.features[21](self.vgg.features[20](y3)))
        y3 = self.vgg.features[23](y3)
        y4 = self.vgg.features[26](self.vgg.features[25](self.vgg.features[24](y3)))
        y4 = self.vgg.features[29](self.vgg.features[28](self.vgg.features[27](y4)))
        y4 = self.vgg.features[32](self.vgg.features[31](self.vgg.features[30](y4)))
        y4 = self.vgg.features[33](y4)
        y5 = self.vgg.features[36](self.vgg.features[35](self.vgg.features[34](y4)))
        y5 = self.vgg.features[39](self.vgg.features[38](self.vgg.features[37](y5)))
        y5 = self.vgg.features[42](self.vgg.features[41](self.vgg.features[40](y5)))
        y5 = self.vgg.features[43](y5)
        y6 = self.vgg.avgpool(y5)
        img_output = self.vgg.classifier(y6.view(-1, 25088))
        cache_for_seg.extend([y2, y3, y4, y5])
                
        # Element-wise multiplication of layer 4 output and downsampled mask
        img_output_masked = y6 * mask_downsampled
        
        # Forward pass through the classifier head (average pooling and final fc layer)
        masked_output = img_output_masked.view(img_output_masked.size(0), -1)
        masked_output = self.vgg.classifier(masked_output)
         
        #decoding for seg
        center = F.relu(self.center(cache_for_seg[-1])) #7x7

        #decoder
        dec5 = self.decoder5(torch.cat([center, cache_for_seg[-1]], 1)) #14x14
        dec4 = self.decoder4(torch.cat([dec5, cache_for_seg[-2]], 1)) #28x28
        dec3 = self.decoder3(torch.cat([dec4, cache_for_seg[-3]], 1)) #56x56
        dec2 = self.decoder2(torch.cat([dec3, cache_for_seg[-4]], 1)) #112x112
        dec1 = self.decoder1(dec2) #224x224
        dec0 = F.relu(self.decoder0(dec1))
        seg = self.seg(dec0)

        return img_output, masked_output, seg

#############################################################################################################

###################################################### Dataset Class ###############################################

class CustomDatasetWithMask(Dataset):
    def __init__(self, filepaths, labels, train=True, inference=False):
        self.filepaths = filepaths
        self.labels = labels
        self.train = train
        self.inference = inference
        self.transform = self._get_transform()

    def _get_transform(self):
        if self.train:
            transform = transforms.Compose([
                #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=(-0.3, 0.3)),
                #transforms.RandomPosterize(bits=5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4605, 0.2799, 0.1762], std=[0.2111, 0.1551, 0.1120])
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4605, 0.2799, 0.1762], std=[0.2111, 0.1551, 0.1120])
            ])
        return transform

    def _load_mask(self, img_path, flip):
        mask_path = img_path.replace('/images/', '/annotations/')
        mask_path = mask_path.replace('img', 'ann')
        mask = Image.open(mask_path).convert('L')
        if flip:
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        return mask

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        label = self.labels[idx]

        flip_horizontal = flip_vertical = blur = rotate = False
        if self.train:
            flip_horizontal = random.random() > 0.5  
            flip_vertical = random.random() > 0.5    
            blur = random.random() > 0.5
            rotate = random.random() > 0.5

        img = img.transpose(Image.FLIP_LEFT_RIGHT) if flip_horizontal else img
        img = img.transpose(Image.FLIP_TOP_BOTTOM) if flip_vertical else img
        angle = random.randint(0, 180)
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False) if rotate else img
        img = (transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)))(img) if blur else img
        
        if not self.inference:
            mask = self._load_mask(img_path, flip_horizontal) if flip_horizontal else self._load_mask(img_path, False)
            mask = self._load_mask(img_path, flip_vertical) if flip_vertical else mask
            mask = mask.rotate(angle, resample=Image.BILINEAR, expand=False) if rotate else mask

        else: # during inference, create a dummy mask that is not going to be used
            mask = Image.new('L', (width, height), 0)
        
        img = self.transform(img)
        mask = transforms.ToTensor()(mask)

        if not self.inference and label == 0:
            mask = 1 - mask # required so that for non-bleeding images, the whole feature map is considered in the attention branch
        
        return img, mask, label, img_path

##############################################################################################################################################

def main(args):
    # Calculate mean and std
    root_dir = args.root_dir #'../datasets/WCEBleedGen'
    batch_size = args.batch_size

    filenames = []
    labels = []
    for class_name in ['bleeding', 'non-bleeding']:
        image_folder = 'images'
        class_folder = os.path.join(root_dir, class_name, image_folder)
        for filename in os.listdir(class_folder):
            filenames.append(os.path.join(class_folder, filename))
            labels.append(1 if class_name == 'bleeding' else 0)

    # Split data into train and validation sets
    train_filepaths, val_filepaths, train_labels, val_labels = train_test_split(
        filenames, labels, test_size=0.2, random_state=42) # random_state set to 42 for reproducability
    
    """# Calculate mean and std for normalization; We calcualted mean and std based on the WCE train set and used it in the dataset transform. See dataset class above.
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for filepath in train_filepaths:
        img = Image.open(filepath).convert('RGB')
        img_tensor = transforms.ToTensor()(img)
        mean += torch.mean(img_tensor, dim=(1, 2))
        std += torch.std(img_tensor, dim=(1, 2))

    mean /= len(train_filepaths)
    std /= len(train_filepaths)"""

    # Create datasets and data loaders
    train_dataset = CustomDatasetWithMask(train_filepaths, train_labels, train=True)
    val_dataset = CustomDatasetWithMask(val_filepaths, val_labels, train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, optimizer, and learning rate scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = CustomResNet18WithMask(num_classes=1).to(device)
    model2 = CustomVGGWithMask(num_classes=1).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    parameters = list(model1.parameters()) + list(model2.parameters()) 
    optimizer = optim.SGD(parameters, lr=0.01, momentum=0.9, weight_decay=1.00e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Training loop
    num_epochs = args.epochs
    save_dir = args.save_dir #'./checkpoints_custom'

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        model1.train()
        model2.train()
        
        total_loss = 0.0
        
        for i, (images, masks, labels, img_path) in enumerate(train_loader):
            images, masks, labels = images.to(device), masks.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs11, outputs12, outputs13 = model1(images, masks)
            outputs11 = outputs11.squeeze(dim=1)
            outputs12 = outputs12.squeeze(dim=1)
            outputs21, outputs22, outputs23 = model2(images, masks)
            outputs21 = outputs21.squeeze(dim=1)
            outputs22 = outputs22.squeeze(dim=1)
            
            for ii in range(len(labels)):
                if labels[ii] == 0:
                    masks[ii] = 1 - masks[ii] # masks of non-bleeding images were inverted in the dataset class for attention branch; 
                                              # but for segmentation branch, we need to invert it back.

            loss = criterion(outputs11, labels) + 0.2*criterion(outputs12, labels) + hybrid_seg_loss(outputs13, masks.to(torch.long)) 
            loss += criterion(outputs21, labels) + 0.2*criterion(outputs22, labels) + hybrid_seg_loss(outputs23, masks.to(torch.long))
            loss.backward()
            
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            val_labels = []
            val_preds = []

            val_loss = 0.0
            
            for i, (images, masks, labels, img_path) in enumerate(val_loader):
                images, masks, labels = images.to(device), masks.to(device), labels.float().to(device)
                outputs11, _, _ = model1(images, masks)
                outputs11 = outputs11.squeeze(dim=1)
                outputs21, _, _ = model2(images, masks)
                outputs21 = outputs21.squeeze(dim=1)
                val_loss += (criterion(outputs11, labels) + criterion(outputs21, labels)).item() 

                # Compute metrics
                probs1 = torch.sigmoid(outputs11)
                probs2 = torch.sigmoid(outputs21)
                avg_probs = (probs1 + probs2)/2 

                y_probs = avg_probs.detach().cpu().numpy()
                y_true = labels.cpu().numpy()

                best_threshold = 0.5
                y_pred = (avg_probs > best_threshold).float()
                y_pred = y_pred.cpu().numpy()
                val_labels.extend(y_true)
                val_preds.extend(y_pred)
            
            val_accuracy = accuracy_score(val_labels, val_preds)
            val_precision = precision_score(val_labels, val_preds, average='weighted')
            val_recall = recall_score(val_labels, val_preds, average='weighted')
            val_f1 = f1_score(val_labels, val_preds, average='weighted')

            # Save model every epoch
            save_dict = {'resnet18': model1.state_dict(),
                         'vgg': model2.state_dict()}
            torch.save(save_dict, os.path.join(save_dir, f'{epoch}_model.pth'))

            # Print epoch details
            tqdm.write(f"Epoch {epoch + 1}/{num_epochs}, "
                       f"Train Loss: {total_loss / len(train_loader):.4f}, "
                       f"Val Loss: {val_loss / len(val_loader):.4f}, "
                       f"Val Accuracy: {val_accuracy:.4f}, "
                       f"Val Prec: {val_precision:.4f}, "
                       f"Val Recall: {val_recall:.4f}, "
                       f"Val F1-score: {val_f1:.4f}")

        # LR Scheduler step
        lr_scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WCE ClassifyViStA')
    
    parser.add_argument('--root_dir', type=str, default='../datasets/WCEBleedGen', help='Data root directory path. Default is ../datasets/WCEBleedGen.')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_custom', help='Model save directory path. Default is ./checkpoints_custom. Model will be saved every epoch.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training. Default is 32.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training. Default is 100.')

    args = parser.parse_args()
    main(args)
