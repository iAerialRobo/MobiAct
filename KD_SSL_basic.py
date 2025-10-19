import random
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
from collections import Counter
import termcolor
from termcolor import colored
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import build_models

# Setup
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device --> {device}")

modal_num = 3  # Change to 1 for single-modality

# Data paths and class-modality mappings
dataset_path = '/home/zhangnb/videoUAV/data/hdf5/videoDataIntegrate/middle'
class_modality_folders = {
    '0': {'RGB': 'MAV_inv_vShapeRGB', 'FLOW': 'MAV_inv_vShapeFLOW', 'MASK': 'MAV_inv_vShapeMASK'},
    '1': {'RGB': 'MAV_left_rightRGB', 'FLOW': 'MAV_left_rightFLOW', 'MASK': 'MAV_left_rightMASK'},
    '2': {'RGB': 'MAV_up_downRGB', 'FLOW': 'MAV_up_downFLOW', 'MASK': 'MAV_up_downMASK'},
    '3': {'RGB': 'MAV_vShapeRGB', 'FLOW': 'MAV_vShapeFLOW', 'MASK': 'MAV_vShapeMASK'}
}


# Orthogonal Projection Layer
class OrthogonalProjection(nn.Module):
    def __init__(self, in_features, out_features):
        super(OrthogonalProjection, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.A = nn.Parameter(torch.randn(in_features, in_features) * 0.01)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if x.size(-1) != self.in_features:
            raise ValueError(f"Expected input dimension {self.in_features}, got {x.size(-1)}")
        A_skew = 0.5 * (self.A - self.A.transpose(-1, -2))
        I = torch.eye(self.in_features, device=x.device)
        A2 = torch.matmul(A_skew, A_skew)
        exp_A = I + A_skew + 0.5 * A2
        proj = exp_A[:, :self.out_features]
        return torch.matmul(x, proj)


# Contrastive Learning Head
class ContrastiveHead(nn.Module):
    def __init__(self, input_dim, projection_dim=128):
        super(ContrastiveHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, projection_dim)
        )

    def forward(self, x):
        return F.normalize(self.projection(x), dim=1)


# Modified ResNet18 Teacher
class ResNet18(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNet18, self).__init__()
        base = models.resnet18(weights='IMAGENET1K_V1')
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, return_features=False):
        batch_size, num_frames, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        features = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x.mean([2, 3]).view(batch_size, num_frames, -1).mean(dim=1))
        x = self.maxpool(x)

        x = self.layer1(x)
        features.append(x.mean([2, 3]).view(batch_size, num_frames, -1).mean(dim=1))

        x = self.layer2(x)
        features.append(x.mean([2, 3]).view(batch_size, num_frames, -1).mean(dim=1))

        x = self.layer3(x)
        features.append(x.mean([2, 3]).view(batch_size, num_frames, -1).mean(dim=1))

        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(batch_size, num_frames, -1).mean(dim=1)
        features.append(x)

        if return_features:
            return features
        x = self.classifier(x)
        return x


# Modified Student Network
class MultimodalWorkoutClassifier(nn.Module):
    def __init__(self, num_classes=4, modal_num=3):
        super(MultimodalWorkoutClassifier, self).__init__()
        self.modal_num = modal_num

        # RGB model
        base = build_models.MobileNetV4('MobileNetV4ConvSmall')
        self.layers = base
        self.stage1 = self.layers.conv0
        self.stage2 = self.layers.layer1
        self.stage3 = self.layers.layer2
        self.stage4 = self.layers.layer3
        self.stage5 = nn.Sequential(self.layers.layer4, self.layers.layer5)

        # Additional modalities
        if modal_num == 3:
            self.flow_model = build_models.MobileNetV4('MobileNetV4ConvSmall')
            self.mask_model = build_models.MobileNetV4('MobileNetV4ConvSmall')
            self.flow_stage1 = self.flow_model.conv0
            self.flow_stage2 = self.flow_model.layer1
            self.flow_stage3 = self.flow_model.layer2
            self.flow_stage4 = self.flow_model.layer3
            self.flow_stage5 = nn.Sequential(self.flow_model.layer4, self.flow_model.layer5)
            self.mask_stage1 = self.mask_model.conv0
            self.mask_stage2 = self.mask_model.layer1
            self.mask_stage3 = self.mask_model.layer2
            self.mask_stage4 = self.mask_model.layer3
            self.mask_stage5 = nn.Sequential(self.mask_model.layer4, self.mask_model.layer5)

        self.channel_counts = [32, 32, 64, 96, 1280]
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.channel_counts[-1] * modal_num, num_classes)
        )

        # Contrastive learning head
        self.contrastive_head = ContrastiveHead(self.channel_counts[-1] * modal_num, 128)

    def forward(self, rgb, flow=None, mask=None, return_features=False, return_contrastive=False):
        batch_size, num_frames, c, h, w = rgb.shape
        rgb = rgb.view(-1, c, h, w)

        # Process RGB
        features_rgb = []
        x = self.stage1(rgb)
        x = x.view(batch_size, num_frames, x.size(1), x.size(2), x.size(3))
        feat = x.mean([3, 4]).mean(dim=1)
        features_rgb.append(feat)

        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.stage2(x)
        x = x.view(batch_size, num_frames, x.size(1), x.size(2), x.size(3))
        feat = x.mean([3, 4]).mean(dim=1)
        features_rgb.append(feat)

        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.stage3(x)
        x = x.view(batch_size, num_frames, x.size(1), x.size(2), x.size(3))
        feat = x.mean([3, 4]).mean(dim=1)
        features_rgb.append(feat)

        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.stage4(x)
        x = x.view(batch_size, num_frames, x.size(1), x.size(2), x.size(3))
        feat = x.mean([3, 4]).mean(dim=1)
        features_rgb.append(feat)

        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.stage5(x)
        x = x.view(batch_size, num_frames, x.size(1), x.size(2), x.size(3))
        rgb_features = x.mean([3, 4]).mean(dim=1)
        features_rgb.append(rgb_features)

        all_features = [features_rgb]

        if self.modal_num == 3:
            flow = flow.view(-1, c, h, w)
            mask = mask.view(-1, c, h, w)

            # Process FLOW
            features_flow = []
            x_flow = self.flow_stage1(flow)
            x_flow = x_flow.view(batch_size, num_frames, x_flow.size(1), x_flow.size(2), x_flow.size(3))
            feat = x_flow.mean([3, 4]).mean(dim=1)
            features_flow.append(feat)

            x_flow = x_flow.view(-1, x_flow.size(2), x_flow.size(3), x_flow.size(4))
            x_flow = self.flow_stage2(x_flow)
            x_flow = x_flow.view(batch_size, num_frames, x_flow.size(1), x_flow.size(2), x_flow.size(3))
            feat = x_flow.mean([3, 4]).mean(dim=1)
            features_flow.append(feat)

            x_flow = x_flow.view(-1, x_flow.size(2), x_flow.size(3), x_flow.size(4))
            x_flow = self.flow_stage3(x_flow)
            x_flow = x_flow.view(batch_size, num_frames, x_flow.size(1), x_flow.size(2), x_flow.size(3))
            feat = x_flow.mean([3, 4]).mean(dim=1)
            features_flow.append(feat)

            x_flow = x_flow.view(-1, x_flow.size(2), x_flow.size(3), x_flow.size(4))
            x_flow = self.flow_stage4(x_flow)
            x_flow = x_flow.view(batch_size, num_frames, x_flow.size(1), x_flow.size(2), x_flow.size(3))
            feat = x_flow.mean([3, 4]).mean(dim=1)
            features_flow.append(feat)

            x_flow = x_flow.view(-1, x_flow.size(2), x_flow.size(3), x_flow.size(4))
            x_flow = self.flow_stage5(x_flow)
            x_flow = x_flow.view(batch_size, num_frames, x_flow.size(1), x_flow.size(2), x_flow.size(3))
            flow_features = x_flow.mean([3, 4]).mean(dim=1)
            features_flow.append(flow_features)

            # Process MASK
            features_mask = []
            x_mask = self.mask_stage1(mask)
            x_mask = x_mask.view(batch_size, num_frames, x_mask.size(1), x_mask.size(2), x_mask.size(3))
            feat = x_mask.mean([3, 4]).mean(dim=1)
            features_mask.append(feat)

            x_mask = x_mask.view(-1, x_mask.size(2), x_mask.size(3), x_mask.size(4))
            x_mask = self.mask_stage2(x_mask)
            x_mask = x_mask.view(batch_size, num_frames, x_mask.size(1), x_mask.size(2), x_mask.size(3))
            feat = x_mask.mean([3, 4]).mean(dim=1)
            features_mask.append(feat)

            x_mask = x_mask.view(-1, x_mask.size(2), x_mask.size(3), x_mask.size(4))
            x_mask = self.mask_stage3(x_mask)
            x_mask = x_mask.view(batch_size, num_frames, x_mask.size(1), x_mask.size(2), x_mask.size(3))
            feat = x_mask.mean([3, 4]).mean(dim=1)
            features_mask.append(feat)

            x_mask = x_mask.view(-1, x_mask.size(2), x_mask.size(3), x_mask.size(4))
            x_mask = self.mask_stage4(x_mask)
            x_mask = x_mask.view(batch_size, num_frames, x_mask.size(1), x_mask.size(2), x_mask.size(3))
            feat = x_mask.mean([3, 4]).mean(dim=1)
            features_mask.append(feat)

            x_mask = x_mask.view(-1, x_mask.size(2), x_mask.size(3), x_mask.size(4))
            x_mask = self.mask_stage5(x_mask)
            x_mask = x_mask.view(batch_size, num_frames, x_mask.size(1), x_mask.size(2), x_mask.size(3))
            mask_features = x_mask.mean([3, 4]).mean(dim=1)
            features_mask.append(mask_features)

            all_features.extend([features_flow, features_mask])

        if return_features:
            return all_features

        combined_features = torch.cat([f[-1] for f in all_features], dim=1)

        if return_contrastive:
            contrastive_features = self.contrastive_head(combined_features)
            return combined_features, contrastive_features

        logits = self.classifier(combined_features)
        return logits


# LayerWiseVkD with Contrastive Learning
class LayerWiseVkDContrastive(nn.Module):
    def __init__(self, teacher, student, feature_dims=[64, 64, 128, 256, 512], modal_num=3):
        super(LayerWiseVkDContrastive, self).__init__()
        self.teacher = teacher
        self.student = student
        self.modal_num = modal_num

        student_channel_counts = [32, 32, 64, 96, 1280]

        self.feature_adapters = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(student_channel_counts[stage_idx], feature_dims[stage_idx])
                for mod_idx in range(modal_num)
            ])
            for stage_idx in range(len(feature_dims))
        ])

        self.projections = nn.ModuleList([
            OrthogonalProjection(feature_dims[stage_idx], feature_dims[stage_idx])
            for stage_idx in range(len(feature_dims))
        ])

        self.student.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dims[-1] * modal_num, student.classifier[1].out_features)
        )

    def forward(self, rgb, flow=None, mask=None, train_student=True, return_contrastive=False):
        batch_size = rgb.size(0)

        with torch.no_grad():
            teacher_features = self.teacher(rgb, return_features=True)
            for i in range(len(teacher_features)):
                teacher_features[i] = (teacher_features[i] - teacher_features[i].mean(dim=0)) / (
                            teacher_features[i].std(dim=0) + 1e-6)

        if train_student:
            student_features = self.student(rgb, flow, mask, return_features=True)
            adapted_features = []

            for mod_idx in range(self.modal_num):
                mod_adapted = []
                for stage_idx in range(5):
                    feat = student_features[mod_idx][stage_idx]
                    feat = self.feature_adapters[stage_idx][mod_idx](feat)
                    feat = self.projections[stage_idx](feat)
                    mod_adapted.append(feat)
                adapted_features.append(mod_adapted)

            combined_features = torch.cat([f[-1] for f in adapted_features], dim=1)
            logits = self.student.classifier(combined_features)

            if return_contrastive:
                contrastive_features = self.student.contrastive_head(combined_features)
                return logits, adapted_features, teacher_features, contrastive_features

            return logits, adapted_features, teacher_features
        else:
            student_features = self.student(rgb, flow, mask, return_features=True)
            adapted_features = []

            for mod_idx in range(self.modal_num):
                feat = student_features[mod_idx][-1]
                feat = self.feature_adapters[-1][mod_idx](feat)
                feat = self.projections[-1](feat)
                adapted_features.append(feat)

            combined_features = torch.cat(adapted_features, dim=1)
            logits = self.student.classifier(combined_features)
            return logits


# Contrastive Loss Function
def contrastive_loss(features, labels, temperature=0.07):
    """
    Supervised contrastive loss
    """
    batch_size = features.shape[0]
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(features.device)

    # Compute similarity matrix
    anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temperature)
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # Mask-out self-contrast cases
    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(features.device), 0)
    mask = mask * logits_mask

    # Compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

    # Compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

    # Loss
    loss = -mean_log_prob_pos.mean()
    return loss


# Loss functions
def feature_loss(student_features, teacher_features):
    total_loss = 0
    for mod_features in student_features:
        for s, t in zip(mod_features, teacher_features):
            total_loss += F.mse_loss(s, t)
    return total_loss / (len(student_features) * len(teacher_features))


# Training function
def train_vkd_contrastive(model, train_loader, optimizer, epoch, device, modal_num, alpha=0.5, beta=0.3):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_feat_loss = 0
    total_contrastive_loss = 0

    for rgb, flow, mask, labels in tqdm(train_loader):
        rgb = rgb.to(device)
        flow = flow.to(device) if flow is not None else None
        mask = mask.to(device) if mask is not None else None
        labels = labels.to(device)

        optimizer.zero_grad()

        if modal_num == 3:
            logits, student_features, teacher_features, contrastive_features = model(
                rgb, flow, mask, return_contrastive=True)
        else:
            logits, student_features, teacher_features, contrastive_features = model(
                rgb, return_contrastive=True)

        # Classification loss
        cls_loss = F.cross_entropy(logits, labels)

        # Feature distillation loss
        feat_loss = feature_loss(student_features, teacher_features)

        # Contrastive loss
        contrast_loss = contrastive_loss(contrastive_features, labels)

        # Combined loss
        loss = cls_loss + alpha * feat_loss + beta * contrast_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_feat_loss += feat_loss.item()
        total_contrastive_loss += contrast_loss.item()

    avg_loss = total_loss / len(train_loader)
    avg_cls_loss = total_cls_loss / len(train_loader)
    avg_feat_loss = total_feat_loss / len(train_loader)
    avg_contrast_loss = total_contrastive_loss / len(train_loader)

    print(
        f'Epoch {epoch}: Train Loss: {avg_loss:.4f} (Cls: {avg_cls_loss:.4f}, Feat: {avg_feat_loss:.4f}, Contrast: {avg_contrast_loss:.4f})')
    return avg_loss


# Evaluation function
def test_model(model, test_loader, device, modal_num):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_labels = []
    test_predictions = []

    with torch.no_grad():
        for rgb, flow, mask, labels in test_loader:
            rgb = rgb.to(device)
            flow = flow.to(device) if flow is not None else None
            mask = mask.to(device) if mask is not None else None
            labels = labels.to(device)

            if modal_num == 3:
                outputs = model(rgb, flow, mask, train_student=False)
            else:
                outputs = model(rgb, train_student=False)

            loss = F.cross_entropy(outputs, labels, reduction='sum')
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_labels.extend(labels.cpu().numpy())
            test_predictions.extend(predicted.cpu().numpy())

    test_loss /= total
    accuracy = 100 * correct / total
    f1 = f1_score(test_labels, test_predictions, average='macro')
    precision = precision_score(test_labels, test_predictions, average='macro')
    recall = recall_score(test_labels, test_predictions, average='macro')

    print(
        f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    return test_loss, accuracy, test_labels, test_predictions


# Data loading functions
def load_video_frames(video_path, label, modal_num, num_frames=30):
    height, width = 128, 128
    label_str = str(label)
    modalities = ['RGB'] if modal_num == 1 else ['RGB', 'FLOW', 'MASK']
    frames = {mod: [] for mod in modalities}

    for mod in modalities:
        mod_folder = class_modality_folders[label_str][mod]
        mod_path = video_path.replace(class_modality_folders[label_str]['RGB'], mod_folder)
        mod_path = mod_path.replace('_RGB.', f'_{mod}.')

        try:
            with h5py.File(mod_path, 'r') as hdf:
                for i in range(num_frames):
                    array = np.array(hdf[f'array_{i}'])
                    frames[mod].append(array)
        except Exception as e:
            print(f"Error loading {mod} from {mod_path}: {e}")
            frames[mod] = [np.zeros((height, width, 3))] * num_frames

    if modal_num == 1:
        return frames['RGB'], None, None
    return frames['RGB'], frames['FLOW'], frames['MASK']


class MultimodalWorkoutDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None, modal_num=3):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.modal_num = modal_num
        self.height, self.width = 128, 128
        self.num_frames = 30

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label_name = str(self.labels[idx])
        label_index = class_to_index[label_name]

        try:
            rgb_frames, flow_frames, mask_frames = load_video_frames(video_path, label_name, self.modal_num)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            rgb_frames = [np.zeros((self.height, self.width, 3))] * self.num_frames
            flow_frames = mask_frames = None if self.modal_num == 1 else [np.zeros(
                (self.height, self.width, 3))] * self.num_frames

        rgb_frames = [torch.tensor(frame).permute(2, 0, 1).to(device) for frame in rgb_frames]
        if self.modal_num == 3:
            flow_frames = [torch.tensor(frame).permute(2, 0, 1).to(device) for frame in flow_frames]
            mask_frames = [torch.tensor(frame).permute(2, 0, 1).to(device) for frame in mask_frames]

        if self.transform:
            rgb_frames = [self.transform(frame) for frame in rgb_frames]
            if self.modal_num == 3:
                flow_frames = [self.transform(frame) for frame in flow_frames]
                mask_frames = [self.transform(frame) for frame in mask_frames]

        rgb_tensor = torch.stack(rgb_frames)
        flow_tensor = torch.stack(flow_frames) if self.modal_num == 3 else None
        mask_tensor = torch.stack(mask_frames) if self.modal_num == 3 else None
        label_tensor = torch.tensor(label_index, dtype=torch.long).to(device)

        return rgb_tensor, flow_tensor, mask_tensor, label_tensor


def create_csv(videos, labels, file_name=""):
    df = pd.DataFrame({'path': videos, 'label': labels})
    df.to_csv(f"{file_name}.csv", index=False)


def get_metrics(test_losses, test_accuracies, actual_labels, model_predictions, class_names):
    test_mean_loss = np.mean(test_losses)
    test_accuracy = np.mean(test_accuracies)
    f1_macro = f1_score(actual_labels, model_predictions, average='macro')
    precision = precision_score(actual_labels, model_predictions, average='macro')
    recall = recall_score(actual_labels, model_predictions, average='macro')
    per_class_f1 = f1_score(actual_labels, model_predictions, average=None)

    print(colored("=" * 75, 'dark_grey', attrs=['bold']))
    print(colored("Test Results:", 'blue', attrs=['bold', 'underline']))
    print(colored(f"\tF1 Score (Macro): {f1_macro:.4f}", 'blue', attrs=['bold']))
    print(colored(f"\tPrecision (Macro): {precision:.4f}", 'blue', attrs=['bold']))
    print(colored(f"\tRecall (Macro): {recall:.4f}", 'blue', attrs=['bold']))
    print(colored(f"\tMean Accuracy: {test_accuracy:.3f}%", 'blue', attrs=['bold']))
    print(colored(f"\tMean Loss: {test_mean_loss:.3f}", 'blue', attrs=['bold']))
    print(colored("Per-Class F1 Scores:", 'blue', attrs=['bold']))
    for class_name, f1_score_value in zip(class_names, per_class_f1):
        print(colored(f"\tClass {class_name}: {f1_score_value:.4f}", 'blue', attrs=['bold']))
    print(colored("=" * 75, 'dark_grey', attrs=['bold']))


# Main execution
if __name__ == "__main__":
    # Data preparation
    val_split_size = 0.2
    test_split_size = 0.2
    batch_size = 8
    num_epochs = 50
    learning_rate = 0.001

    # Class mappings
    class_names = ['inv_vShape', 'left_right', 'up_down', 'vShape']
    class_to_index = {str(i): i for i in range(len(class_names))}
    index_to_class = {i: name for i, name in enumerate(class_names)}

    print(colored("Starting Multimodal Workout Classification with Knowledge Distillation", 'green', attrs=['bold']))
    print(f"Modal number: {modal_num}")
    print(f"Device: {device}")

    # Load data paths and labels
    all_video_paths = []
    all_labels = []

    for class_label, modality_folders in class_modality_folders.items():
        rgb_folder = modality_folders['RGB']
        class_path = os.path.join(dataset_path, rgb_folder)

        if os.path.exists(class_path):
            video_files = glob(os.path.join(class_path, "*_RGB.h5"))
            print(f"Found {len(video_files)} videos for class {class_label} ({class_names[int(class_label)]})")

            all_video_paths.extend(video_files)
            all_labels.extend([class_label] * len(video_files))
        else:
            print(f"Warning: Path {class_path} does not exist")

    print(f"Total videos found: {len(all_video_paths)}")
    print(f"Label distribution: {Counter(all_labels)}")

    # Split data
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_video_paths, all_labels, test_size=(val_split_size + test_split_size),
        random_state=42, stratify=all_labels
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=test_split_size / (val_split_size + test_split_size),
        random_state=42, stratify=temp_labels
    )

    print(f"Train samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    print(f"Test samples: {len(test_paths)}")

    # Create CSV files for reference
    create_csv(train_paths, train_labels, "train_data")
    create_csv(val_paths, val_labels, "val_data")
    create_csv(test_paths, test_labels, "test_data")

    # Data transforms
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = MultimodalWorkoutDataset(train_paths, train_labels, transform, modal_num)
    val_dataset = MultimodalWorkoutDataset(val_paths, val_labels, transform, modal_num)
    test_dataset = MultimodalWorkoutDataset(test_paths, test_labels, transform, modal_num)

    # Calculate class weights for balanced training
    train_label_indices = [class_to_index[label] for label in train_labels]
    class_counts = Counter(train_label_indices)
    total_samples = len(train_label_indices)
    class_weights = [total_samples / (len(class_counts) * class_counts[i]) for i in range(len(class_names))]

    sample_weights = [class_weights[label] for label in train_label_indices]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Class weights: {class_weights}")

    # Initialize models
    print("Initializing models...")
    teacher = ResNet18(num_classes=len(class_names)).to(device)
    student = MultimodalWorkoutClassifier(num_classes=len(class_names), modal_num=modal_num).to(device)

    # Create the combined model for knowledge distillation
    vkd_model = LayerWiseVkDContrastive(teacher, student, modal_num=modal_num).to(device)

    # Optimizer
    optimizer = optim.Adam(vkd_model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Training history
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_accuracy = 0
    best_model_state = None
    patience_counter = 0
    early_stopping_patience = 10

    print("Starting training...")
    print(colored("=" * 75, 'green', attrs=['bold']))

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 50)

        # Training
        train_loss = train_vkd_contrastive(
            vkd_model, train_loader, optimizer, epoch, device, modal_num,
            alpha=0.5, beta=0.3
        )
        train_losses.append(train_loss)

        # Validation
        val_loss, val_accuracy, val_labels_list, val_predictions = test_model(
            vkd_model, val_loader, device, modal_num
        )
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = vkd_model.state_dict().copy()
            patience_counter = 0
            print(colored(f"New best validation accuracy: {val_accuracy:.2f}%", 'green', attrs=['bold']))

            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': val_loss
            }, f'best_vkd_contrastive_model_modal{modal_num}.pth')
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(colored(f"Early stopping triggered after {epoch} epochs", 'yellow', attrs=['bold']))
            break

        print(f"Validation Accuracy: {val_accuracy:.2f}% (Best: {best_val_accuracy:.2f}%)")
        print(f"Patience: {patience_counter}/{early_stopping_patience}")

    # Load best model for testing
    print(colored("\nLoading best model for final evaluation...", 'blue', attrs=['bold']))
    vkd_model.load_state_dict(best_model_state)

    # Final testing
    print(colored("Final Testing:", 'red', attrs=['bold']))
    test_loss, test_accuracy, test_labels_list, test_predictions = test_model(
        vkd_model, test_loader, device, modal_num
    )

    # Calculate and display comprehensive metrics
    get_metrics([test_loss], [test_accuracy], test_labels_list, test_predictions, class_names)

    # Plot training history
    plt.figure(figsize=(15, 5))

    # Plot losses
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Plot validation accuracy
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot confusion matrix
    plt.subplot(1, 3, 3)
    cm = confusion_matrix(test_labels_list, test_predictions)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations to confusion matrix
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    # Save the plot
    plt.savefig(f'training_results_modal{modal_num}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save final results
    results = {
        'best_val_accuracy': best_val_accuracy,
        'test_accuracy': test_accuracy,
        'test_loss': test_loss,
        'f1_macro': f1_score(test_labels_list, test_predictions, average='macro'),
        'precision_macro': precision_score(test_labels_list, test_predictions, average='macro'),
        'recall_macro': recall_score(test_labels_list, test_predictions, average='macro'),
        'modal_num': modal_num,
        'num_epochs_trained': epoch
    }

    # Save results to file
    results_df = pd.DataFrame([results])
    results_df.to_csv(f'final_results_modal{modal_num}.csv', index=False)

    print(colored("\nTraining completed successfully!", 'green', attrs=['bold']))
    print(colored(f"Best model saved as: best_vkd_contrastive_model_modal{modal_num}.pth", 'green'))
    print(colored(f"Results saved as: final_results_modal{modal_num}.csv", 'green'))
    print(colored(f"Training plots saved as: training_results_modal{modal_num}.png", 'green'))