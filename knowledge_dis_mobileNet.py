import random
import os
import imageio
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from glob import glob
from IPython.display import Image
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
from models import build_models  # Assuming this contains MobileNetV4 implementation
import time
from thop import profile

# Specify GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device --> {device}")

# Set modality number (1 for single-modality RGB, 3 for tri-modal RGB+FLOW+MASK)
modal_num = 3  # Change to 1 for single-modality training/testing

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


# Knowledge Distillation Model
class VkD(nn.Module):
    def __init__(self, teacher, student, feature_dim=512, modal_num=3):
        super(VkD, self).__init__()
        self.teacher = teacher
        self.student = student
        self.modal_num = modal_num
        self.feature_dim = feature_dim
        # Feature adapters for each modality
        self.student_feature_adapters = nn.ModuleList([
            nn.Linear(1280, feature_dim) for _ in range(modal_num)
        ])
        self.proj = OrthogonalProjection(feature_dim, feature_dim)
        # Replace student's final classifier
        self.student.fc2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim * modal_num, student.fc2.out_features)
        )

    def forward(self, rgb, flow=None, mask=None, train_student=True):
        batch_size = rgb.size(0)
        # Teacher forward pass (always uses RGB)
        with torch.no_grad():
            teacher_features = self.teacher(rgb)
            teacher_features = teacher_features.view(batch_size, -1)
            teacher_features = (teacher_features - teacher_features.mean(dim=0)) / (teacher_features.std(dim=0) + 1e-6)

        # Student forward pass
        student_features = self.student(rgb, flow, mask)
        # Adapt features for each modality
        adapted_features = []
        for i, mod_features in enumerate(student_features):
            adapted = self.student_feature_adapters[i](mod_features)
            adapted = self.proj(adapted)
            adapted_features.append(adapted)
        combined_features = torch.cat(adapted_features, dim=1)
        logits = self.student.fc2(combined_features)

        if train_student:
            return logits, adapted_features, teacher_features
        else:
            return logits


# Custom feature loss
def feature_loss(student_features, teacher_features):
    total_loss = 0
    for s_features in student_features:
        total_loss += F.mse_loss(s_features, teacher_features)
    return total_loss / len(student_features)


# Modified ResNet18 for teacher model
class ResNet18(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNet18, self).__init__()
        base = models.resnet18(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(*list(base.children())[:-2])  # Up to conv5
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(batch_size, num_frames, -1)
        x = torch.mean(x, dim=1)  # Temporal average
        features = x  # Save features before classification
        return features  # Return (batch_size, 512) for distillation


# Multimodal Workout Classifier (MobileNetV4-based student model)
class MultimodalWorkoutClassifier(nn.Module):
    def __init__(self, num_classes=4, modal_num=3):
        super(MultimodalWorkoutClassifier, self).__init__()
        self.modal_num = modal_num
        self.rgb_model = build_models.MobileNetV4('MobileNetV4ConvSmall')
        self.rgb_model.fc = nn.Identity()
        if modal_num == 3:
            self.flow_model = build_models.MobileNetV4('MobileNetV4ConvSmall')
            self.mask_model = build_models.MobileNetV4('MobileNetV4ConvSmall')
            self.flow_model.fc = nn.Identity()
            self.mask_model.fc = nn.Identity()
            self.fc1 = nn.Linear(1280 * 3, 256)
        else:
            self.fc1 = nn.Linear(1280, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, rgb, flow=None, mask=None):
        batch_size, num_frames, c, h, w = rgb.shape
        # Process RGB
        rgb = rgb.view(-1, c, h, w)
        rgb_features = self.rgb_model(rgb)
        rgb_features = rgb_features.view(batch_size, num_frames, -1)
        rgb_features = torch.mean(rgb_features, dim=1)  # Temporal average
        features = [rgb_features]

        if self.modal_num == 3:
            # Process FLOW
            flow = flow.view(-1, c, h, w)
            flow_features = self.flow_model(flow)
            flow_features = flow_features.view(batch_size, num_frames, -1)
            flow_features = torch.mean(flow_features, dim=1)
            features.append(flow_features)
            # Process MASK
            mask = mask.view(-1, c, h, w)
            mask_features = self.mask_model(mask)
            mask_features = mask_features.view(batch_size, num_frames, -1)
            mask_features = torch.mean(mask_features, dim=1)
            features.append(mask_features)
            # Concatenate features
            combined_features = torch.cat((rgb_features, flow_features, mask_features), dim=1)
        else:
            combined_features = rgb_features

        out = self.fc1(combined_features)
        out = F.relu(out)
        return features  # Return modality features for distillation


# Training function for VkD
def train_vkd(model, train_loader, optimizer, epoch, device, modal_num):
    model.train()
    total_loss = 0
    for rgb, flow, mask, labels in tqdm(train_loader):
        rgb, flow, mask, labels = rgb.to(device), flow.to(device) if flow is not None else None, mask.to(
            device) if mask is not None else None, labels.to(device)
        optimizer.zero_grad()
        if modal_num == 3:
            logits, student_features, teacher_features = model(rgb, flow, mask)
        else:
            logits, student_features, teacher_features = model(rgb)
        cls_loss = F.cross_entropy(logits, labels)
        feat_loss = feature_loss(student_features, teacher_features)
        alpha = 0.5
        loss = cls_loss + alpha * feat_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch}: Train Loss: {avg_loss:.4f} (Cls: {cls_loss.item():.4f}, Feat: {feat_loss.item():.4f})')
    return avg_loss


# Training function for baseline (without distillation)
def train_baseline(model, train_loader, optimizer, epoch, device, modal_num):
    model.train()
    total_loss = 0
    for rgb, flow, mask, labels in tqdm(train_loader):
        rgb, flow, mask, labels = rgb.to(device), flow.to(device) if flow is not None else None, mask.to(
            device) if mask is not None else None, labels.to(device)
        optimizer.zero_grad()
        if modal_num == 3:
            outputs = model(rgb, flow, mask)
            combined_features = torch.cat(outputs, dim=1)
            combined_features = model.fc1(combined_features)  # Pass through fc1
            combined_features = F.relu(combined_features)  # Apply ReLU activation
            outputs = model.fc2(combined_features)  # Pass through fc2
        else:
            outputs = model(rgb)
            combined_features = outputs[0]  # Single modality
            combined_features = model.fc1(combined_features)  # Pass through fc1
            combined_features = F.relu(combined_features)  # Apply ReLU activation
            outputs = model.fc2(combined_features)  # Pass through fc2
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch}: Train Loss: {avg_loss:.4f}')
    return avg_loss


# Evaluation function
def test_model(model, test_loader, device, modal_num, is_vkd=False):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_labels = []
    test_predictions = []
    with torch.no_grad():
        for rgb, flow, mask, labels in test_loader:
            rgb, flow, mask, labels = rgb.to(device), flow.to(device) if flow is not None else None, mask.to(
                device) if mask is not None else None, labels.to(device)
            if is_vkd:
                if modal_num == 3:
                    outputs = model(rgb, flow, mask, train_student=False)
                else:
                    outputs = model(rgb, train_student=False)
            else:
                if modal_num == 3:
                    outputs = model(rgb, flow, mask)
                else:
                    outputs = model(rgb)
                combined_features = torch.cat(outputs, dim=1)
                combined_features = model.fc1(combined_features)  # Pass through fc1
                combined_features = F.relu(combined_features)  # Apply ReLU activation
                outputs = model.fc2(combined_features)  # Pass through fc2
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


# Utility functions
def denormalize_img(img, mean=0.5, std=0.5):
    img = (img * std) + mean
    img = (img * 255).astype("uint8")
    return img.clip(0, 255)


def create_gif(video_tensor, filename="sample.gif"):
    frames = []
    for video_frame in video_tensor:
        frame_denormalized = denormalize_img(video_frame.permute(1, 2, 0).cpu().numpy())
        frames.append(frame_denormalized)
    imageio.mimsave(filename, frames, "GIF", duration=0.25)
    return filename


def display_gif(video_tensor, gif_name="sample.gif"):
    gif_filename = create_gif(video_tensor, gif_name)
    return Image(filename=gif_filename)


def show_gifs(inputs_rgb, inputs_flow, inputs_mask, labels, predicted, index_to_class, modal_num):
    for j, rgb_tensor in enumerate(inputs_rgb):
        if j % 3 == 0:
            category_idx = labels[j].item()
            pred_category_idx = predicted[j].item()
            color = "green" if category_idx == pred_category_idx else "red"
            category_text = colored(f"True Category   -> {index_to_class[category_idx]}", color, attrs=['bold'])
            prediction_text = colored(f"Model Prediction -> {index_to_class[pred_category_idx]}", color, attrs=['bold'])
            separator = colored("=" * 50, 'light_grey', attrs=['bold'])
            print(separator)
            print(category_text)
            print(prediction_text)
            print("RGB Modality:")
            display_gif(rgb_tensor.cpu())
            if modal_num == 3:
                print("FLOW Modality:")
                display_gif(inputs_flow[j].cpu())
                print("MASK Modality:")
                display_gif(inputs_mask[j].cpu())


def plot_results(train_losses, val_losses, val_accuracies, val_f1_scores, val_precisions, val_recalls, modal_num,
                 suffix=''):
    suffix = f'mobilenetv4_single_{suffix}' if modal_num == 1 else f'mobilenetv4_multimodal_{suffix}'
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss', linewidth=2, color='#03346E')
    plt.plot(val_losses, label='Validation Loss', linewidth=2, color='#399918')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', linewidth=2, color='#399918')
    plt.title('Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(val_f1_scores, label='Validation F1 Score', linewidth=2, color='#FF6F61')
    plt.title('Validation F1 Score Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(val_precisions, label='Validation Precision', linewidth=2, color='#6B728E')
    plt.plot(val_recalls, label='Validation Recall', linewidth=2, color='#FFD93D')
    plt.title('Validation Precision and Recall Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'training_results_{suffix}.png', format='png', dpi=300)
    plt.close()


def plot_class_distribution(labels, title, modal_num):
    suffix = 'mobilenetv4_single' if modal_num == 1 else 'mobilenetv4_multimodal'
    class_counts = Counter(labels)
    classes = [str(k) for k in class_counts.keys()]
    counts = list(class_counts.values())
    sorted_indices = sorted(range(len(counts)), key=lambda i: counts[i], reverse=True)
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_classes, sorted_counts, color='#03346E')
    plt.xlabel('Number of Clips')
    plt.ylabel('Classes')
    plt.title(title)
    plt.xticks(rotation=90)
    plt.savefig(f'class_distribution_{suffix}.png', format='png', dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, normalize=True, figsize=(10, 8), cmap='Blues', suffix='',
                          modal_num=3):
    suffix = f'_{suffix}_mobilenetv4_single' if modal_num == 1 else f'_{suffix}_mobilenetv4_multimodal'
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.0%' if normalize else 'd', cmap=cmap, cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'Video Classification - Confusion Matrix {suffix}', fontsize=15)
    plt.savefig(f'confusion_matrix{suffix}.png', format='png', dpi=300)
    plt.close()


def plot_per_class_f1_scores(f1_scores, class_names, title, suffix='', modal_num=3):
    suffix = f'_{suffix}_mobilenetv4_single' if modal_num == 1 else f'_{suffix}_mobilenetv4_multimodal'
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, f1_scores, color='#03346E')
    plt.xlabel('Classes')
    plt.ylabel('F1 Score')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    for i, v in enumerate(f1_scores):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(f'per_class_f1_scores{suffix}.png', format='png', dpi=300)
    plt.close()


def get_metrics(test_losses, test_accuracies, actual_labels, model_predictions, class_names, modal_num):
    suffix = 'mobilenetv4_single' if modal_num == 1 else 'mobilenetv4_multimodal'
    test_mean_loss = np.mean(test_losses)
    test_accuracy = np.mean(test_accuracies)
    f1_macro = f1_score(actual_labels, model_predictions, average='macro')
    precision = precision_score(actual_labels, model_predictions, average='macro')
    recall = recall_score(actual_labels, model_predictions, average='macro')
    per_class_f1 = f1_score(actual_labels, model_predictions, average=None)
    color = 'blue'
    print(colored("=" * 75, 'dark_grey', attrs=['bold']))
    print(colored("Test Results:", color, attrs=['bold', 'underline']))
    print(colored(f"\tF1 Score (Macro): {f1_macro:.4f}", color, attrs=['bold']))
    print(colored(f"\tPrecision (Macro): {precision:.4f}", color, attrs=['bold']))
    print(colored(f"\tRecall (Macro): {recall:.4f}", color, attrs=['bold']))
    print(colored(f"\tMean Accuracy: {test_accuracy:.3f}%", color, attrs=['bold']))
    print(colored(f"\tMean Loss: {test_mean_loss:.3f}", color, attrs=['bold']))
    print(colored("Per-Class F1 Scores:", color, attrs=['bold']))
    for class_name, f1_score_value in zip(class_names, per_class_f1):
        print(colored(f"\tClass {class_name}: {f1_score_value:.4f}", color, attrs=['bold']))
    print(colored("=" * 75, 'dark_grey', attrs=['bold']))
    plot_per_class_f1_scores(per_class_f1, class_names, 'Per-Class F1 Scores - Test', suffix='test',
                             modal_num=modal_num)


# Data loading
val_split_size = 0.3
test_split_size = 0.3
height = 128
width = 128
img_size = (height, width)
num_frames = 30

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(img_size),
    transforms.ToTensor(),
])


def load_video_frames(video_path, label, modal_num, num_frames=30):
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
        self.num_videos = len(video_paths)

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
            rgb_frames = [np.zeros((height, width, 3))] * num_frames
            flow_frames = mask_frames = None if self.modal_num == 1 else [np.zeros((height, width, 3))] * num_frames

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


# Load and split data
all_video_paths = []
all_labels = []
for class_id in class_modality_folders:
    rgb_folder = class_modality_folders[class_id]['RGB']
    class_paths = glob(os.path.join(dataset_path, rgb_folder, '*.h5'), recursive=True)
    class_labels = [class_id] * len(class_paths)
    all_video_paths.extend(class_paths)
    all_labels.extend(class_labels)

num_classes = len(set(all_labels))
class_to_index = {str(l): idx for idx, l in enumerate(set(all_labels))}
index_to_class = {idx: str(l) for idx, l in enumerate(set(all_labels))}

trainval_videos, test_videos, trainval_labels, test_labels = train_test_split(
    all_video_paths, all_labels, test_size=test_split_size, random_state=42)
train_videos, val_videos, train_labels, val_labels = train_test_split(
    trainval_videos, trainval_labels, test_size=val_split_size, random_state=42)

create_csv_files = True
csv_dir = "/home/zhangnb/videoUAV/videoWork/myModelCSV"
if create_csv_files:
    os.makedirs(csv_dir, exist_ok=True)
    create_csv(train_videos, train_labels, os.path.join(csv_dir, "Multimodal_train1"))
    create_csv(val_videos, val_labels, os.path.join(csv_dir, "Multimodal_val1"))
    create_csv(test_videos, test_labels, os.path.join(csv_dir, "Multimodal_test1"))

train_df = pd.read_csv(os.path.join(csv_dir, "Multimodal_train1.csv"))
val_df = pd.read_csv(os.path.join(csv_dir, "Multimodal_val1.csv"))
test_df = pd.read_csv(os.path.join(csv_dir, "Multimodal_test1.csv"))

df = pd.concat([train_df, val_df, test_df], axis=0)
num_samples = df.shape[0]
class_counts = df.label.value_counts().to_dict()
plot_class_distribution(class_counts, 'Multimodal Workout Video Dataset - Classes Distribution', modal_num)

class_weights = {cat: num_samples / count for cat, count in class_counts.items()}
all_weights = [class_weights[label] for label in train_df['label'].values]
n_samples = len(all_weights)
sampler = WeightedRandomSampler(weights=all_weights, num_samples=n_samples, replacement=False)

train_dataset = MultimodalWorkoutDataset(train_df['path'].values, train_df['label'].values, transform=transform,
                                         modal_num=modal_num)
val_dataset = MultimodalWorkoutDataset(val_df['path'].values, val_df['label'].values, transform=transform,
                                       modal_num=modal_num)
test_dataset = MultimodalWorkoutDataset(test_df['path'].values, test_df['label'].values, transform=transform,
                                        modal_num=modal_num)

balanced_train = False
if balanced_train:
    train_loader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=50, sampler=sampler)
else:
    train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=20, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

# Initialize models
teacher = ResNet18(num_classes=num_classes).to(device)
student_vkd = MultimodalWorkoutClassifier(num_classes=num_classes, modal_num=modal_num).to(device)
student_baseline = MultimodalWorkoutClassifier(num_classes=num_classes, modal_num=modal_num).to(device)
teacher.eval()

# Initialize VkD model
model_vkd = VkD(teacher, student_vkd, feature_dim=512, modal_num=modal_num).to(device)

# Optimizers
optimizer_vkd = optim.Adam(
    list(model_vkd.student.parameters()) +
    list(model_vkd.student_feature_adapters.parameters()) +
    list(model_vkd.proj.parameters()),
    lr=0.0001, weight_decay=5e-4
)
optimizer_baseline = optim.Adam(student_baseline.parameters(), lr=0.0001, weight_decay=5e-4)
scheduler_vkd = optim.lr_scheduler.StepLR(optimizer_vkd, step_size=10, gamma=0.1)
scheduler_baseline = optim.lr_scheduler.StepLR(optimizer_baseline, step_size=10, gamma=0.1)

# Training loop
num_epochs = 30
vkd_train_losses, vkd_val_losses, vkd_val_accuracies, vkd_val_f1_scores, vkd_val_precisions, vkd_val_recalls = [], [], [], [], [], []
baseline_train_losses, baseline_val_losses, baseline_val_accuracies, baseline_val_f1_scores, baseline_val_precisions, baseline_val_recalls = [], [], [], [], [], []
max_gained_acc_vkd, max_gained_acc_baseline = 0, 0
epoch_of_max_acc_vkd, epoch_of_max_acc_baseline = 0, 0
early_stopping = False
epochs_tolerance = 5

for epoch in range(num_epochs):
    print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
    # Train VkD
    print("Training VkD (with distillation):")
    vkd_loss = train_vkd(model_vkd, train_loader, optimizer_vkd, epoch + 1, device, modal_num)
    vkd_train_losses.append(vkd_loss)

    # Train baseline
    print("Training MobileNetV4 (without distillation):")
    baseline_loss = train_baseline(student_baseline, train_loader, optimizer_baseline, epoch + 1, device, modal_num)
    baseline_train_losses.append(baseline_loss)

    # Evaluate VkD
    print("Evaluating VkD:")
    vkd_test_loss, vkd_accuracy, vkd_labels, vkd_predictions = test_model(model_vkd, val_loader, device, modal_num,
                                                                          is_vkd=True)
    vkd_val_losses.append(vkd_test_loss)
    vkd_val_accuracies.append(vkd_accuracy)
    vkd_f1 = f1_score(vkd_labels, vkd_predictions, average='macro')
    vkd_precision = precision_score(vkd_labels, vkd_predictions, average='macro')
    vkd_recall = recall_score(vkd_labels, vkd_predictions, average='macro')
    vkd_val_f1_scores.append(vkd_f1)
    vkd_val_precisions.append(vkd_precision)
    vkd_val_recalls.append(vkd_recall)
    plot_confusion_matrix(vkd_labels, vkd_predictions, list(class_to_index.keys()), suffix=f'vkd_epoch_{epoch + 1}',
                          modal_num=modal_num)
    plot_per_class_f1_scores(f1_score(vkd_labels, vkd_predictions, average=None), list(class_to_index.keys()),
                             f'Per-Class F1 Scores VkD - Epoch {epoch + 1}', suffix=f'vkd_epoch_{epoch + 1}',
                             modal_num=modal_num)

    # Evaluate baseline
    print("Evaluating MobileNetV4:")
    baseline_test_loss, baseline_accuracy, baseline_labels, baseline_predictions = test_model(student_baseline,
                                                                                              val_loader, device,
                                                                                              modal_num, is_vkd=False)
    baseline_val_losses.append(baseline_test_loss)
    baseline_val_accuracies.append(baseline_accuracy)
    baseline_f1 = f1_score(baseline_labels, baseline_predictions, average='macro')
    baseline_precision = precision_score(baseline_labels, baseline_predictions, average='macro')
    baseline_recall = recall_score(baseline_labels, baseline_predictions, average='macro')
    baseline_val_f1_scores.append(baseline_f1)
    baseline_val_precisions.append(baseline_precision)
    baseline_val_recalls.append(baseline_recall)
    plot_confusion_matrix(baseline_labels, baseline_predictions, list(class_to_index.keys()),
                          suffix=f'baseline_epoch_{epoch + 1}', modal_num=modal_num)
    plot_per_class_f1_scores(f1_score(baseline_labels, baseline_predictions, average=None), list(class_to_index.keys()),
                             f'Per-Class F1 Scores Baseline - Epoch {epoch + 1}', suffix=f'baseline_epoch_{epoch + 1}',
                             modal_num=modal_num)

    # Update max accuracies
    if vkd_accuracy >= max_gained_acc_vkd:
        max_gained_acc_vkd = vkd_accuracy
        epoch_of_max_acc_vkd = epoch
    if baseline_accuracy >= max_gained_acc_baseline:
        max_gained_acc_baseline = baseline_accuracy
        epoch_of_max_acc_baseline = epoch

    # Early stopping
    if early_stopping and (epoch - max(epoch_of_max_acc_vkd, epoch_of_max_acc_baseline) > epochs_tolerance):
        print("Early stopping triggered")
        break

    scheduler_vkd.step()
    scheduler_baseline.step()

# Plot results
plot_results(vkd_train_losses, vkd_val_losses, vkd_val_accuracies, vkd_val_f1_scores, vkd_val_precisions,
             vkd_val_recalls, modal_num, suffix='vkd')
plot_results(baseline_train_losses, baseline_val_losses, baseline_val_accuracies, baseline_val_f1_scores,
             baseline_val_precisions, baseline_val_recalls, modal_num, suffix='baseline')

# Create accuracy comparison chart
# chartjs_code = f"""
# {{
#     "type": "line",
#     "data": {{
#         "labels": {list(range(1, len(vkd_val_accuracies) + 1))},
#         "datasets": [
#             {{
#                 "label": "VkD (with KD) Accuracy",
#                 "data": {vkd_val_accuracies},
#                 "borderColor": "#007bff",
#                 "backgroundColor": "rgba(0, 123, 255, 0.1)",
#                 "fill": true
#             }},
#             {{
#                 "label": "MobileNetV4 (without KD) Accuracy",
#                 "data": {baseline_val_accuracies},
#                 "borderColor": "#dc3545",
#                 "backgroundColor": "rgba(220, 53, 69, 0.1)",
#                 "fill": true
#             }}
#         ]
#     }},
#     "options": {{
#         "responsive": true,
#         "plugins": {{
#             "legend": {{"position": "top"}},
#             "title": {{"display": true, "text": "Accuracy Comparison Over Epochs"}}
#         }},
#         "scales": {{
#             "x": {{"title": {{"display": true, "text": "Epoch"}}}},
#             "y": {{"title": {{"display": true, "text": "Accuracy (%)"}}, "min": 0, "max": 100}}
#         }}
#     }}
# }}
# """
# print("ChartJS Code for Accuracy Comparison:")
# print(chartjs_code)