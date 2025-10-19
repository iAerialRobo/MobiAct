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
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
from models import build_models  # Assuming this contains MobileNetV4 implementation
import time
from thop import profile  # For model complexity calculation

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


def plot_results(train_losses, val_losses, val_accuracies, val_f1_scores, val_precisions, val_recalls, modal_num):
    suffix = 'mobilenetv4_single' if modal_num == 1 else 'mobilenetv4_multimodal'
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


def plot_confusion_matrix(y_true, y_pred, class_names, normalize=True, figsize=(10, 8), cmap='Blues', suffix=''):
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


def plot_per_class_f1_scores(f1_scores, class_names, title, suffix=''):
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
    plot_per_class_f1_scores(per_class_f1, class_names, 'Per-Class F1 Scores - Test', suffix='test')


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
                    print(f"Loaded {mod} frame {i} from {mod_path}, shape: {array.shape}")
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

train_csv_path = os.path.join(csv_dir, "Multimodal_train1.csv")
val_csv_path = os.path.join(csv_dir, "Multimodal_val1.csv")
test_csv_path = os.path.join(csv_dir, "Multimodal_test1.csv")

train_df = pd.read_csv(train_csv_path)
val_df = pd.read_csv(val_csv_path)
test_df = pd.read_csv(test_csv_path)

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


# Model definition
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
            self.fc1 = nn.Linear(1280 * 3, 256)  # 1280 features per modality
        else:
            self.fc1 = nn.Linear(1280, 256)  # Single modality
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, rgb, flow=None, mask=None):
        print(f"RGB shape: {rgb.shape}", end="")
        if self.modal_num == 3:
            print(f", Flow shape: {flow.shape}, Mask shape: {mask.shape}")
        else:
            print()

        batch_size, num_frames, c, h, w = rgb.shape

        # Process RGB
        rgb = rgb.view(-1, c, h, w)
        rgb_features = self.rgb_model(rgb)
        rgb_features = rgb_features.view(batch_size, num_frames, -1)
        rgb_features = torch.mean(rgb_features, dim=1)  # Temporal average

        if self.modal_num == 3:
            # Process FLOW
            flow = flow.view(-1, c, h, w)
            flow_features = self.flow_model(flow)
            flow_features = flow_features.view(batch_size, num_frames, -1)
            flow_features = torch.mean(flow_features, dim=1)  # Temporal average

            # Process MASK
            mask = mask.view(-1, c, h, w)
            mask_features = self.mask_model(mask)
            mask_features = mask_features.view(batch_size, num_frames, -1)
            mask_features = torch.mean(mask_features, dim=1)  # Temporal average

            # Concatenate features
            combined_features = torch.cat((rgb_features, flow_features, mask_features), dim=1)
        else:
            combined_features = rgb_features

        # Fully connected layers
        out = self.fc1(combined_features)
        out = F.relu(out)
        out = self.fc2(out)
        return out


# Single modality model for inference time measurement
class SingleModalityMobileNetV4(nn.Module):
    def __init__(self):
        super(SingleModalityMobileNetV4, self).__init__()
        self.model = build_models.MobileNetV4('MobileNetV4ConvSmall')
        self.model.fc = nn.Identity()

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        features = self.model(x)
        features = features.view(batch_size, num_frames, -1)
        features = torch.mean(features, dim=1)  # Temporal average
        return features


# Initialize models
model = MultimodalWorkoutClassifier(num_classes=num_classes, modal_num=modal_num).to(device)
single_modality_model = SingleModalityMobileNetV4().to(device)

# Calculate model complexity using thop
try:
    input_rgb = torch.randn(1, num_frames, 3, 128, 128).to(device)
    if modal_num == 3:
        input_flow = torch.randn(1, num_frames, 3, 128, 128).to(device)
        input_mask = torch.randn(1, num_frames, 3, 128, 128).to(device)
        macs, params = profile(model, inputs=(input_rgb, input_flow, input_mask))
    else:
        macs, params = profile(model, inputs=(input_rgb,))
    print('{:<30}  {:<8}'.format('Computational complexity: ', f'{macs / 1e9:.2f} GFLOPs'))
    print('{:<30}  {:<8}'.format('Number of parameters: ', f'{params / 1e6:.2f} M'))
except Exception as e:
    print(f"Error calculating model complexity: {e}")


# Measure inference time for single modality
def measure_inference_time(model, input_tensor, num_runs=100):
    model.eval()
    with torch.no_grad():
        # Warm-up run
        for _ in range(10):
            _ = model(input_tensor)
        torch.cuda.synchronize()

        # Measure inference time
        start_time = time.time()
        for _ in range(num_runs):
            _ = model(input_tensor)
        torch.cuda.synchronize()
        end_time = time.time()

        avg_time = (end_time - start_time) / num_runs * 1000  # Convert to milliseconds
    return avg_time


# Measure inference time for single modality model
single_input = torch.randn(1, num_frames, 3, 128, 128).to(device)
avg_inference_time = measure_inference_time(single_modality_model, single_input)
print(f"Average inference time for single modality (RGB) MobileNetV4: {avg_inference_time:.2f} ms")

# Training setup
lr = 0.0001
weight_decay = 5e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

train_losses = []
val_losses = []
val_accuracies = []
val_f1_scores = []
val_precisions = []
val_recalls = []
num_epochs = 30
max_gained_acc = 0
epoch_of_max_acc = 0
early_stopping = False
epochs_tolerance = 5

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for rgb, flow, mask, labels in tqdm(train_loader):
        rgb, flow, mask, labels = rgb.to(device), flow.to(device) if flow is not None else None, mask.to(
            device) if mask is not None else None, labels.to(device)
        optimizer.zero_grad()
        if modal_num == 3:
            outputs = model(rgb, flow, mask)
        else:
            outputs = model(rgb)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}')

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    val_labels = []
    val_predictions = []
    with torch.no_grad():
        for rgb, flow, mask, labels in val_loader:
            rgb, flow, mask, labels = rgb.to(device), flow.to(device) if flow is not None else None, mask.to(
                device) if mask is not None else None, labels.to(device)
            if modal_num == 3:
                outputs = model(rgb, flow, mask)
            else:
                outputs = model(rgb)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_labels.extend(labels.cpu().numpy())
            val_predictions.extend(predicted.cpu().numpy())

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total
    f1_macro = f1_score(val_labels, val_predictions, average='macro')
    precision = precision_score(val_labels, val_predictions, average='macro')
    recall = recall_score(val_labels, val_predictions, average='macro')
    val_per_class_f1 = f1_score(val_labels, val_predictions, average=None)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    val_f1_scores.append(f1_macro)
    val_precisions.append(precision)
    val_recalls.append(recall)

    plot_confusion_matrix(val_labels, val_predictions, list(class_to_index.keys()),
                          suffix=f'epoch_{epoch + 1}', modal_num=modal_num)
    plot_per_class_f1_scores(val_per_class_f1, list(class_to_index.keys()),
                             f'Per-Class F1 Scores - Epoch {epoch + 1}',
                             suffix=f'epoch_{epoch + 1}', modal_num=modal_num)

    if val_accuracy >= max_gained_acc:
        max_gained_acc = val_accuracy
        epoch_of_max_acc = epoch

    print(
        f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}%, Val F1: {f1_macro:.4f}, Val Precision: {precision:.4f}, Val Recall: {recall:.4f}')
    print(f'Per-Class F1 Scores for Epoch {epoch + 1}:')
    for class_name, f1_score_value in zip(list(class_to_index.keys()), val_per_class_f1):
        print(f'  Class {class_name}: {f1_score_value:.4f}')

    if early_stopping and epoch - epoch_of_max_acc > epochs_tolerance:
        print("Early stopping triggered")
        break

    scheduler.step()

plot_results(train_losses, val_losses, val_accuracies, val_f1_scores, val_precisions, val_recalls, modal_num)


# Testing
def test():
    test_losses = []
    test_accuracies = []
    cum_labels = []
    cum_predicted = []
    show_gifs_results = True
    show_gifs_at = 5

    model.eval()
    with torch.no_grad():
        for i, (rgb, flow, mask, labels) in enumerate(test_loader):
            rgb, flow, mask, labels = rgb.to(device), flow.to(device) if flow is not None else None, mask.to(
                device) if mask is not None else None, labels.to(device)
            if modal_num == 3:
                outputs = model(rgb, flow, mask)
            else:
                outputs = model(rgb)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()

            test_loss = loss.item() / len(test_loader)
            test_accuracy = 100 * (correct / total)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            cum_labels.extend(labels.cpu().numpy())
            cum_predicted.extend(predicted.cpu().numpy())

            if show_gifs_results and i % show_gifs_at == 0:
                show_gifs(rgb, flow, mask, labels, predicted, index_to_class, modal_num)

    print('Test Accuracy:', np.mean(test_accuracies))
    get_metrics(test_losses, test_accuracies, cum_labels, cum_predicted, list(class_to_index.keys()), modal_num)
    plot_confusion_matrix(cum_labels, cum_predicted, list(class_to_index.keys()), suffix='test', modal_num=modal_num)


test()

# Save model
save_model = True
model_suffix = 'mobilenetv4_single' if modal_num == 1 else 'mobilenetv4_multimodal'
model_path = f"/home/zhangnb/videoUAV/videoWork/models/{model_suffix}.pth"
if save_model:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
