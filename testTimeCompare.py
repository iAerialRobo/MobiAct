import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from glob import glob
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming these are available in your environment
from i3d_transformer import I3dTransformer
from C3D_model import C3D
from Cost_model import Cost, costblock
from models import build_models  # For MobileNetV4 in vKd_UAV_recognition

# Specify GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device --> {device}")

# Data paths and class mappings
dataset_path_remote = '/home/zhangnb/videoUAV/data/hdf5/videoDataIntegrate/remote'
dataset_path_middle = '/home/zhangnb/videoUAV/data/hdf5/videoDataIntegrate/middle'
class_folders = {
    '0': 'MAV_inv_vShapeRGB',
    '1': 'MAV_left_rightRGB',
    '2': 'MAV_up_downRGB',
    '3': 'MAV_vShapeRGB'
}
class_modality_folders = {
    '0': {'RGB': 'MAV_inv_vShapeRGB', 'FLOW': 'MAV_inv_vShapeFLOW', 'MASK': 'MAV_inv_vShapeMASK'},
    '1': {'RGB': 'MAV_left_rightRGB', 'FLOW': 'MAV_left_rightFLOW', 'MASK': 'MAV_left_rightMASK'},
    '2': {'RGB': 'MAV_up_downRGB', 'FLOW': 'MAV_up_downFLOW', 'MASK': 'MAV_up_downMASK'},
    '3': {'RGB': 'MAV_vShapeRGB', 'FLOW': 'MAV_vShapeFLOW', 'MASK': 'MAV_vShapeMASK'}
}


# Data loading utilities
def load_video_frames(video_path, label, num_frames, height, width, modalities=['RGB']):
    frames = {mod: [] for mod in modalities}
    label_str = str(label)
    for mod in modalities:
        if mod == 'RGB':
            mod_path = video_path
        else:
            mod_folder = class_modality_folders[label_str][mod]
            mod_path = video_path.replace(class_modality_folders[label_str]['RGB'], mod_folder)
            mod_path = mod_path.replace('_RGB.', f'_{mod}.')
        try:
            with h5py.File(mod_path, 'r') as hdf:
                total_frames = len([key for key in hdf.keys() if key.startswith('array_')])
                if total_frames < num_frames:
                    indices = list(range(total_frames)) + [total_frames - 1] * (num_frames - total_frames)
                else:
                    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
                for i in indices:
                    array = np.array(hdf[f'array_{i}'])
                    if array.shape[-1] != 3:
                        print(f"Skipping {mod_path}: Expected 3 channels, got {array.shape[-1]}")
                        array = np.zeros((height, width, 3), dtype=np.uint8)
                    if array.shape[:2] != (height, width):
                        img = Image.fromarray(array.astype(np.uint8))
                        img = img.resize((width, height), Image.Resampling.LANCZOS)
                        array = np.array(img)
                    frames[mod].append(array)
        except Exception as e:
            print(f"Error loading {mod} from {mod_path}: {e}")
            frames[mod] = [np.zeros((height, width, 3), dtype=np.uint8)] * num_frames
    return [frames[mod] for mod in modalities]


class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform, num_frames, height, width, modalities=['RGB']):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.modalities = modalities

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label_name = str(self.labels[idx])
        label_index = class_to_index[label_name]
        frames = load_video_frames(video_path, label_name, self.num_frames, self.height, self.width, self.modalities)
        tensors = []
        for mod_frames in frames:
            mod_frames = [torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) for frame in mod_frames]
            if self.transform:
                mod_frames = [self.transform(frame) for frame in mod_frames]
            mod_tensor = torch.stack(mod_frames)  # (T, C, H, W)
            mod_tensor = mod_tensor.permute(1, 0, 2, 3)  # (C, T, H, W)
            tensors.append(mod_tensor)
        label_tensor = torch.tensor(label_index, dtype=torch.long)
        tensors = [t.to(device) for t in tensors]
        label_tensor = label_tensor.to(device)
        return (*tensors, label_tensor)


def create_csv(videos, labels, file_name):
    df = pd.DataFrame({'path': videos, 'label': labels})
    df.to_csv(file_name, index=False)


# Model definitions
class I3DClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(I3DClassifier, self).__init__()
        self.base_model = I3dTransformer(num_classes=num_classes, d_model=64,
                                         transformer_config={'d_ff': 32, 'num_heads': 8, 'dropout': 0, 'num_layers': 2})
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, rgb):
        rgb = rgb.repeat(1, 1, 2, 1, 1)  # (B, C, T, H, W) -> (B, C*2, T, H, W)
        result = self.base_model(rgb)
        logits = self.fc2(result["embds"])
        return logits


class C3DClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(C3DClassifier, self).__init__()
        self.c3d = C3D(num_classes=num_classes, pretrained=False)

    def forward(self, rgb):
        return self.c3d(rgb)


class CostClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(CostClassifier, self).__init__()
        self.cost = Cost(num_classes, costblock, [2, 3, 4, 2])

    def forward(self, rgb):
        return self.cost(rgb)


class MultimodalWorkoutClassifier(nn.Module):
    def __init__(self, num_classes=4, modal_num=3):
        super(MultimodalWorkoutClassifier, self).__init__()
        self.modal_num = modal_num
        base = build_models.MobileNetV4('MobileNetV4ConvSmall')
        self.stage1 = base.conv0
        self.stage2 = base.layer1
        self.stage3 = base.layer2
        self.stage4 = base.layer3
        self.stage5 = nn.Sequential(base.layer4, base.layer5)
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

    def forward(self, rgb, flow=None, mask=None):
        batch_size, num_frames, c, h, w = rgb.shape
        rgb = rgb.view(-1, c, h, w)
        x = self.stage1(rgb)
        x = x.view(batch_size, num_frames, x.size(1), x.size(2), x.size(3))
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.stage2(x)
        x = x.view(batch_size, num_frames, x.size(1), x.size(2), x.size(3))
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.stage3(x)
        x = x.view(batch_size, num_frames, x.size(1), x.size(2), x.size(3))
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.stage4(x)
        x = x.view(batch_size, num_frames, x.size(1), x.size(2), x.size(3))
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.stage5(x)
        x = x.view(batch_size, num_frames, x.size(1), x.size(2), x.size(3))
        rgb_features = x.mean([3, 4]).mean(dim=1)
        all_features = [rgb_features]
        if self.modal_num == 3:
            flow = flow.view(-1, c, h, w)
            mask = mask.view(-1, c, h, w)
            x_flow = self.flow_stage1(flow)
            x_flow = x_flow.view(batch_size, num_frames, x_flow.size(1), x_flow.size(2), x_flow.size(3))
            x_flow = x_flow.view(-1, x_flow.size(2), x_flow.size(3), x_flow.size(4))
            x_flow = self.flow_stage2(x_flow)
            x_flow = x_flow.view(batch_size, num_frames, x_flow.size(1), x_flow.size(2), x_flow.size(3))
            x_flow = x_flow.view(-1, x_flow.size(2), x_flow.size(3), x_flow.size(4))
            x_flow = self.flow_stage3(x_flow)
            x_flow = x_flow.view(batch_size, num_frames, x_flow.size(1), x_flow.size(2), x_flow.size(3))
            x_flow = x_flow.view(-1, x_flow.size(2), x_flow.size(3), x_flow.size(4))
            x_flow = self.flow_stage4(x_flow)
            x_flow = x_flow.view(batch_size, num_frames, x_flow.size(1), x_flow.size(2), x_flow.size(3))
            x_flow = x_flow.view(-1, x_flow.size(2), x_flow.size(3), x_flow.size(4))
            x_flow = self.flow_stage5(x_flow)
            x_flow = x_flow.view(batch_size, num_frames, x_flow.size(1), x_flow.size(2), x_flow.size(3))
            flow_features = x_flow.mean([3, 4]).mean(dim=1)
            x_mask = self.mask_stage1(mask)
            x_mask = x_mask.view(batch_size, num_frames, x_mask.size(1), x_mask.size(2), x_mask.size(3))
            x_mask = x_mask.view(-1, x_mask.size(2), x_mask.size(3), x_mask.size(4))
            x_mask = self.mask_stage2(x_mask)
            x_mask = x_mask.view(batch_size, num_frames, x_mask.size(1), x_mask.size(2), x_mask.size(3))
            x_mask = x_mask.view(-1, x_mask.size(2), x_mask.size(3), x_mask.size(4))
            x_mask = self.mask_stage3(x_mask)
            x_mask = x_mask.view(batch_size, num_frames, x_mask.size(1), x_mask.size(2), x_mask.size(3))
            x_mask = x_mask.view(-1, x_mask.size(2), x_mask.size(3), x_mask.size(4))
            x_mask = self.mask_stage4(x_mask)
            x_mask = x_mask.view(batch_size, num_frames, x_mask.size(1), x_mask.size(2), x_mask.size(3))
            x_mask = x_mask.view(-1, x_mask.size(2), x_mask.size(3), x_mask.size(4))
            x_mask = self.mask_stage5(x_mask)
            x_mask = x_mask.view(batch_size, num_frames, x_mask.size(1), x_mask.size(2), x_mask.size(3))
            mask_features = x_mask.mean([3, 4]).mean(dim=1)
            all_features.extend([flow_features, mask_features])
        combined_features = torch.cat(all_features, dim=1)
        logits = self.classifier(combined_features)
        return logits


class MultimodalWorkoutClassifierWithFPN(nn.Module):
    def __init__(self, num_classes=4):
        super(MultimodalWorkoutClassifierWithFPN, self).__init__()
        self.rgb_model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.flow_model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.mask_model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        self.rgb_model.fc = nn.Identity()
        self.flow_model.fc = nn.Identity()
        self.mask_model.fc = nn.Identity()
        self.rgb_fpn = FeaturePyramidNetwork()
        self.flow_fpn = FeaturePyramidNetwork()
        self.mask_fpn = FeaturePyramidNetwork()
        self.cross_attention = nn.MultiheadAttention(embed_dim=3072, num_heads=12)
        self.fc1 = nn.Linear(3072, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def extract_features(self, model, x):
        features = []
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        features.append(x)
        x = model.layer2(x)
        features.append(x)
        x = model.layer3(x)
        features.append(x)
        x = model.layer4(x)
        features.append(x)
        return features

    def forward(self, rgb, flow, mask):
        batch_size, c, num_frames, h, w = rgb.shape
        # Reshape to (batch_size * num_frames, C, H, W) for ResNet
        rgb = rgb.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
        flow = flow.permute(0, 2, 1, 3, 4).contiguous()
        mask = mask.permute(0, 2, 1, 3, 4).contiguous()
        rgb = rgb.view(batch_size * num_frames, c, h, w)
        flow = flow.view(batch_size * num_frames, c, h, w)
        mask = mask.view(batch_size * num_frames, c, h, w)
        # Extract features using ResNet
        rgb_features = self.extract_features(self.rgb_model, rgb)
        flow_features = self.extract_features(self.flow_model, flow)
        mask_features = self.extract_features(self.mask_model, mask)
        # Process through FPN
        rgb_fpn = self.rgb_fpn(rgb_features).view(batch_size, num_frames, -1)
        flow_fpn = self.flow_fpn(flow_features).view(batch_size, num_frames, -1)
        mask_fpn = self.mask_fpn(mask_features).view(batch_size, num_frames, -1)
        # Combine features for cross-attention
        all_features = torch.stack([rgb_fpn, flow_fpn, mask_fpn], dim=0)  # (3, B, T, F)
        all_features = all_features.permute(1, 2, 0, 3)  # (B, T, 3, F)
        all_features = all_features.reshape(batch_size, num_frames, -1)  # (B, T, 3*F)
        # Apply cross-attention
        attn_output, _ = self.cross_attention(all_features, all_features, all_features)
        combined = torch.mean(attn_output, dim=1)  # (B, 3*F)
        # Final classification
        out = self.fc1(combined)
        out = torch.nn.functional.relu(out)
        out = self.fc2(out)
        return out


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list=[64, 128, 256, 512], out_channels=256):
        super(FeaturePyramidNetwork, self).__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1) for in_channels in in_channels_list
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, features):
        laterals = [lateral_conv(f) for lateral_conv, f in zip(self.lateral_convs, features)]
        out = laterals[-1]
        outputs = [out]
        for lateral in reversed(laterals[:-1]):
            out = torch.nn.functional.interpolate(out, scale_factor=2, mode='nearest') + lateral
            outputs.append(out)
        outputs = reversed(outputs)
        outputs = [self.pool(conv(out)).view(out.size(0), -1) for conv, out in zip(self.output_convs, outputs)]
        return torch.cat(outputs, dim=-1)


# Model configurations
model_configs = [
    {
        'name': 'I3D',
        'model_class': I3DClassifier,
        'model_path': '/home/zhangnb/videoUAV/videoWork/models/i3d_rgb_only_model.pth',
        'num_frames': 30,
        'height': 224,
        'width': 224,
        'modalities': ['RGB'],
        'dataset_path': dataset_path_remote,
        'batch_size': 6
    },
    {
        'name': 'C3D',
        'model_class': C3DClassifier,
        'model_path': '/home/zhangnb/videoUAV/videoWork/models/c3d_rgb_only_model.pth',
        'num_frames': 16,
        'height': 112,
        'width': 112,
        'modalities': ['RGB'],
        'dataset_path': dataset_path_remote,
        'batch_size': 20
    },
    {
        'name': 'Cost',
        'model_class': CostClassifier,
        'model_path': '/home/zhangnb/videoUAV/videoWork/models/cost_rgb_only_model.pth',
        'num_frames': 16,
        'height': 224,
        'width': 224,
        'modalities': ['RGB'],
        'dataset_path': dataset_path_remote,
        'batch_size': 16
    },
    {
        'name': 'MultiAlignment',
        'model_class': MultimodalWorkoutClassifierWithFPN,
        'model_path': '/home/zhangnb/videoUAV/videoWork/models/corrected_multimodal_final_model.pth',
        'num_frames': 30,
        'height': 128,
        'width': 128,
        'modalities': ['RGB', 'FLOW', 'MASK'],
        'dataset_path': dataset_path_middle,
        'batch_size': 20
    },
    {
        'name': 'MobileNetV4',
        'model_class': MultimodalWorkoutClassifier,
        'model_path': '/home/zhangnb/videoUAV/videoWork/models/multimodal_final_model.pth',
        'num_frames': 30,
        'height': 128,
        'width': 128,
        'modalities': ['RGB', 'FLOW', 'MASK'],
        'dataset_path': dataset_path_middle,
        'batch_size': 20
    }
]

# Load and split data
num_classes = 4
class_to_index = {str(l): idx for idx, l in enumerate(['0', '1', '2', '3'])}
index_to_class = {idx: str(l) for idx, l in enumerate(['0', '1', '2', '3'])}

# Prepare datasets
datasets = {}
for config in model_configs:
    dataset_path = config['dataset_path']
    modalities = config['modalities']
    all_video_paths = []
    all_labels = []
    for class_id in class_folders if dataset_path == dataset_path_remote else class_modality_folders:
        rgb_folder = class_folders[class_id] if dataset_path == dataset_path_remote else \
        class_modality_folders[class_id]['RGB']
        class_paths = glob(os.path.join(dataset_path, rgb_folder, '*.h5'), recursive=True)
        class_labels = [class_id] * len(class_paths)
        all_video_paths.extend(class_paths)
        all_labels.extend(class_labels)

    _, test_videos, _, test_labels = train_test_split(
        all_video_paths, all_labels, test_size=0.3, random_state=42)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config['height'], config['width'])),
        transforms.ToTensor(),
    ])

    test_dataset = VideoDataset(
        video_paths=test_videos,
        labels=test_labels,
        transform=transform,
        num_frames=config['num_frames'],
        height=config['height'],
        width=config['width'],
        modalities=config['modalities']
    )
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    datasets[config['name']] = test_loader


# Function to measure inference time per sample
def measure_inference_time(model, test_loader, modalities, device):
    model.eval()
    inference_times_per_sample = []
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference time for batch"):
            inputs = [batch[i].to(device) for i in range(len(modalities))]
            labels = batch[-1].to(device)
            # Print input shapes for debugging
            print(f"Input shapes: {[inp.shape for inp in inputs]}")
            batch_size = inputs[0].size(0)  # Number of samples in the batch
            torch.cuda.synchronize()
            start_time = time.time()
            if len(modalities) == 1:
                _ = model(inputs[0])
            else:
                _ = model(*inputs)
            torch.cuda.synchronize()
            end_time = time.time()
            # Calculate time per sample in this batch
            time_per_sample = (end_time - start_time) / batch_size * 1000  # Convert to milliseconds
            inference_times_per_sample.append(time_per_sample * batch_size)
            total_samples += batch_size
    # Compute average inference time per sample
    avg_time_per_sample = sum(inference_times_per_sample) / total_samples
    return avg_time_per_sample


# Measure inference time for each model
results = []
for config in model_configs:
    print(f"\n=== Measuring inference time for {config['name']} ===")
    model = config['model_class'](num_classes=num_classes).to(device)

    # Load model weights
    try:
        model.load_state_dict(torch.load(config['model_path'], map_location=device))
        print(f"Loaded weights from {config['model_path']}")
    except Exception as e:
        print(f"Error loading weights for {config['name']} from {config['model_path']}: {e}")
        continue

    test_loader = datasets[config['name']]
    avg_inference_time_per_sample = measure_inference_time(model, test_loader, config['modalities'], device)
    results.append({
        'Model': config['name'],
        'Average Inference Time per Sample (ms)': avg_inference_time_per_sample,
        'Modalities': ', '.join(config['modalities']),
        'Input Size': f"{config['height']}x{config['width']}x{config['num_frames']}",
        'Batch Size': config['batch_size']
    })
    print(f"Average inference time per sample for {config['name']}: {avg_inference_time_per_sample:.2f} ms")

# Save results to a DataFrame and plot
results_df = pd.DataFrame(results)
print("\n=== Inference Time Results (per Sample) ===")
print(results_df)

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x='Average Inference Time per Sample (ms)', y='Model', hue='Modalities', data=results_df)
plt.title('Average Inference Time per Sample per Model')
plt.xlabel('Inference Time per Sample (ms)')
plt.ylabel('Model')
plt.tight_layout()
plt.savefig('inference_times_per_sample.png', dpi=300)
plt.close()

# Save results to CSV
results_df.to_csv('inference_times_per_sample.csv', index=False)
print("\nResults saved to 'inference_times_per_sample.csv' and 'inference_times_per_sample.png'")
