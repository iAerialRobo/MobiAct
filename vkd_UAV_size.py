import torch
import os
from thop import profile, clever_format
from models import build_models  # Assuming this contains MobileNetV4 implementation

# Define the MultimodalWorkoutClassifier class
class MultimodalWorkoutClassifier(torch.nn.Module):
    def __init__(self, num_classes=4, modal_num=3):
        super(MultimodalWorkoutClassifier, self).__init__()
        self.modal_num = modal_num
        try:
            base = build_models.MobileNetV4('MobileNetV4ConvSmall')
        except Exception as e:
            print(f"Error initializing MobileNetV4: {e}")
            raise
        self.layers = base
        self.stage1 = self.layers.conv0
        self.stage2 = self.layers.layer1
        self.stage3 = self.layers.layer2
        self.stage4 = self.layers.layer3
        self.stage5 = torch.nn.Sequential(self.layers.layer4, self.layers.layer5)
        if modal_num == 3:
            self.flow_model = build_models.MobileNetV4('MobileNetV4ConvSmall')
            self.mask_model = build_models.MobileNetV4('MobileNetV4ConvSmall')
            self.flow_stage1 = self.flow_model.conv0
            self.flow_stage2 = self.flow_model.layer1
            self.flow_stage3 = self.flow_model.layer2
            self.flow_stage4 = self.flow_model.layer3
            self.flow_stage5 = torch.nn.Sequential(self.flow_model.layer4, self.flow_model.layer5)
            self.mask_stage1 = self.mask_model.conv0
            self.mask_stage2 = self.mask_model.layer1
            self.mask_stage3 = self.mask_model.layer2
            self.mask_stage4 = self.mask_model.layer3
            self.mask_stage5 = torch.nn.Sequential(self.mask_model.layer4, self.mask_model.layer5)
        self.channel_counts = [32, 32, 64, 96, 1280]
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(self.channel_counts[-1] * modal_num, num_classes)
        )

    def forward(self, rgb, flow=None, mask=None, return_features=False):
        batch_size, num_frames, c, h, w = rgb.shape
        rgb = rgb.view(-1, c, h, w)
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
            if flow is None or mask is None:
                raise ValueError("Flow and mask inputs are required for modal_num=3")
            flow = flow.view(-1, c, h, w)
            mask = mask.view(-1, c, h, w)
            features_flow = []
            features_mask = []
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
        logits = self.classifier(combined_features)
        return logits

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modal_num = 3  # Set to 1 for single-modality or 3 for tri-modality
num_classes = 4
num_frames = 30
height = 128
width = 128

# Initialize the student model
try:
    student_model = MultimodalWorkoutClassifier(num_classes=num_classes, modal_num=modal_num).to(device)
    print("Student model initialized successfully.")
except Exception as e:
    print(f"Error initializing student model: {e}")
    exit(1)

# Test forward pass to debug model
try:
    rgb_input = torch.randn(1, num_frames, 3, height, width).to(device)
    if modal_num == 3:
        flow_input = torch.randn(1, num_frames, 3, height, width).to(device)
        mask_input = torch.randn(1, num_frames, 3, height, width).to(device)
        output = student_model(rgb_input, flow_input, mask_input)
    else:
        flow_input = None
        mask_input = None
        output = student_model(rgb_input)
    print(f"Forward pass successful. Output shape: {output.shape}")
except Exception as e:
    print(f"Error in forward pass: {e}")
    exit(1)

# 1. Calculate the number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

param_count = count_parameters(student_model)
print(f"Total Parameters: {param_count:,}")

# 2. Calculate FLOPs and MACs using thop
try:
    if modal_num == 3:
        macs, params = profile(student_model, inputs=(rgb_input, flow_input, mask_input), verbose=True)
    else:
        macs, params = profile(student_model, inputs=(rgb_input,), verbose=True)
    macs, params = clever_format([macs, params], "%.3f")
    print(f"MACs (Multiply-Accumulate Operations): {macs}")
    print(f"FLOPs (approximated as 2 * MACs): {float(macs.replace('G', '')) * 2:.3f} G")
    print(f"Parameters from thop: {params}")
except Exception as e:
    print(f"Error calculating FLOPs/MACs with thop: {e}")
    macs, params = 0, 0

# 3. Save the model and calculate its size
model_path = "student_model.pth"
try:
    torch.save(student_model.state_dict(), model_path)
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
    print(f"Model Size: {model_size:.2f} MB")
except Exception as e:
    print(f"Error saving model: {e}")
    model_size = 0

# Clean up
if os.path.exists(model_path):
    os.remove(model_path)  # Remove the saved model file to avoid clutter


# /home/zhangnb/miniconda3/bin/conda run -n videoWork --no-capture-output python /home/zhangnb/videoUAV/videoWork/lightWeightVideo/vkd_UAV_size.py
# /home/zhangnb/videoUAV/videoWork/models/build_models.py:358: UserWarning: Overwriting mobilenetv4_hybrid_medium in registry with models.build_models.mobilenetv4_hybrid_medium. This is because the name being registered conflicts with an existing name. Please check if this is not expected.
#   def mobilenetv4_hybrid_medium(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
# /home/zhangnb/videoUAV/videoWork/models/build_models.py:364: UserWarning: Overwriting mobilenetv4_hybrid_large in registry with models.build_models.mobilenetv4_hybrid_large. This is because the name being registered conflicts with an existing name. Please check if this is not expected.
#   def mobilenetv4_hybrid_large(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
# Student model initialized successfully.
# Forward pass successful. Output shape: torch.Size([1, 4])
# Total Parameters: 11,337,436
# [INFO] Register count_convNd() for <class 'torch.nn.modules.conv.Conv2d'>.
# [INFO] Register count_normalization() for <class 'torch.nn.modules.batchnorm.BatchNorm2d'>.
# [INFO] Register zero_ops() for <class 'torch.nn.modules.activation.ReLU6'>.
# [INFO] Register zero_ops() for <class 'torch.nn.modules.container.Sequential'>.
# [INFO] Register count_linear() for <class 'torch.nn.modules.linear.Linear'>.
# [INFO] Register zero_ops() for <class 'torch.nn.modules.dropout.Dropout'>.
# MACs (Multiply-Accumulate Operations): 29.601G
# FLOPs (approximated as 2 * MACs): 59.202 G
# Parameters from thop: 14.974M
# Model Size: 43.99 MB
#
# Process finished with exit code 0