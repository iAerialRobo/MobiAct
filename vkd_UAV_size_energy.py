import torch
import os
import time  # Added for measuring inference time
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

# Print input image size
print(
    f"Input Image Size: ({num_frames}, 3, {height}, {width}) per modality (RGB{' + Flow + Mask' if modal_num == 3 else ''})")

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

# 3. Measure CPU inference time FIRST (before thop profiling)
print("\n=== Measuring CPU Inference Time ===")
try:
    # Create a separate model instance for CPU timing to avoid thop interference
    cpu_model = MultimodalWorkoutClassifier(num_classes=num_classes, modal_num=modal_num)
    cpu_model.load_state_dict(student_model.state_dict())  # Copy weights
    cpu_model = cpu_model.to('cpu')
    cpu_model.eval()  # Set to evaluation mode

    # Create CPU inputs
    rgb_input_cpu = torch.randn(1, num_frames, 3, height, width)
    if modal_num == 3:
        flow_input_cpu = torch.randn(1, num_frames, 3, height, width)
        mask_input_cpu = torch.randn(1, num_frames, 3, height, width)

    # Warm-up run to ensure model is compiled
    with torch.no_grad():
        if modal_num == 3:
            _ = cpu_model(rgb_input_cpu, flow_input_cpu, mask_input_cpu)
        else:
            _ = cpu_model(rgb_input_cpu)

    # Measure inference time over multiple runs for accuracy
    num_runs = 50  # Increased for better accuracy
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            if modal_num == 3:
                _ = cpu_model(rgb_input_cpu, flow_input_cpu, mask_input_cpu)
            else:
                _ = cpu_model(rgb_input_cpu)
    end_time = time.time()
    avg_inference_time = (end_time - start_time) / num_runs
    print(f"Average CPU Inference Time (over {num_runs} runs): {avg_inference_time:.4f} seconds")
    print(f"CPU Inference FPS: {1 / avg_inference_time:.2f}")

    # Clean up CPU model
    del cpu_model

except Exception as e:
    print(f"Error measuring CPU inference time: {e}")

# 2. Calculate FLOPs and MACs using thop (do this AFTER timing measurement)
print("\n=== Calculating FLOPs and MACs ===")
try:
    # Use fresh inputs for thop profiling
    if modal_num == 3:
        macs, params = profile(student_model, inputs=(rgb_input, flow_input, mask_input), verbose=False)
    else:
        macs, params = profile(student_model, inputs=(rgb_input,), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print(f"MACs (Multiply-Accumulate Operations): {macs}")
    print(
        f"FLOPs (approximated as 2 * MACs): {float(macs.replace('G', '').replace('M', '').replace('K', '')) * 2:.3f} {'G' if 'G' in macs else 'M' if 'M' in macs else 'K'}")
    print(f"Parameters from thop: {params}")
except Exception as e:
    print(f"Error calculating FLOPs/MACs with thop: {e}")
    print("Note: FLOPs calculation failed, but this doesn't affect the model functionality.")

# 4. Save the model and calculate its size
print("\n=== Model Size Calculation ===")
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

print("\n=== Summary ===")
print(f"Total Parameters: {param_count:,}")
if 'avg_inference_time' in locals():
    print(f"CPU Inference Time: {avg_inference_time:.4f} seconds")
    print(f"CPU Inference FPS: {1 / avg_inference_time:.2f}")
if 'model_size' in locals() and model_size > 0:
    print(f"Model Size: {model_size:.2f} MB")