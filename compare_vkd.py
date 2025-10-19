import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models


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
    def __init__(self, teacher, student, feature_dim=512):
        super(VkD, self).__init__()
        self.teacher = teacher
        self.student = student
        self.student_feature_adapter = nn.Linear(1280, feature_dim)
        self.proj = OrthogonalProjection(feature_dim, feature_dim)
        self.student.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, student.classifier[1].out_features)
        )

    def forward(self, x, train_student=True):
        with torch.no_grad():
            teacher_features = self.teacher.features(x)
            teacher_features = teacher_features.view(teacher_features.size(0), -1)
            teacher_features = (teacher_features - teacher_features.mean(dim=0)) / (teacher_features.std(dim=0) + 1e-6)

        if train_student:
            student_features = self.student.features(x)
            student_features = student_features.mean([2, 3])
            student_features = self.student_feature_adapter(student_features)
            student_features = self.proj(student_features)
            student_logits = self.student.classifier(student_features)
            return student_logits, student_features, teacher_features
        else:
            features = self.student.features(x)
            features = features.mean([2, 3])
            features = self.student_feature_adapter(features)
            return self.student.classifier(features)


# Custom feature loss
def feature_loss(student_features, teacher_features):
    return F.mse_loss(student_features, teacher_features)


# Training function for VkD (with distillation)
def train_vkd(model, train_loader, optimizer, epoch, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits, student_features, teacher_features = model(data)
        cls_loss = F.cross_entropy(logits, target)
        feat_loss = feature_loss(student_features, teacher_features)
        alpha = 0.5
        loss = cls_loss + alpha * feat_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item():.4f} (Cls: {cls_loss.item():.4f}, Feat: {feat_loss.item():.4f})')
    avg_loss = total_loss / len(train_loader)
    return avg_loss


# Training function for MobileNetV2 (without distillation)
def train_baseline(model, train_loader, optimizer, epoch, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item():.4f}')
    avg_loss = total_loss / len(train_loader)
    return avg_loss


# Evaluation function
def test_model(model, test_loader, device, is_vkd=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if is_vkd:
                output = model(data, train_student=False)
            else:
                output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(
        f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return test_loss, accuracy


# Modified ResNet-18 and MobileNetV2 for feature extraction
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        base = models.resnet18(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        base = models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.features = base.features
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x


# Main function
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loading (CIFAR-10)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # Initialize models
    teacher = ResNet18(num_classes=10).to(device)
    student_vkd = MobileNetV2(num_classes=10).to(device)
    student_baseline = MobileNetV2(num_classes=10).to(device)
    teacher.eval()

    # Initialize VkD model (with distillation)
    model_vkd = VkD(teacher, student_vkd, feature_dim=512).to(device)
    optimizer_vkd = torch.optim.Adam(
        list(model_vkd.student.parameters()) + list(model_vkd.student_feature_adapter.parameters()) +
        list(model_vkd.proj.parameters()), lr=0.001
    )

    # Initialize optimizer for baseline (without distillation)
    optimizer_baseline = torch.optim.Adam(student_baseline.parameters(), lr=0.001)

    # Lists to store metrics
    vkd_losses, vkd_accuracies = [], []
    baseline_losses, baseline_accuracies = [], []

    # Training loop for both models
    epochs = 10
    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch} ===")
        print("Training VkD (with distillation):")
        vkd_loss = train_vkd(model_vkd, train_loader, optimizer_vkd, epoch, device)
        print("Training MobileNetV2 (without distillation):")
        baseline_loss = train_baseline(student_baseline, train_loader, optimizer_baseline, epoch, device)

        print("Evaluating VkD:")
        vkd_test_loss, vkd_accuracy = test_model(model_vkd, test_loader, device, is_vkd=True)
        print("Evaluating MobileNetV2:")
        baseline_test_loss, baseline_accuracy = test_model(student_baseline, test_loader, device, is_vkd=False)

        vkd_losses.append(vkd_test_loss)
        vkd_accuracies.append(vkd_accuracy)
        baseline_losses.append(baseline_test_loss)
        baseline_accuracies.append(baseline_accuracy)

    # Plot comparison
    print("\n=== Final Comparison ===")
    print("VkD (with distillation) final accuracy: {:.2f}%".format(vkd_accuracies[-1]))
    print("MobileNetV2 (without distillation) final accuracy: {:.2f}%".format(baseline_accuracies[-1]))

    # Create accuracy comparison chart
    '''chartjs
    {
        "type": "line",
        "data": {
            "labels": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "datasets": [
                {
                    "label": "VkD (with KD) Accuracy",
                    "data": ${vkd_accuracies},
    "borderColor": "#007bff",
    "backgroundColor": "rgba(0, 123, 255, 0.1)",
    "fill": true
    },
    {
        "label": "MobileNetV2 (without KD) Accuracy",
        "data": ${baseline_accuracies},
    "borderColor": "#dc3545",
    "backgroundColor": "rgba(220, 53, 69, 0.1)",
    "fill": true
    }
    ]
    },
    "options": {
        "responsive": true,
        "plugins": {
            "legend": {"position": "top"},
            "title": {"display": true, "text": "Accuracy Comparison Over Epochs"}
        },
        "scales": {
            "x": {"title": {"display": true, "text": "Epoch"}},
            "y": {"title": {"display": true, "text": "Accuracy (%)"}, "min": 0, "max": 100}
        }
    }
    }
    ```

    # Create loss comparison chart
    ```chartjs
    {
        "type": "line",
        "data": {
            "labels": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "datasets": [
                {
                    "label": "VkD (with KD) Loss",
                    "data": ${vkd_losses},
    "borderColor": "#007bff",
    "backgroundColor": "rgba(0, 123, 255, 0.1)",
    "fill": true
    },
    {
        "label": "MobileNetV2 (without KD) Loss",
        "data": ${baseline_losses},
    "borderColor": "#dc3545",
    "backgroundColor": "rgba(220, 53, 69, 0.1)",
    "fill": true
    }
    ]
    },
    "options": {
        "responsive": true,
        "plugins": {
            "legend": {"position": "top"},
            "title": {"display": true, "text": "Test Loss Comparison Over Epochs"}
        },
        "scales": {
            "x": {"title": {"display": true, "text": "Epoch"}},
            "y": {"title": {"display": true, "text": "Test Loss"}, "min": 0}
        }
    }
    }
    '''


if __name__ == '__main__':
    main()
'''
### 代码说明
1. ** 带蒸馏的模型（VkD） ** ：
- 使用你提供的
`VkD`
类，包含教师模型（ResNet18）和学生模型（MobileNetV2）。
- 训练时使用分类损失和特征蒸馏损失的组合（`alpha = 0.5
`）。
- 优化器包括学生模型、特征适配器和投影层的参数。

2. ** 不带蒸馏的模型（Baseline） ** ：
- 直接训练
`MobileNetV2`，仅使用交叉熵损失。
- 使用相同的优化器（Adam，学习率0
.001）和训练设置（10
个epoch，批大小128）。

3. ** 训练与评估 **：
- `train_vkd`：训练带蒸馏的模型，返回平均训练损失。
- `train_baseline`：训练不带蒸馏的模型，返回平均训练损失。
- `test_model`：评估模型在测试集上的损失和准确率，适配
`VkD`
和基线模型的推理模式。

4. ** 结果对比 **：
- 每个epoch记录两者的测试损失和准确率。
- 最终打印两者的最终准确率。
- 使用Chart.js生成两个折线图：
- 一个比较每个epoch的测试准确率。
- 另一个比较每个epoch的测试损失。
- 图表使用不同颜色（蓝色表示VkD，红色表示基线）以区分两者的性能。

### 预期输出
运行代码后，你将看到：
- 每个epoch的训练日志，显示VkD和基线模型的损失。
- 每个epoch的测试结果，显示两者的测试损失和准确率。
- 最终对比，显示两者的最终准确率。
- 两个折线图：
- ** 准确率对比图 **：显示VkD和基线模型在10个epoch中的准确率变化。
- ** 损失对比图 **：显示VkD和基线模型在10个epoch中的测试损失变化。

### 注意事项
1. ** 运行环境 **：确保你的环境有
`torch`、`torchvision`
等依赖，并有足够的GPU / CPU资源。CIFAR - 10
数据集会自动下载到
`. / data
`。
2. ** 训练时间 **：训练10个epoch可能需要一些时间，具体取决于硬件（GPU会显著加速）。
3. ** 预训练权重 **：代码使用
`IMAGENET1K_V1`
预训练权重，确保网络连接正常以下载权重。
4. ** 图表显示 **：Chart.js代码块假设你的环境支持交互式图表显示。如果你在非交互式环境中运行（如终端），可以手动将
`vkd_accuracies`、`baseline_accuracies`、`vkd_losses`、`baseline_losses`
保存并用其他工具（如Matplotlib）绘制。

### 预期结果分析
- ** 带蒸馏（VkD） ** ：由于有ResNet18教师模型的指导，学生模型（MobileNetV2）通常会获得更高的准确率，尤其是在早期epoch，因为特征蒸馏帮助学生学习教师的中间表示。
- ** 不带蒸馏（Baseline） ** ：基线模型仅依靠交叉熵损失，可能收敛较慢或最终准确率略低，因为它没有教师模型的额外信息。
- ** 图表观察 **：
- 准确率图：VkD的曲线可能上升更快，最终值可能高于基线。
- 损失图：VkD的测试损失可能下降更快，表明更好的泛化。

### 运行代码
# 保存代码为
# `vkd_comparison.py`，并运行：
# ```bash
# conda
# run - n
# videoWork - -no - capture - output
# python
# vkd_comparison.py
# ```

# 如果遇到其他错误（例如数据加载问题、设备问题或数值不稳定），请提供错误信息，我会进一步帮助调试。如果需要调整参数（例如
# `alpha`、学习率或epoch数）或添加其他对比指标（例如训练时间或推理速度），请告诉我！
'''