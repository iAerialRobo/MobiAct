import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import uuid

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

# Modified ResNet18 for stage-wise feature extraction
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
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
        features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)  # Stage 1: after conv1 (64 channels)
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)  # Stage 2: after layer1 (64 channels)
        x = self.layer2(x)
        features.append(x)  # Stage 3: after layer2 (128 channels)
        x = self.layer3(x)
        features.append(x)  # Stage 4: after layer3 (256 channels)
        x = self.layer4(x)
        x = self.avgpool(x)
        features.append(x.view(x.size(0), -1))  # Stage 5: after avgpool (512 channels)
        if return_features:
            return features
        x = self.classifier(features[-1])
        return x

# Modified MobileNetV2 for stage-wise feature extraction
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        base = models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.features = base.features
        self.stage1 = nn.Sequential(*base.features[0:2])  # Conv + block 1
        self.stage2 = nn.Sequential(*base.features[2:4])  # Blocks 2-3
        self.stage3 = nn.Sequential(*base.features[4:7])  # Blocks 4-6
        self.stage4 = nn.Sequential(*base.features[7:14]) # Blocks 7-13
        self.stage5 = nn.Sequential(*base.features[14:])  # Blocks 14-18 + conv
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x, return_features=False):
        features = []
        x = self.stage1(x)
        features.append(x)  # Stage 1: 16 channels
        x = self.stage2(x)
        features.append(x)  # Stage 2: 24 channels
        x = self.stage3(x)
        features.append(x)  # Stage 3: 32 channels
        x = self.stage4(x)
        features.append(x)  # Stage 4: 96 channels
        x = self.stage5(x)
        features.append(x.mean([2, 3]))  # Stage 5: 1280 channels (after pooling)
        if return_features:
            return features
        x = self.classifier(features[-1])
        return x

# Layer-wise VkD Model
class LayerWiseVkD(nn.Module):
    def __init__(self, teacher, student, feature_dims=[64, 64, 128, 256, 512]):
        super(LayerWiseVkD, self).__init__()
        self.teacher = teacher
        self.student = student
        # Feature adapters for each stage, adjusted for MobileNetV2 channel counts
        self.feature_adapters = nn.ModuleList([
            nn.Linear(16, feature_dims[0]),   # Stage 1: MobileNetV2 (16) -> ResNet18 (64)
            nn.Linear(24, feature_dims[1]),   # Stage 2: MobileNetV2 (24) -> ResNet18 (64)
            nn.Linear(32, feature_dims[2]),   # Stage 3: MobileNetV2 (32) -> ResNet18 (128)
            nn.Linear(96, feature_dims[3]),   # Stage 4: MobileNetV2 (96) -> ResNet18 (256)
            nn.Linear(1280, feature_dims[4])  # Stage 5: MobileNetV2 (1280) -> ResNet18 (512)
        ])
        self.projections = nn.ModuleList([
            OrthogonalProjection(dim, dim) for dim in feature_dims
        ])
        self.student.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dims[-1], student.classifier[1].out_features)
        )

    def forward(self, x, train_student=True):
        with torch.no_grad():
            teacher_features = self.teacher(x, return_features=True)
            for i in range(len(teacher_features)):
                if teacher_features[i].dim() > 2:
                    teacher_features[i] = teacher_features[i].mean([2, 3])
                teacher_features[i] = (teacher_features[i] - teacher_features[i].mean(dim=0)) / (teacher_features[i].std(dim=0) + 1e-6)

        if train_student:
            student_features = self.student(x, return_features=True)
            adapted_features = []
            for i in range(len(student_features)):
                if student_features[i].dim() > 2:
                    student_features[i] = student_features[i].mean([2, 3])
                student_features[i] = self.feature_adapters[i](student_features[i])
                student_features[i] = self.projections[i](student_features[i])
                adapted_features.append(student_features[i])
            student_logits = self.student.classifier(adapted_features[-1])
            return student_logits, adapted_features, teacher_features
        else:
            features = self.student(x, return_features=True)
            features = features[-1]  # Last feature is already pooled in MobileNetV2
            features = self.feature_adapters[-1](features)
            return self.student.classifier(features)

# Custom feature loss
def feature_loss(student_features, teacher_features):
    losses = [F.mse_loss(s, t) for s, t in zip(student_features, teacher_features)]
    return sum(losses) / len(losses)

# Training function
def train_vkd(model, train_loader, optimizer, epoch, device):
    model.train()
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
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item():.4f} (Cls: {cls_loss.item():.4f}, Feat: {feat_loss.item():.4f})')

# Evaluation function
def test_vkd(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, train_student=False)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

# Main training loop
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    teacher = ResNet18(num_classes=10).to(device)
    student = MobileNetV2(num_classes=10).to(device)
    teacher.eval()
    model = LayerWiseVkD(teacher, student, feature_dims=[64, 64, 128, 256, 512]).to(device)
    optimizer = torch.optim.Adam(
        list(model.student.parameters()) +
        list(model.feature_adapters.parameters()) +
        list(model.projections.parameters()),
        lr=0.001
    )
    for epoch in range(1, 11):
        train_vkd(model, train_loader, optimizer, epoch, device)
        test_vkd(model, test_loader, device)

if __name__ == '__main__':
    main()