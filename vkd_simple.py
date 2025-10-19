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
        # Initialize skew-symmetric matrix A for matrix exponential
        self.A = nn.Parameter(torch.randn(in_features, in_features) * 0.01)

    def forward(self, x):
        # Ensure x has shape (batch_size, in_features)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # Flatten spatial dimensions if present
        if x.size(-1) != self.in_features:
            raise ValueError(f"Expected input dimension {self.in_features}, got {x.size(-1)}")

        # Ensure A is skew-symmetric: A = -A^T
        A_skew = 0.5 * (self.A - self.A.transpose(-1, -2))
        # Matrix exponential approximation (Pade approximation, simplified)
        I = torch.eye(self.in_features, device=x.device)
        A2 = torch.matmul(A_skew, A_skew)
        # Simplified Pade approximation: exp(A) ≈ I + A + 0.5 * A^2
        exp_A = I + A_skew + 0.5 * A2
        # Truncate to match output dimensions
        proj = exp_A[:, :self.out_features]
        # Apply projection to input features
        return torch.matmul(x, proj)


# Knowledge Distillation Model
class VkD(nn.Module):
    def __init__(self, teacher, student, feature_dim=512):
        super(VkD, self).__init__()
        self.teacher = teacher
        self.student = student
        # Adjust student features to match teacher feature dimension
        self.student_feature_adapter = nn.Linear(1280, feature_dim)
        self.proj = OrthogonalProjection(feature_dim, feature_dim)
        # Redefine student's classifier to accept feature_dim (512) instead of 1280
        self.student.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feature_dim, student.classifier[1].out_features)  # Match output to number of classes
        )

    def forward(self, x, train_student=True):
        # Get teacher features (from the layer before classifier)
        with torch.no_grad():
            teacher_features = self.teacher.features(x)
            teacher_features = teacher_features.view(teacher_features.size(0), -1)
            # Standardization (task-specific normalization)
            teacher_features = (teacher_features - teacher_features.mean(dim=0)) / (teacher_features.std(dim=0) + 1e-6)

        if train_student:
            # Get student features
            student_features = self.student.features(x)
            student_features = student_features.mean([2, 3])  # Global average pooling
            # Adapt student features to match teacher feature dimension
            student_features = self.student_feature_adapter(student_features)
            # Apply orthogonal projection
            student_features = self.proj(student_features)
            # Get student logits
            student_logits = self.student.classifier(student_features)
            return student_logits, student_features, teacher_features
        else:
            # Inference mode
            features = self.student.features(x)
            features = features.mean([2, 3])
            features = self.student_feature_adapter(features)
            return self.student.classifier(features)


# Custom feature loss
def feature_loss(student_features, teacher_features):
    return F.mse_loss(student_features, teacher_features)


# Training function
def train_vkd(model, train_loader, optimizer, epoch, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        logits, student_features, teacher_features = model(data)

        # Classification loss
        cls_loss = F.cross_entropy(logits, target)
        # Feature distillation loss
        feat_loss = feature_loss(student_features, teacher_features)
        # Combined loss
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
    print(
        f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy


# Modified ResNet-18 and MobileNetV2 for feature extraction
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        base = models.resnet18(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(*list(base.children())[:-1])  # Remove classifier
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
        x = x.mean([2, 3])  # Global average pooling
        x = self.classifier(x)
        return x


# Main training loop
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
    student = MobileNetV2(num_classes=10).to(device)
    teacher.eval()  # Teacher is pre-trained and fixed

    # Initialize VkD model
    model = VkD(teacher, student, feature_dim=512).to(device)

    # Optimizer (optimize both student and projection layer)
    optimizer = torch.optim.Adam(
        list(model.student.parameters()) + list(model.student_feature_adapter.parameters()) + list(
            model.proj.parameters()),
        lr=0.001
    )

    # Training loop
    for epoch in range(1, 11):  # 10 epochs for simplicity
        train_vkd(model, train_loader, optimizer, epoch, device)
        test_vkd(model, test_loader, device)


if __name__ == '__main__':
    main()