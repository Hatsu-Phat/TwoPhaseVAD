import torch
import torch.nn as nn
import torchvision.models as models


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x)


class FrameModelWrapper(nn.Module):
    def __init__(self, frame_model):
        super().__init__()
        self.frame_model = frame_model

    def forward(self, x):
        if x.ndim != 5:
            raise RuntimeError(f"Expected input [B,T,3,H,W], got {tuple(x.shape)}")
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        logits = self.frame_model(x)
        return logits.reshape(b, t, -1).mean(dim=1)


def _get_torchvision_weights(name, pretrained):
    if not pretrained:
        return None
    try:
        weights_enum = models.get_model_weights(name)
        return weights_enum.DEFAULT
    except Exception:
        return 'DEFAULT'


def _replace_classifier_head(model, num_classes):
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            for i in reversed(range(len(model.classifier))):
                if isinstance(model.classifier[i], nn.Linear):
                    model.classifier[i] = nn.Linear(model.classifier[i].in_features, num_classes)
                    return model
        if isinstance(model.classifier, nn.Linear):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            return model

    if hasattr(model, 'heads'):
        if hasattr(model.heads, 'head') and isinstance(model.heads.head, nn.Linear):
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
            return model
        if isinstance(model.heads, nn.Sequential):
            for i in reversed(range(len(model.heads))):
                if isinstance(model.heads[i], nn.Linear):
                    model.heads[i] = nn.Linear(model.heads[i].in_features, num_classes)
                    return model


    if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    raise RuntimeError('Do not know how to replace classifier head for this torchvision model.')


def _torchvision_model(name, num_classes, pretrained=True):
    weights = _get_torchvision_weights(name, pretrained)

    model = models.get_model(name, weights=weights)
    return _replace_classifier_head(model, num_classes)


def build_phase2_model(name, num_classes=13, pretrained=True):
    name = name.lower()
    if name == 'simple_cnn':
        return FrameModelWrapper(SimpleCNN(num_classes=num_classes))

    supported = {'resnet18','resnet50','efficientnet_b0','efficientnet_b3','vit_b_16','swin_t','convnext_tiny'}
    if name not in supported:
        raise ValueError(f'Unknown phase2 model: {name}. Supported: {sorted(supported)}')
    return FrameModelWrapper(_torchvision_model(name, num_classes=num_classes, pretrained=pretrained))
