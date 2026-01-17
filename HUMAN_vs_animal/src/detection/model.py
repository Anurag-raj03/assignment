import torch
import torchvision

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_detector():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT"
    )
    model.to(DEVICE)
    model.eval()
    return model
