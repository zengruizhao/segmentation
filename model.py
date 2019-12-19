import segmentation_models_pytorch as smp
import torch
from torchsummary import summary


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = smp.Unet(classes=2).to(device)
    # summary(model, (3, 256, 256))
    print(model)