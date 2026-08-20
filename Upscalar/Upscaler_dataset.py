from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as T

class UpscalerDataset(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.hr_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.lr_transform = T.Compose([
            T.Resize((64, 64), T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index]["image"].convert("RGB")
        hr = self.hr_transform(image)
        lr = self.lr_transform(image)
        return {"lr": lr, "hr": hr}