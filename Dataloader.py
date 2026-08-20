import torch
import torch.utils.data as data
import torchvision as tv


class TTIDataset(data.Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.transform = tv.transforms.Compose([
            tv.transforms.Resize((64, 64)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        image = self.transform(item["image"])
        text = self.tokenizer.encode_ids(
            item["blip_caption"]
        )
        return {
            "image": image,
            "input_ids": torch.tensor(
                text["input_ids"],
                dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                text["attention_mask"],
                dtype=torch.long
            ),
            "label": torch.tensor(
                item["label"],
                dtype=torch.long
            )
        }