import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTFeatureExtractor
from sklearn.metrics import accuracy_score
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_ckpt = "vit_cifar10_finetuned.pth"
dataset_name = "ImageNet"  # CIFAR100 or ImageNet
data_path = "../data"


feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])


if dataset_name == "CIFAR100":
    dataset = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform)
    num_classes = 100
elif dataset_name == "ImageNet":
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    num_classes = 1000

else:
    raise ValueError("Unsupported dataset")

loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)



model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=100  # CIFAR-100
).to(device)


state_dict = torch.load("../vit_cifar10_finetuned.pth")


del state_dict["classifier.weight"]
del state_dict["classifier.bias"]


model.load_state_dict(state_dict, strict=False)

model.eval().cuda()


all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in tqdm(loader, desc=f"Evaluating on {dataset_name}"):
        imgs = imgs.cuda()
        labels = labels.cuda()
        outputs = model(pixel_values=imgs)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu())
        all_labels.extend(labels.cpu())


acc = accuracy_score(all_labels, all_preds)
print(f"Accuracy on {dataset_name}: {acc:.4f}")
