from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vit_pytorch import ViT
import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score

# CIFAR-10
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ViT
vit = ViT(
    image_size = 32,
    patch_size = 4,
    num_classes = 10,
    dim = 512,
    depth = 6,
    heads = 8,
    mlp_dim = 1024,
    dropout = 0.1,
    emb_dropout = 0.1
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vit = vit.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vit.parameters(), lr=3e-4)

# train
for epoch in range(50):
    vit.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = vit(imgs)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

def evaluate(model, test_loader, name):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu()
            all_preds.extend(preds)
            all_labels.extend(labels)
    acc = accuracy_score(all_labels, all_preds)
    print(f"[Eval] {name} Accuracy: {acc:.4f}")

# CIFAR-10
evaluate(vit, test_loader, "CIFAR-10")

# CIFAR-100
cifar100_test = CIFAR100(root='./data', train=False, download=True, transform=transform)
cifar100_loader = DataLoader(cifar100_test, batch_size=64, shuffle=False)
vit.head = nn.Linear(512, 100).to(device)
evaluate(vit, cifar100_loader, "CIFAR-100")

# ImageNet-1k
try:
    from torchvision.datasets import ImageNet
    imagenet_test = ImageNet(root='./imagenet', split='val', transform=transform)
    imagenet_loader = DataLoader(imagenet_test, batch_size=64, shuffle=False)
    vit.head = nn.Linear(512, 1000).to(device)  # ImageNet head
    evaluate(vit, imagenet_loader, "ImageNet-1k")
except:
    print("ImageNet-1k fail。")

# VTAB
try:
    from datasets import load_dataset
    from PIL import Image

    dataset = load_dataset("caltech101", split="test")

    def transform_vtab(example):
        img = transform(example['image'])
        label = example['label']
        return img, label

    vtab_imgs, vtab_labels = zip(*[transform_vtab(x) for x in dataset])
    vtab_imgs = torch.stack(vtab_imgs)
    vtab_labels = torch.tensor(vtab_labels)
    vit.head = nn.Linear(512, 102).to(device)  # Caltech101 102class

    vit.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(vtab_imgs), 64):
            batch_imgs = vtab_imgs[i:i+64].to(device)
            batch_preds = vit(batch_imgs).argmax(dim=1).cpu()
            preds.extend(batch_preds)

    acc = accuracy_score(vtab_labels, preds)
    print(f"[Eval] VTAB-Caltech101 Accuracy: {acc:.4f}")
except:
    print("VTAB-Caltech101 fiail。")