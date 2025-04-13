import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from vit_patch_vit import VisionTransformer as ViT

image_size = 32
patch_size = 4
num_classes = 10
dim = 384
depth = 12
heads = 6
mlp_dim = 768
epochs = 20
batch_size = 64
lr = 3e-4

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT(
    img_size=image_size,
    patch_size=patch_size,
    num_classes=num_classes,
    embed_dim=dim,
    depth=depth,
    num_heads=heads,
    mlp_dim=mlp_dim,
    in_channels=3
).to(device)

# loss optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

# train
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, labels = images.to(device), labels.to(device)
        preds = model(images)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# test
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        preds = model(images).argmax(dim=1).cpu()
        all_preds.extend(preds)
        all_labels.extend(labels)

acc = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {acc:.4f}")

# save
torch.save(model.state_dict(), "vit_custom_cifar10.pth")
print("模型已保存到 vit_custom_cifar10.pth")