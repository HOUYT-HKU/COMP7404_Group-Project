import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from sklearn.metrics import accuracy_score


data_choice = 'CIFAR10'  #'CIFAR10', 'CIFAR100', 'ImageNet', 'VTAB'
batch_size = 64
epochs = 10
lr = 3e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if data_choice in ['CIFAR10', 'CIFAR100']:
    image_size = 224
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
else:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


if data_choice == 'CIFAR10':
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    num_classes = 10
elif data_choice == 'CIFAR100':
    train_set = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    num_classes = 100
elif data_choice == 'ImageNet':
    train_set = datasets.ImageNet(root='./imagenet', split='train', transform=transform)
    test_set = datasets.ImageNet(root='./imagenet', split='val', transform=transform)
    num_classes = 1000
elif data_choice == 'VTAB':
    raise NotImplementedError("VTAB loading not implemented. Consider TFDS or HuggingFace datasets.")
else:
    raise ValueError("Unsupported dataset")

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


model = resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).cpu()
        all_preds.extend(preds)
        all_labels.extend(labels)

acc = accuracy_score(all_labels, all_preds)
print(f"Final Test Accuracy on {data_choice}: {acc:.4f}")
