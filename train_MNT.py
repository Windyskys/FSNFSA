import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from MNetTimm import *
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from img_set import *
import os
from transFDA import *
from tqdm import tqdm
logging = open(os.path.join('./','logging.txt'), 'w+')
# Set the device to GPU if available
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # Use GPU 0 and GPU 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 20

target=Image.open('/root/autodl-tmp/CRC-VAL-HE-7K/STR/STR-TCGA-ANKSFAYN.tif')
target = torch.from_numpy(np.array(target))
target = target.permute(2, 0, 1).unsqueeze(0)
target = target.reshape(1,3,224,224)
target = target.float().requires_grad_()

# Define the transforms for the training and validation sets
train_transform=transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

test_transform=transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
# Hyperparameters
num_epochs = 100
batch_size = 32
lr = 1e-4

# Load the dataset and split it into training and validation sets
trainset = ImgSet(root='/root/autodl-tmp/NCT-CRC-HE-100K-NONORM',transform_list=[train_transform])
train_dataset, val_dataset = torch.utils.data.random_split(trainset, [int(len(trainset)*0.8), len(trainset)-int(len(trainset)*0.8)])

# Define the data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=False)
# Initialize the model
model = TFUNet(3, 3)
model.to(device)
# Define the loss function
criterion = Fourier_loss()

# Use the Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train the model
torch.cuda.empty_cache()
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * len(f'Epoch {epoch+1}/{num_epochs}'))
    logging.write(f'Epoch {epoch+1}/{num_epochs}')
    train_loss = 0.0
    val_loss = 0.0
    best_loss=10000000000
    for images, targets in tqdm(train_loader):
        # Forward pass
        outputs = model(images.squeeze(1).to(device))
        train_loss = criterion(outputs.detach().cpu().numpy(), images, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss.item()))
    model.eval()
    with torch.no_grad():
        for x, _ in val_loader:
            mask= model(x.squeeze(1).to(device))
            val_loss=criterion(mask.detach().cpu().numpy(), x, target)
            val_loss += val_loss.item()
        print('Epoch [{}/{}], Val Loss: {:.4f}'.format(epoch+1, num_epochs, val_loss))
    logging.write('Epoch [{}/{}], Val Loss: {:.4f}'.format(epoch+1, num_epochs, val_loss))
    if val_loss<best_loss:
        best_loss=val_loss
        torch.save(model.state_dict(), 'best_model.pth')
# Save the trained model
torch.save(model.state_dict(), 'model.pth')
