import torch
import torch.optim as optim
from MNet import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.transforms import functional as F
from img_set import *
import os
from transFDA import *
logging = open(os.path.join('./','logging.txt'), 'w+')
# Set the device to GPU if available
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" # Use GPU 0 and GPU 1
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
num_epochs = 20

target=Image.open('/root/autodl-tmp/NCT-CRC-HE-100K/STR/STR-AADLAKYL.tif')
target = torch.from_numpy(np.array(target))
target = target.permute(2, 0, 1).unsqueeze(0)


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
# Load the dataset and split it into training and validation sets
trainset = ImgSet(root='/root/autodl-tmp/NCT-CRC-HE-100K-NONORM',transform_list=[train_transform])
print(len(trainset))
train_dataset, val_dataset = torch.utils.data.random_split(trainset, [int(len(trainset)*0.8), len(trainset)-int(len(trainset)*0.8)])

# Define the data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, pin_memory=False)

# Define the UnsupervisedTransformerUNet
unsupervised_transformer = UnsupervisedTransformerUNet(in_channels=3,out_channels=3,num_layers=8,num_heads=8,dim_feedforward=128)

# Wrap the UnsupervisedTransformerUNet in a DataParallel wrapper
# unsupervised_transformer = torch.nn.DataParallel(unsupervised_transformer)
unsupervised_transformer=unsupervised_transformer.to(device)

# Define the reconstruction loss functions
# def image_reconstruction_loss(x_recon, x):
#     return F.mse_loss(x_recon, x)

# def mask_reconstruction_loss(mask_recon, mask):
#     return F.binary_cross_entropy(mask_recon, mask)

loss_fn=Fourier_loss()

def cycle_consistency_loss(x, x_recon, mask, mask_recon):
    image_loss = F.mse_loss(x, x_recon)
    mask_loss = F.binary_cross_entropy(mask_recon, mask)
    return image_loss + mask_loss
# Define the optimizer
optimizer = torch.optim.Adam(unsupervised_transformer.parameters(), lr=1e-3)

# Train the UnsupervisedTransformerUNet
torch.cuda.empty_cache()
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * len(f'Epoch {epoch+1}/{num_epochs}'))
    logging.write(f'Epoch {epoch+1}/{num_epochs}')
    train_loss = 0.0
    val_loss = 0.0
    unsupervised_transformer.train()
    for x, _ in train_loader:
        x = x.squeeze(1).to(device)
        print(x.shape)
        # mask = mask.to(device)
        x_recon, mask_recon = unsupervised_transformer(x)
        loss=Fourier_loss(mask_recon, x.permute(2, 0, 1).unsqueeze(0), x_recon, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    unsupervised_transformer.eval()
    with torch.no_grad():
        for x, mask in val_loader:
            
            x = x.to(device)
            mask = mask.to(device)
            x_recon, mask_recon = unsupervised_transformer(x)
            loss=Fourier_loss(mask_recon, x, x_recon, target)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}')
    logging.write(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}')
# Save the trained model
torch.save(unsupervised_transformer.state_dict(), 'unsupervised_transformer.pth')

