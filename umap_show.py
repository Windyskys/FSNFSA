from transFDA import *
import torch.utils.data as data
from img_set import *
import torch
from torchvision import transforms
import umap
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
# Define your PyTorch dataset
my_dataset = ImgSet(root='/root/autodl-tmp/NCT-CRC-HE-100K-NONORM',transform_list=[train_transform])
my_dataset, val_dataset = torch.utils.data.random_split(my_dataset, [int(len(my_dataset)*0.1), len(my_dataset)-int(len(my_dataset)*0.1)])
# Define a PyTorch dataloader
dataloader = data.DataLoader(my_dataset, batch_size=128, shuffle=False, pin_memory=False)
# Extract the data from the PyTorch dataset
data_list = []
element = 3*224*224
targets=[]
for data_batch,target in dataloader:
    data_list.append(torch.reshape(data_batch,(data_batch.shape[0],element)).numpy())
    targets.extend(list(target.numpy()))
data_array = np.concatenate(data_list, axis=0)

# Use UMAP to reduce the dimensionality of the data

umap_result = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='correlation').fit_transform(data_array)
print(umap_result.shape)
# Visualize the results
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
cmap = mpl_colors.ListedColormap(['red', 'green', 'blue', 'purple','pink','yellow','orange','teal','brown'])

plt.scatter(umap_result[:, 0], umap_result[:, 1], c=targets, cmap=cmap, s=2,alpha=0.5)
plt.title('UMAP projection of the Baseline',fontsize=20)

cb=plt.colorbar(ticks=range(9))
cb.ax.set_yticklabels(['0','1','2','3','4','5','6','7','8'])

plt.savefig('./umap/baseline_umap.png')
