import torch
import torchvision
from torchvision import datasets, transforms


kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
dataset = torchvision.datasets.ImageFolder('rfw_data_unpacked_cropped',transform=transforms.Compose([
    transforms.Resize((244,244)),transforms.ToTensor()]))



loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=10,
    shuffle=False
)


mean = 0.
std = 0.
nb_samples = 0.
for data,target in loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples


print('Mean: {}'.format(mean))
print('STD: {}'.format(std))
