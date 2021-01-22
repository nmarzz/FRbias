from facenet_pytorch import MTCNN, InceptionResnetV1
from get_file_img import get_random_pair
from models.resnet_ft import resnet50_ft_dag
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
from sklearn.preprocessing import normalize

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Get models
mtcnn = MTCNN(
    image_size=224,
    margin=0,
    device=device,
    selection_method='center_weighted_size'
)

# =================== From Facenet =======================
#model = InceptionResnetV1(pretrained='vggface2').eval()


model = resnet50_ft_dag('resnet50_ft.pth').eval()





res_same = []
res_not = []
start = time.time()
for i in range(1,10):
    # Get images
    img1,img2,path1,path2,same= get_random_pair('rfw','African')
    # noah = Image.open('/Users/nm/Desktop/noah.jpg')
    # bubs = Image.open('/Users/nm/Desktop/bubs.jpg')
    # pops = Image.open('/Users/nm/Desktop/pops.jpg')
    # heath = Image.open('/Users/nm/Desktop/heath.jpg')
    # george = Image.open('/Users/nm/Desktop/george.jpg')
    # tim = Image.open('/Users/nm/Desktop/tim.jpg')
    # quinn = Image.open('/Users/nm/Desktop/quinn.jpg')

    with torch.no_grad():

        # image 1
        #tensor_img = PILtoTensor(img1)
        img_cropped = mtcnn(img1)

        tensor_img = torch.unsqueeze(img_cropped, 0)
        vector0 = model(tensor_img)[1].detach().cpu().numpy()[:, :, 0, 0]
        vector0 = normalize(vector0,norm='l2')

        # image 2
        img_cropped = mtcnn(img2)
        tensor_img = torch.unsqueeze(img_cropped, 0)
        vector1 = model(tensor_img)[1].detach().cpu().numpy()[:, :, 0, 0]
        vector1 = normalize(vector1,norm='l2')





        print(same)
        if same:
            res_same.append(cosine_similarity(vector0,vector1))
        else:
            res_not.append(cosine_similarity(vector0,vector1))
        print(cosine_similarity(vector0,vector1))


res_same = np.array(res_same)
res_not = np.array(res_not)

print('Average result of same: ' + str(np.mean(res_same)))
print('Average result of not: ' + str(np.mean(res_not)))
print('Took {} seconds'.format(time.time() - start))

    # img1.show()
    # img2.show()
