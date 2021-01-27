from facenet_pytorch import MTCNN, InceptionResnetV1
from get_file_img import get_random_pair
from models.resnet_ft import resnet50_ft_dag
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np
import models.net_sphere as net_sphere
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



model = getattr(net_sphere,'sphere20a')()
weights = torch.load('sphereface.pth')
model.load_state_dict(weights)
model.eval()

img1,img2,path1,path2,same = get_random_pair('rfw','Caucasian')

resize = transforms.Compose([
    transforms.Resize((112,96)),transforms.ToTensor()])

resize_flip = transforms.Compose([
    transforms.Resize((112,96)),transforms.RandomHorizontalFlip(1),transforms.ToTensor()])

toPIL = transforms.ToPILImage()





res_same = []
res_not = []
start = time.time()
for i in range(1,30):
    # Get images
    img1,img2,path1,path2,same= get_random_pair('rfw','Caucasian')
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
        # img_cropped = mtcnn(img1)


        # image 1
        # #img_cropped = mtcnn(img1)
        # tensor_img = resize(img1)
        # tensor_img = torch.unsqueeze(tensor_img, 0)
        # tensor_img_flip = resize_flip(img1).unsqueeze(0)
        # feat_flipped = model(tensor_img_flip)
        # # img_flip = toPIL(tensor_img_flip.squeeze()).show()
        # vector0 = (model(tensor_img) + feat_flipped)/2
        # #vector0 = torch.cat((vector0,feat_flipped),1)
        #
        #
        # # image 2
        # #img_cropped = mtcnn(img2)
        # tensor_img = resize(img2)
        # tensor_img = torch.unsqueeze(tensor_img, 0)
        # tensor_img_flip = resize_flip(img2).unsqueeze(0)
        # feat_flipped = model(tensor_img_flip)
        # # img_flip = toPIL(tensor_img_flip.squeeze()).show()
        # vector1 = (model(tensor_img) + feat_flipped)/2


        # image 1
        #img_cropped = mtcnn(img1)
        tensor_img1 = resize(img1).unsqueeze(0)
        tensor_img_flip1 = resize_flip(img1).unsqueeze(0)

        tensor_img2 = resize(img2).unsqueeze(0)
        tensor_img_flip2 = resize_flip(img2).unsqueeze(0)
        # input = torch.vstack([tensor_img,tensor_img_flip])
        input = torch.vstack([tensor_img1,tensor_img_flip1,tensor_img2,tensor_img_flip2])

        # feat_flipped = model(tensor_img_flip)
        # img_flip = toPIL(tensor_img_flip.squeeze()).show()
        output = model(input)
        #vector0 = torch.cat((vector0,feat_flipped),1)


        f1 = output[0]
        f2 = output[2]

        cosdistance = f1.dot(f2)/(f1.norm()*f2.norm()+1e-5)
        print(cosdistance)



        print(same)
        if same:
            # res_same.append(cosine_similarity(vector0,vector1))
            res_same.append(cosdistance)
        else:
            res_not.append(cosdistance)
        print(cosdistance)


res_same = np.array(res_same)
res_not = np.array(res_not)

print('Average result of same: ' + str(np.mean(res_same)))
print('Average result of not: ' + str(np.mean(res_not)))
print('Took {} seconds'.format(time.time() - start))

    # img1.show()
    # img2.show()
