from facenet_pytorch import MTCNN, InceptionResnetV1
from get_file_img import get_random_pair
import torchvision.transforms as transforms
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Get models
mtcnn = MTCNN(
    image_size=160,
    margin=14,
    device=device,
    selection_method='center_weighted_size'
)
model = InceptionResnetV1(pretrained='vggface2').eval()

for i in range(1,10):
    # Get images
    img1,img2,same = get_random_pair('rfw','Indian')
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
        vector0 = model(tensor_img)


        # image 2
        img_cropped = mtcnn(img2)
        tensor_img = torch.unsqueeze(img_cropped, 0)
        vector1 = model(tensor_img)

        print(same)
        print(cosine_similarity(vector0,vector1))


    # img1.show()
    # img2.show()
    #
    # with torch.no_grad():
    #     noah_tens = mtcnn(noah)
    #     pops_tens = mtcnn(pops)
    #     bubs_tens = mtcnn(bubs)
    #     heath_tens = mtcnn(heath)
    #     tim_tens = mtcnn(tim)
    #     quinn_tens = mtcnn(quinn)
    #
    #     vec_noah = model(noah_tens.unsqueeze(0))
    #     vec_pops = model(pops_tens.unsqueeze(0))
    #     vec_bubs = model(bubs_tens.unsqueeze(0))
    #     vec_heath = model(heath_tens.unsqueeze(0))
    #     vec_tim = model(tim_tens.unsqueeze(0))
    #     vec_quinn = model(quinn_tens.unsqueeze(0))
    #
    # print('Noah and Dad' + str(cosine_similarity(vec_noah,vec_pops)))
    # print('Noah and Heath' + str(cosine_similarity(vec_noah,vec_heath)))
    # print('Noah and Bubs' + str(cosine_similarity(vec_noah,vec_bubs)))
    # print('Heath and Dad' + str(cosine_similarity(vec_heath,vec_pops)))
    # print('Noah and Heath' + str(cosine_similarity(vec_noah,vec_heath)))
    # print('Tim and Heath' + str(cosine_similarity(vec_tim,vec_heath)))
    # print('Tim and Pops' + str(cosine_similarity(vec_tim,vec_pops)))
    # print('Tim and Noah' + str(cosine_similarity(vec_tim,vec_noah)))
    # print('Quinn and Noah' + str(cosine_similarity(vec_quinn,vec_noah)))
    # print('Pops and Quinn' + str(cosine_similarity(vec_pops,vec_quinn)))
    # print('Heath and Quinn' + str(cosine_similarity(vec_heath,vec_quinn)))
    # print('Quinn and Bubs' + str(cosine_similarity(vec_bubs,vec_quinn)))
