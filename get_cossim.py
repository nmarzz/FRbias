from facenet_pytorch import MTCNN, InceptionResnetV1
from get_file_img import interp_pair
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
import time
from sklearn.metrics.pairwise import cosine_similarity

root = 'rfw'



model = InceptionResnetV1(pretrained='vggface2').eval()
modelName = 'facenet'
model_input_size = (160,160)

#ethnicities = ['Asian','African','Caucasian','Indian']
ethnicities = ['Asian']


PILtoTensor = transforms.Compose([transforms.Resize(model_input_size),transforms.ToTensor()])


start_time = time.time()
num_bad_paths = {'Asian': 0, 'Caucasian': 0,'Indian': 0,'African':0 }
embedding_dict = {}
for ethnic in ethnicities:
    pair_path = os.path.join(root,'txts',ethnic,ethnic + '_pairs.txt')
    data_path = os.path.join(root,'data',ethnic + '_cropped')

    if not (os.path.exists(pair_path) and os.path.exists(data_path)):
        raise ValueError('Could not build data or text paths')

    pairs = open(pair_path,'r')

    for line in pairs:
        path1,path2,issame  = interp_pair(data_path,line)


        try:
            im1 = Image.open(path1)
            im2 = Image.open(path2)

            with torch.no_grad():
                start_time = time.time()
                ten1 = PILtoTensor(im1).unsqueeze(0)
                ten2 = PILtoTensor(im2).unsqueeze(0)

                embedding1 = model(ten1)
                embedding2 = model(ten2)
                
                embedding_dict[path1] = embedding1
                embedding_dict[path2] = embedding2

        except FileNotFoundError:
            num_bad_paths[ethnic] += 1





print('Took {} seconds'.format(time.time() - start_time))
print(embedding_dict)
print(num_bad_paths)
