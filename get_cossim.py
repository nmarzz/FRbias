from facenet_pytorch import MTCNN, InceptionResnetV1
from get_file_img import interp_pair
import os.path
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
import time
import pickle
from sklearn.metrics.pairwise import cosine_similarity

root = 'rfw'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Model on ' + str(device))

# Define the model

# ================ code for facenet ===========================
# model = InceptionResnetV1(pretrained='vggface2').to(device).eval()
# modelName = 'facenet'
# model_input_size = (160,160)
# ================================================================

# ================ code for sphereface ===========================
import models.net_sphere
model = getattr(models.net_sphere,'sphere20a')()
model.load_state_dict(torch.load('sphereface.pth'))
model.to(device).eval()
modelName = 'sphereface'
model_input_size = (96,112)
# ================================================================


ethnicities = ['Asian','African','Caucasian','Indian']

embedding_path = os.path.join(root,'embeddings')
cossim_path = os.path.join(root,'cossim_data')
if not os.path.exists(embedding_path):
    os.makedirs(embedding_path)

if not os.path.exists(cossim_path):
    os.makedirs(cossim_path)

PILtoTensor = transforms.Compose([transforms.Resize(model_input_size),transforms.ToTensor()])


start_time = time.time()
num_bad_paths = {'Asian': 0, 'Caucasian': 0,'Indian': 0,'African':0 }

for ethnic in ethnicities:

    print('*' * 30)
    print('Evaluating {} '.format(ethnic))
    print('*' * 30)

    data_file_name = os.path.join(cossim_path,'{}_{}_cossim.csv'.format(ethnic,modelName))
    with open(data_file_name,'w+') as data_file:
        data_file.write('ethnicity,id1,num1,id2,num2,same,{} \n'.format(modelName))
        embedding_dict = {}
        pair_path = os.path.join(root,'txts',ethnic,ethnic + '_pairs.txt')
        data_path = os.path.join(root,'data',ethnic + '_cropped')

        if not (os.path.exists(pair_path) and os.path.exists(data_path)):
            raise ValueError('Could not build data or text paths')

        pairs = open(pair_path,'r')

        for i,line in enumerate(pairs):

            if i % 50 == 0: # temporary for testing
                print('Completed {} {} pairs'.format(i,ethnic))


            path1,path2,id1,id2,pic1,pic2,issame  = interp_pair(data_path,line)

            try:
                im1 = Image.open(path1)
                im2 = Image.open(path2)

                with torch.no_grad():
                    start_time = time.time()
                    ten1 = PILtoTensor(im1).unsqueeze(0).to(device)
                    ten2 = PILtoTensor(im2).unsqueeze(0).to(device)

                    embedding1 = model(ten1)
                    embedding2 = model(ten2)

                    embedding_dict[path1] = embedding1
                    embedding_dict[path2] = embedding2

                    cosine_sim = cosine_similarity(embedding1.cpu(),embedding2.cpu())

                    csv_line = '{},{},{},{},{},{},{} \n'.format(ethnic,id1,pic1,id2,pic2,issame,cosine_sim.item())
                    data_file.write(csv_line)

            except FileNotFoundError:
                num_bad_paths[ethnic] += 1


        dict_filename = os.path.join(embedding_path,'{}_{}_embeddings.pickle'.format(ethnic,modelName))
        csv_filename = '{}_{}_cossim'.format(ethnic,modelName)

        with open(dict_filename,'wb+') as dict_file:
            pickle.dump(embedding_dict,dict_file,protocol=pickle.HIGHEST_PROTOCOL)






print('Took {} seconds'.format(time.time() - start_time))
print(num_bad_paths)


with open(os.path.join(cossim_path,'bad_pairs.txt'),'w+') as badpairs:
    print(num_bad_paths,file=badpairs)
