import argparse
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


parser = argparse.ArgumentParser(description='Get Cosine Similarity scores from model and RFW dataset')

parser.add_argument('-data_dir',default = 'rfw', metavar='DIR', type=str,
                    help='Root to RFW dataset')
parser.add_argument('-model', default = 'senet' ,metavar='MOD', type=str,
                    help='Model to use (facenet or sphereface)')
parser.add_argument('-save-embed', default = False ,metavar='SVEMB', type=bool)
args = parser.parse_args()

root = args.data_dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Model on ' + str(device))

# Define the model

if args.model == 'facenet':
# ================ code for facenet ===========================
    model = InceptionResnetV1(pretrained='vggface2').to(device).eval()
    modelName = 'facenet'
    model_input_size = (160,160)
elif args.model == 'sphereface':

# ================ code for sphereface ===========================
    import models.net_sphere
    model = getattr(models.net_sphere,'sphere20a')()
    model.load_state_dict(torch.load('sphereface.pth'))
    model.to(device).eval()
    modelName = 'sphereface_112-96'
    model_input_size = (112,96)

elif args.model == 'senet':
    from models.sennet_VGG import senet50_scratch_dag
    model = senet50_scratch_dag('senet50_scratch_dag.pth').to(device).eval()
    modelName = 'senet'
    model_input_size = (244,244)
else:
    raise ValueError('Invalid model choice')


ethnicities = ['Asian','African','Caucasian','Indian']

embedding_path = os.path.join(root,'embeddings')
cossim_path = os.path.join(root,'cossim_data')
if not os.path.exists(embedding_path):
    os.makedirs(embedding_path)

if not os.path.exists(cossim_path):
    os.makedirs(cossim_path)

PILtoTensor = transforms.Compose([transforms.Resize(model_input_size),transforms.ToTensor()])
PILtoTensor_flip = transforms.Compose([transforms.Resize(model_input_size),transforms.RandomHorizontalFlip(1),transforms.ToTensor()])


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

                    if args.model == 'sphereface':
                        ten1 = PILtoTensor(im1).unsqueeze(0).to(device)
                        ten1_flip = PILtoTensor_flip(im1).unsqueeze(0).to(device)
                        ten2 = PILtoTensor(im2).unsqueeze(0).to(device)
                        ten2_flip = PILtoTensor_flip(im1).unsqueeze(0).to(device)

                        input = torch.vstack([ten1,ten1_flip,ten2,ten2_flip])
                        output = model(input)

                        embedding1 = output[0].unsqueeze(0)
                        embedding = output[2].unsqueeze(0)

                    else:

                        ten1 = PILtoTensor(im1).unsqueeze(0).to(device)
                        ten2 = PILtoTensor(im2).unsqueeze(0).to(device)

                        embedding1 = model(ten1)
                        embedding2 = model(ten2)

                        if args.model == 'senet':
                            embedding1 = torch.linalg.norm(embedding1[1],dim = (2,3))
                            embedding2 = torch.linalg.norm(embedding2[1],dim = (2,3))

                        if args.save_embed:
                            embedding_dict[path1] = embedding1
                            embedding_dict[path2] = embedding2

                    cosine_sim = cosine_similarity(embedding1.cpu(),embedding2.cpu())

                    csv_line = '{},{},{},{},{},{},{}\n'.format(ethnic,id1,pic1,id2,pic2,issame,cosine_sim.item())
                    data_file.write(csv_line)

            except FileNotFoundError:
                num_bad_paths[ethnic] += 1


        dict_filename = os.path.join(embedding_path,'{}_{}_embeddings.pickle'.format(ethnic,modelName))
        csv_filename = '{}_{}_cossim'.format(ethnic,modelName)

        if args.save_embed:
            with open(dict_filename,'wb+') as dict_file:
                pickle.dump(embedding_dict,dict_file,protocol=pickle.HIGHEST_PROTOCOL)






print('Took {} seconds'.format(time.time() - start_time))
print(num_bad_paths)


with open(os.path.join(cossim_path,'bad_pairs.txt'),'w+') as badpairs:
    print(num_bad_paths,file=badpairs)
