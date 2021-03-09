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


parser = argparse.ArgumentParser(description='Get Cosine Similarity scores from model and BFW dataset')

parser.add_argument('--data_dir',default = 'bfw_cropped', metavar='DIR', type=str,
                    help='Root to BFW dataset')
parser.add_argument('--model', default = 'facenet' ,metavar='MOD', type=str,
                    help='Model to use (senet or facenet or sphereface)')
parser.add_argument('--save-embed', default = False ,metavar='SVEMB', type=bool)
args = parser.parse_args()


for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')



root = args.data_dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Model on ' + str(device))

# Define the model

if args.model == 'facenet':
# ================ code for facenet ===========================
    model = InceptionResnetV1(pretrained='vggface2').to(device).eval()
    modelName = 'facenet'
    model_input_size = (160,160)

elif args.model == 'facenet-webface':
# ================ code for facenet ===========================
    model = InceptionResnetV1(pretrained='casia-webface').to(device).eval()
    modelName = 'facenet-webface'
    model_input_size = (160,160)

elif args.model == 'sphereface':

# ================ code for sphereface ===========================
    import models.net_sphere
    model = getattr(models.net_sphere,'sphere20a')()
    model.load_state_dict(torch.load('sphereface.pth'))
    model.to(device).eval()
    modelName = 'sphereface'
    model_input_size = (112,96)

elif args.model == 'senet':
    from models.sennet_VGG import senet50_scratch_dag
    model = senet50_scratch_dag('senet50_scratch_dag.pth').to(device).eval()
    modelName = 'senet'
    model_input_size = (244,244)
else:
    raise ValueError('Invalid model choice')


# ethnicities = ['Asian','African','Caucasian','Indian']

embedding_path = os.path.join(root,'embeddings')
embedding_dict = {}

cossim_path = os.path.join(root,'cossim_data')
if not os.path.exists(embedding_path):
    os.makedirs(embedding_path)

if not os.path.exists(cossim_path):
    os.makedirs(cossim_path)

PILtoTensor = transforms.Compose([transforms.Resize(model_input_size),transforms.ToTensor()])
PILtoTensor_flip = transforms.Compose([transforms.Resize(model_input_size),transforms.RandomHorizontalFlip(1),transforms.ToTensor()])

num_bad_paths = {'asian_females': 0, 'asian_males': 0,'black_females': 0,'black_males':0,'indian_females':0,'indian_males':0 ,'white_females':0,'white_males':0}

pair_path = os.path.join(root,'pairsdata.csv')

data_file_name = os.path.join(cossim_path,'BFWdata_{}.csv'.format(modelName))

pairs = open(pair_path,'r')

init_dict_pairs = open(pair_path,'r')

# initialize the embedding dictionary with appropriate keys
if args.save_embed:
    for line in pairs:
        fold,path1,path2,same,id1,id2,att1,att2,g1,g2,e1,e2  = interp_pair(root,line,dataset = 'bfw')
        embedding_dict[path1] = None
        embedding_dict[path2] = None



start_time = time.time()

with open(data_file_name,'w+',buffering=1024) as data_file:
    data_file.write('fold,path1,path2,same,id1,id2,att1,att2,g1,g2,e1,e2,{}\n'.format(modelName))

    for i,line in enumerate(pairs):

        if i == 0:
            continue

        if i % 1000 == 0:
            print('Completed {} pairs'.format(i))


        fold,path1,path2,same,id1,id2,att1,att2,g1,g2,e1,e2  = interp_pair(root,line,dataset = 'bfw')

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
                    embedding2 = output[2].unsqueeze(0)

                else:

                    ten1 = PILtoTensor(im1).unsqueeze(0).to(device)
                    ten2 = PILtoTensor(im2).unsqueeze(0).to(device)

                    input = torch.vstack([ten1,ten2])
                    output = model(input)

                    embedding1 = output[0].unsqueeze(0)
                    embedding2 = output[1].unsqueeze(0)

                    if args.model == 'senet':
                        embedding1 = torch.linalg.norm(embedding1[1],dim = (2,3))
                        embedding2 = torch.linalg.norm(embedding2[1],dim = (2,3))

                    if args.save_embed:
                        embedding_dict[path1] = embedding1
                        embedding_dict[path2] = embedding2

                cosine_sim = cosine_similarity(embedding1.cpu(),embedding2.cpu())

                csv_line = '{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(fold,path1,path2,same,id1,id2,att1,att2,g1,g2,e1,e2,cosine_sim.item())
                data_file.write(csv_line)

        except FileNotFoundError:
            num_bad_paths[att1] += 1
            num_bad_paths[att2] += 1




    # csv_filename = '{}_{}_cossim'.format(ethnic,modelName)

    if args.save_embed:
        with open(dict_filename,'wb+') as dict_file:
            pickle.dump(embedding_dict,dict_file,protocol=pickle.HIGHEST_PROTOCOL)






print('Took {} seconds'.format(time.time() - start_time))
print(num_bad_paths)


with open(os.path.join(cossim_path,'bad_pairs.txt'),'w+') as badpairs:
    print(num_bad_paths,file=badpairs)
