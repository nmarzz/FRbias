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

parser = argparse.ArgumentParser(description='Get Embeddings from the Adience Dataset')

parser.add_argument('--data_dir',default = 'adience', metavar='DIR', type=str,
                    help='Root to Adience dataset')
parser.add_argument('--model', default = 'facenet' ,metavar='MOD', type=str,
                    help='Model to use (senet or facenet or sphereface)')

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


if not os.path.exists(embedding_path):
    os.makedirs(embedding_path)



PILtoTensor = transforms.Compose([transforms.Resize(model_input_size),transforms.ToTensor()])
PILtoTensor_flip = transforms.Compose([transforms.Resize(model_input_size),transforms.RandomHorizontalFlip(1),transforms.ToTensor()])


root = 'adience'

info_path = os.path.join(root,'adience_data.csv')

info = open(info_path,'r')

init_info = open(info_path,'r')

for line in init_info:
    path,age,gender = interp_pair(root,line,dataset = 'adience',aligned = True)
    embedding_dict[path] = None

for i,line in enumerate(info):

    if i % 1000 == 0:
        print('Completed {} images'.format(i))

    if i == 0:
        continue

    path,age,gender = interp_pair(root,line,dataset = 'adience',aligned = True)

    img = Image.open(path)


    with torch.no_grad():
        if args.model == 'sphereface':
            ten = PILtoTensor(img).unsqueeze(0).to(device)
            ten_flip = PILtoTensor_flip(img).unsqueeze(0).to(device)

            input = torch.vstack([ten,ten_flip])
            output = model(input)

            embedding = output[0].unsqueeze(0)

        else:

            input = PILtoTensor(img).unsqueeze(0).to(device)
            output = model(input)

            embedding = output[0].unsqueeze(0)


            if args.model == 'senet':
                embedding = torch.linalg.norm(embedding[1],dim = (2,3))


        embedding_dict[path] = embedding.cpu()




# End get embedding loop

dict_filename = os.path.join(embedding_path,'{}_embeddings.pickle'.format(modelName))
with open(dict_filename,'wb+') as dict_file:
    pickle.dump(embedding_dict,dict_file,protocol=pickle.HIGHEST_PROTOCOL)







#
# if args.save_embed:
#     for line in init_dict_pairs:
#         fold,path1,path2,same,id1,id2,att1,att2,g1,g2,e1,e2  = interp_pair(root,line,dataset = 'adience')
#         embedding_dict[path1] = None
#         embedding_dict[path2] = None
