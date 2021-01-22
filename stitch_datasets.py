import os
import argparse

parser = argparse.ArgumentParser(description='Stitch together the data from \'get_cossim\'')

parser.add_argument('-data_dir',default = 'rfw', metavar='DIR', type=str,
                    help='Root to RFW dataset')
parser.add_argument('-modelName', default = 'senet' ,metavar='MOD', type=str,
                    help='Name of model to use (facenet or sphereface)')
args = parser.parse_args()

root = args.data_dir

data_folder = 'cossim_data'
modelName = args.modelName
stitched_file = '{}_cossim.csv'.format(modelName)



path = os.path.join(root,data_folder)

ethnicities = ['Asian','African','Caucasian','Indian']

for i,ethnic in enumerate(ethnicities):
    data_filename = '{}_{}_cossim.csv'.format(ethnic,modelName)
    data_path = os.path.join(path,data_filename)

    stitched_file_path = os.path.join(path,stitched_file)

    with open(stitched_file_path,'a+') as write_file:

        with open(data_path,'r') as read_file:
            data = read_file.readlines()

            if i == 0:
                write_file.writelines(data)
            else:
                write_file.writelines(data[1:])
