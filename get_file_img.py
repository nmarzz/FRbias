import os.path
import random
from PIL import Image



def get_random_pair(rfw_root,race):
    '''
    Returns the PIL images of a random pair of faces, taken from a particular ethnic category
    '''

    def random_line(afile):
        line = next(afile)
        for num, aline in enumerate(afile, 2):
            if random.randrange(num):
                continue
            line = aline
        return line

    if not(race == 'Asian' or race == 'African' or race == 'Indian' or race == 'Caucasian'):
        raise ValueError("Invalid race choice: Must be Asian, African, Indian, or Caucasian")


    TEXT_PATH = os.path.join(rfw_root,'txts',race, race + '_pairs.txt')
    DATA_PATH = os.path.join(rfw_root,'data',race + '_cropped')
    print('Retrieving images from: ' + DATA_PATH)

    if not (os.path.exists(TEXT_PATH) and os.path.exists(DATA_PATH)):
        raise ValueError('Could not build data or text paths')

    TEXT = open(TEXT_PATH,'r')

    pair_line = random_line(TEXT)
    path1,path2,root1,root2,ext1,ext2,same = interp_pair(DATA_PATH,pair_line)

    img1 = Image.open(path1)
    img2 = Image.open(path2)

    return img1,img2,path1,path2,same


def interp_pair(data_root,line,dataset = 'rfw'):
    if dataset == 'rfw':
        return interp_pair_rfw(data_root,line)
    elif dataset == 'bfw':
        return interp_pair_bfw(data_root,line)



def interp_pair_rfw(data_root,line):
    key_words = line.split()

    if len(key_words) == 3:
        same = True

        root1 = key_words[0]
        root2 = key_words[0]
        ext1 = key_words[1].zfill(4)
        ext2 = key_words[2].zfill(4)

        path1 = os.path.join(data_root,root1,root1 + '_' + ext1 + '.jpg')
        path2 = os.path.join(data_root,root2,root2 + '_' + ext2 + '.jpg')

    elif len(key_words) == 4:
        same = False
        root1 = key_words[0]
        ext1 = key_words[1].zfill(4)
        root2 = key_words[2]
        ext2 = key_words[3].zfill(4)

        path1 = os.path.join(data_root,root1,root1 + '_' + ext1 + '.jpg')
        path2 = os.path.join(data_root,root2,root2 + '_' + ext2 + '.jpg')

    return path1,path2,root1,root2,ext1,ext2,same



def interp_pair_bfw(data_root,line):
    key_words = line.split(',')
    fold = key_words[0]
    path1 = os.path.join(data_root,key_words[1])
    path2 = os.path.join(data_root,key_words[2])
    same = key_words[3]
    id1 = key_words[4]
    id2 = key_words[5]
    att1 = key_words[6]
    att2 = key_words[7]
    g1 = key_words[13]
    g2 = key_words[14]
    e1 = key_words[15]
    e2 = key_words[16]


    return fold,path1,path2,same,id1,id2,att1,att2,g1,g2,e1,e2

#
# pairs_data = 'pairsdata.csv'
# bfwroot = 'bfw_cropped'
# bfwpath = os.path.join(bfwroot,pairs_data)
#
#
#
#
#
# with open(bfwpath,'r') as file:
#     for i,pairs in enumerate(file):
#
#         if i > 10:
#             break
#         elif i == 0:
#             continue
#
#
#
#         line = pairs.split(',')
#         print(line)
#         imgpath = os.path.join(bfwroot,line[1])
#
#         img = Image.open(imgpath)
#         # if i == 1:
#         #     img.show()
#         #
#         # elif i ==9:
#         #     img.show()
