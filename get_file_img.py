import os.path
import random
from PIL import Image

root = 'rfw'

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

    def interp_pair(data_root,line):
        key_words = line.split()

        if len(key_words) == 3:
            same = True

            root = key_words[0]
            ext1 = key_words[1].zfill(4)
            ext2 = key_words[2].zfill(4)

            path1 = os.path.join(data_root,root,root + '_' + ext1 + '.jpg')
            path2 = os.path.join(data_root,root,root + '_' + ext2 + '.jpg')

        elif len(key_words) == 4:
            same = False
            root1 = key_words[0]
            ext1 = key_words[1].zfill(4)
            root2 = key_words[2]
            ext2 = key_words[3].zfill(4)

            path1 = os.path.join(data_root,root1,root1 + '_' + ext1 + '.jpg')
            path2 = os.path.join(data_root,root2,root2 + '_' + ext2 + '.jpg')

        return path1,path2,same

    if not(race == 'Asian' or race == 'African' or race == 'Indian' or race == 'Caucasian'):
        raise ValueError("Invalid race choice: Must be Asian, African, Indian, or Caucasian")


    TEXT_PATH = os.path.join(root,'txts',race, race + '_pairs.txt')
    DATA_PATH = os.path.join(root,'data',race + '_cropped')

    if not (os.path.exists(TEXT_PATH) and os.path.exists(DATA_PATH)):
        raise ValueError('Could not build data or text paths')

    TEXT = open(TEXT_PATH,'r')

    pair_line = random_line(TEXT)
    path1,path2,same = interp_pair(DATA_PATH,pair_line)

    img1 = Image.open(path1)
    img2 = Image.open(path2)

    return img1,img2,same
