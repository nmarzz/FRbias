from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision.transforms as transforms
import torch
from sklearn.metrics.pairwise import cosine_similarity


root = 'rfw'
category = 'Asian'
