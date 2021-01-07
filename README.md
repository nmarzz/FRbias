# Building a Dataset exploring Bias in Face Recognition 

## About
When performing facial recognition (FR) tasks, one often uses the cosine similarity score between two faces in order to predict whether the presented pair is positive or negative. 
It has been observed that models are typically better at recognizing caucasian faces.

In order to address this bias, the dataset that this code builds contains the cosine similarity scores of pairs of Asian, African, Caucasian, and Indian faces taken from the [Racial Faces in the Wild](http://www.whdeng.cn/RFW/index.html) dataset. 



## Models

- We use the Inception Resnet trained on VGGFace2 from the Pytorch Facenet port by Tim Esler [Github](https://github.com/timesler/facenet-pytorch#pretrained-models). 
- The [Pytorch Implementation](https://github.com/clcarwin/sphereface_pytorch) of the Sphereface model is also used. One must import the weights and model from the repo to be used with this code.








