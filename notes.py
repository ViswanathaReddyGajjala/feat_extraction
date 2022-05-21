https://drive.google.com/file/d/1ELvwZYeqdsPmDcX1v7_RbFqbQnvHt9sB/view
import gdown

url = 'https://drive.google.com/uc?id=1ELvwZYeqdsPmDcX1v7_RbFqbQnvHt9sB'
output = 'vivit_fac_enc_b16x2_pt_k700_ft_ek100_32x224x224_4630_public.pyth'
gdown.download(url, output, quiet=False)

https://drive.google.com/file/d/1O0-kPlPrPfrOiFyFcNbCDbnTfsflFc5z/view
url = 'https://drive.google.com/uc?id=1O0-kPlPrPfrOiFyFcNbCDbnTfsflFc5z'
output = 'ek100_localization_vivit_feat_vivit_class_1830.pyth'
gdown.download(url, output, quiet=False)


1ELvwZYeqdsPmDcX1v7_RbFqbQnvHt9sB
url = 'https://drive.google.com/uc?id=1ELvwZYeqdsPmDcX1v7_RbFqbQnvHt9sB'
output = 'ar.pyth'
gdown.download(url, output, quiet=False)

python -W ignore runs/run.py --cfg configs/projects/epic-kitchen-ar/ek100/vivit_fac_enc.yaml 

look into line 53 (datasets/base/epickitech100.py)

/home/viswa/TAdaConv/configs/pool/backbone/vivit_fac_enc.yaml


model_path = 'vivit_fac_enc_b16x2_pt_k700_ft_ek100_32x224x224_4630_public.pyth'
model_path = 'ek100_localization_vivit_feat_vivit_class_1830.pyth'

model_path = 'ar.pyth'
import torch 
w = torch.load(model_path)['model_state']

for k, v in w.items():
    print(k, v.shape)

TODO:
1) weight load
2) 3 crop
3) /255 image


  TRAIN_CROP_SIZE: 320 # 224 in configs/pool/run/training/from_scratch_large.yaml

import cv2
cap = cv2.VideoCapture("video.mp4")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( length )

import cv2
videos_list_file = "epick100_videos_list.txt"
with open(videos_list_file) as f:
    lines = [line.rstrip() for line in f]

videos = []

with open("epick100_videos_list_with_frames.txt", "w") as fp:
    for line in lines:
        # vid, num_frames = line.split(" ")
        # vid = line.split(" ")
        vid = line.split(" ")[0]
        cap = cv2.VideoCapture(vid)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(vid, num_frames)
        vid = vid.split("/")[-1].split(".")[0]
        videos.append((vid, num_frames))
        fp.write(vid + " " + str(num_frames) + "\n")

