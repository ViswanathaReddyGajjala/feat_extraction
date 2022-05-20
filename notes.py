https://drive.google.com/file/d/1ELvwZYeqdsPmDcX1v7_RbFqbQnvHt9sB/view
import gdown

url = 'https://drive.google.com/uc?id=1ELvwZYeqdsPmDcX1v7_RbFqbQnvHt9sB'
output = 'vivit_fac_enc_b16x2_pt_k700_ft_ek100_32x224x224_4630_public.pyth'
gdown.download(url, output, quiet=False)

https://drive.google.com/file/d/1O0-kPlPrPfrOiFyFcNbCDbnTfsflFc5z/view
url = 'https://drive.google.com/uc?id=1O0-kPlPrPfrOiFyFcNbCDbnTfsflFc5z'
output = 'ek100_localization_vivit_feat_vivit_class_1830.pyth'
gdown.download(url, output, quiet=False)

python -W ignore runs/run.py --cfg configs/projects/epic-kitchen-ar/ek100/vivit_fac_enc.yaml 

look into line 53 (datasets/base/epickitech100.py)