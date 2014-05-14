from my_utils import *
from parameters import *
import os
os.system('rm -rf '+pkl_images_path+'.gz')
create_formatted_pkl(ori_images_dir, pkl_images_path, 0.9, ishape)
