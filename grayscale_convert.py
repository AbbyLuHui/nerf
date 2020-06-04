import sys
import os, os.path
import imageio
import numpy as np

def main(exp_name):
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
    path="./data/nerf_llff_data/"+exp_name
    valid_images=[".jpg", ".png", ".JPG"]
    folder_extension = ["","_4","_8"]
    for folder_ext in folder_extension:
        for f in os.listdir(path+"/images"+folder_ext+"_rgb"):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            print("---------------------------", f)
            imgs = imread(path+"/images"+folder_ext+"_rgb/"+f) 
            imgs = np.dot(imgs[... , :3],[0.114, 0.587, 0.299])
            imgs = np.stack((imgs,)*3, axis=-1)
            savedir=path+ "/images"+folder_ext
            filename = os.path.join(savedir, '{}'.format(f))
            imageio.imwrite(filename, imgs)

def main_synthetic(exp_name):
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
    path="./data/nerf_synthetic/"+exp_name
    valid_images=[".jpg", ".png", ".JPG"]
    for f in os.listdir(path+"/train_rgb"):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        print("---------------------------", f)
        imgs = imread(path+"/train_rgb/"+f) 
        alpha = imgs[..., 3]
        imgs = np.dot(imgs[... , :3],[0.114, 0.587, 0.299])
        imgs = np.stack((imgs,imgs, imgs, alpha), axis=-1)
        print("---------------------------", imgs.shape)
        savedir=path+ "/train"
        filename = os.path.join(savedir, '{}'.format(f))
        imageio.imwrite(filename, imgs)

if __name__=="__main__":
    if sys.argv[2]=="llff":
        main(sys.argv[1])
    elif sys.argv[2]=="synthetic":
        main_synthetic(sys.argv[1])
    else:
        print("Wrong type, input llff or synthetic as third command line argument")
