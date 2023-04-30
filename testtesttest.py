import torch
import torchvision
import os
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
import rasterio

transform_to_Tensor = transforms.Compose([
    # transforms.Resize((256, 256)),
    transforms.ToTensor()
])
"""ToTensor最好不要在对图片处理的时候用，它会自动将图片的像素进行归一化处理"""

transform_to_PILImage = transforms.Compose([
    transforms.ToPILImage()
])
"""相对的ToPILImage会自动将tensor的图片乘255还原，所以输出的tensor值并不是图片原本的像素值"""


def read_tif(tif_image_path: str):
    """读取tif文件
        输入: tif 图片路径
        返回: 图像对应的 numpy 数组
    """
    # # 用 PIL 来读取，读到的是 8 位图，也就是像素值在 1~2^8 之间
    # big_tif_image = Image.open(tif_image_path)
    # print(big_tif_image.size, len(big_tif_image.split()), max(big_tif_image.getdata()))

    # # 用 opencv 读取试试，读取到的像素值max有达到21664
    # image_cv2 = cv2.imread(filename=tif_image_path, flags=-1)
    # print(f"shape:{image_cv2.shape}, min:{image_cv2.min()}, max:{image_cv2.max()}") # shape:(7429, 10832, 4), min:0, max:21664
    # image_array = np.asarray(image_cv2)
    # print(image_array.shape, image_array.max(), image_cv2.dtype)

    # 用遥感图像的专门的包来打开看看
    big_tif = rasterio.open(tif_image_path, mode="r")
    print(f"name:{big_tif.name}, count:{big_tif.count}, height:{big_tif.height}, width:{big_tif.width}, dtype:{big_tif.dtypes}")

    # rasterio 打开的图像对象好像只能一个波段一个波段的读取数据
    # 这个循环是把这些波段的np数组堆叠起来
    for band_count in range(1, big_tif.count+1):
        if band_count == 1:
            big_tif_array = big_tif.read(band_count)
        elif band_count == 2:
            big_tif_array = np.vstack(
                (np.expand_dims(big_tif_array, 0), np.expand_dims(big_tif.read(band_count), 0)))
        else:
            big_tif_array = np.vstack(
                (big_tif_array, np.expand_dims(big_tif.read(band_count), 0)))

    # shape:(channels, height, width)
    print(
        f"shape:{big_tif_array.shape}, max:{big_tif_array.max()}, \nunique:{np.unique(big_tif_array)}")
    return big_tif_array



if __name__ == "__main__":

    path = r"masks_256\2555.tif"
    # path = r"final_image.tif"

    tif_array = read_tif(path)


###################################################
    # Image.MAX_IMAGE_PIXELS = None
    # # image = cv2.imread(path)
    # # print(image.shape)

    # image = Image.open(path)

    # print("图片通道数：", len(image.split()), image.size)

    # image_tensor = transform_to_Tensor(image)

    # image_tensor *= 255

    # print(image_tensor.unique())
    # print(image_tensor,image_tensor.shape)
    # print(image_tensor, image_tensor.shape)
##########################################################
    


    ##################################

    # image_array = np.asarray(image) 
    # print(image_array, image_array.shape)
    # print(np.unique(image_array))

    # save_image = Image.fromarray(image_array)

    # save_image.show()

    ##########################################




    # tensor = transform_to_Tensor(image)
    # # tensor *= 255
    # print(tensor.unique())
    # print(tensor,tensor.shape)
    # image_mask_tensor = tensor



    # # print(image_mask_tensor)

    # print(image_mask_tensor)

    # image_show = transform_to_PILImage(image_mask_tensor)
    # image_show.show()