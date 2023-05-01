import os
from PIL import Image
import numpy as np


def mask_analysis(mask_path:str):
    """输入:标签所在路径
        效果:
    """
    # 图片所在路径
    mask_path = mask_path
    
    # 获取所有的mask图片名
    name_list = os.listdir(path=mask_path)

    # 所有图片array的unique值的拼接
    unique_cat = np.zeros(1)

    # 所有图片通道数的集合，用于判断是否有多种通道数
    mask_channels_num_set = set()

    # 所有图片分辨率的集合，用于判断是否有多种分辨率
    mask_shape_set = set()

    # 各个像素值数量的字典
    pixel_sum_dict = dict()

    # 遍历这些图片，将它们都转化为np.array处理
    for index, image_mask_name in enumerate(name_list):

        # 通过PIL.Image打开图片
        image_mask = Image.open(os.path.join(mask_path, image_mask_name))

        # 单张图片的尺寸获取，加入集合
        image_mask_shape = image_mask.size
        mask_shape_set.add(image_mask_shape)

        # 单张图片的通道数获取，加入集合
        image_mask_channels_num = len(image_mask.split())
        mask_channels_num_set.add(image_mask_channels_num)

        # 转化为numpy数组
        image_mask_array = np.asarray(image_mask)

        # 去重取值
        unique = np.unique(image_mask_array)

        # 获取各个像素值数量的字典
        for pixel_value in unique:
            if pixel_value in pixel_sum_dict.keys():
                pixel_sum_dict[pixel_value] += np.sum(image_mask_array == pixel_value)
            else:
                pixel_sum_dict[pixel_value] = np.sum(image_mask_array == pixel_value)

        # 将这个张量和全局变量unique_cat拼接，得到所有图片的不同值的集合
        unique_cat = np.concatenate((unique_cat, unique))
        print("\r{}/{}:{}".format(index+1, len(name_list), "▋"*(int((index+1)/len(name_list)*100)//2)),
              "%{:.1f}".format((index+1)/len(name_list)*100),
              flush=True,
              end="")
        # sys.stdout.flush()

    # 对所有图片的张量的不同值的集合（里面有重复的）再取unique，得到所有图片张量的不同值的张量（没有重复）
    ture_unique = np.unique(unique_cat)

    # 输出包含所有图片不同张量值的张量（无重复）
    print("\nmask 标签中存在的像素值有：", ture_unique)
    print("mask 标签的通道数为：", mask_channels_num_set)
    print("mask 标签的分辨率为：", mask_shape_set)

    for key, value in pixel_sum_dict.items():
        print(f"像素值{key}的占比为{value/(image_mask_array.size*len(name_list))*100:.3f}%")



if __name__ == "__main__":

    # 标签所在路径
    mask_path = r"shawan_4fenlei_256\ann_dir\train"

    mask_analysis(mask_path=mask_path)
    