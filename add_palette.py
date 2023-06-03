import os
import rasterio
import numpy as np
import time
from PIL import Image

def read_tif(tif_image_path: str):
    """读取tif文件
        输入: tif 图片路径
        返回: 图像对应的 numpy 数组
    """
    # 用遥感图像的专门的包来打开
    big_tif = rasterio.open(tif_image_path, mode="r")
    # print(f"name:{big_tif.name}, count:{big_tif.count}, height:{big_tif.height}, width:{big_tif.width}, dtype:{big_tif.dtypes}")

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
    # print(
    #     f"shape:{big_tif_array.shape}, max:{big_tif_array.max()}, \nunique:{np.unique(big_tif_array)}")
    return big_tif_array




# 使用PIL库生成调色板格式的图
def Convert_Palette(tif_image, PALETTE):

    PALETTE = PALETTE
    
    PALETTE = np.array(PALETTE)

    PALETTE = PALETTE.astype(np.uint8)

    # 目标图像
    out_img = Image.open(tif_image)

    # 将图像模式变为调色板模式 "P"
    out_img = out_img.convert("P")
    # print(out_img.size, out_img.mode)

    # 将调色板信息添加进目标图像中
    # 就是以刚刚得到的调色板将图片转换为调色板模式的伪彩图
    out_img.putpalette(PALETTE)     

    out_img.save("colorful"+tif_image)


if __name__ == "__main__":

    PALETTE = [[0, 0, 0], [44, 255, 28], [254, 254, 254], [246, 14, 14],
               [249, 255, 33], [74, 254, 254], [59, 0, 252]]
    
    Convert_Palette("2023-05-31_221342.tif", PALETTE=PALETTE)