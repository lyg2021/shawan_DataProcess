import os
import rasterio
import numpy as np

"""用于将大的tif文件切割成小的tif文件,包括原图和标签, 原图为4通道, 标签为单通道
    1. 可调节切割尺寸
    2. 可调节切割步长(滑动窗口切割法)
"""

def read_tif(tif_image_path: str):
    """读取tif文件
        输入: tif 图片路径
        返回: 图像对应的 numpy 数组
    """
    # 用遥感图像的专门的包来打开
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


def split_tif(tif_array: np.ndarray, stride: int, crop_size: tuple):
    """分割图片, 保留细小碎块
        输入: 图像对应的数组(C,H,W), 裁剪步长(采用滑动窗口法), 裁剪尺寸(height, width)
        返回: 分割后的数组列表
    """
    # np.set_printoptions(threshold=np.inf)

    # 最终返回的数组列表, 代表所有的分割图像
    tif_array_list = []

    # 裁剪的高和宽
    crop_height, crop_width = crop_size

    # 获取原图的高和宽
    if len(tif_array.shape) == 2:   # 二维数组(H, W)
        height = tif_array.shape[0]
        width = tif_array.shape[1]
    elif len(tif_array.shape) == 3:   # 三维数组(C, H, W)
        height = tif_array.shape[1]
        width = tif_array.shape[2]
    else:
        print("不对劲，这输入的数组不是二维也不是三维")

    # 开始数组切片, 滑动窗口, 以 stride 为步长,
    # 从左到右, 从上到下, 切出大小为 (crop_height, crop_width) 的小图
    for y in range(0, height+1, stride):
        for x in range(0, width+1, stride):
            if x+crop_width <= width and y+crop_height <= height:
                if len(tif_array.shape) == 3:
                    chip = tif_array[:, y:y+crop_height, x:x+crop_width]
                elif len(tif_array.shape) == 2:
                    chip = tif_array[y:y+crop_height, x:x+crop_width]
                tif_array_list.append(chip)

            elif x+crop_width > width and y+crop_height <= height:
                if len(tif_array.shape) == 3:
                    chip = tif_array[:, y:y+crop_height, x:]
                elif len(tif_array.shape) == 2:
                    chip = tif_array[y:y+crop_height, x:]
                tif_array_list.append(chip)

            elif x+crop_width <= width and y+crop_height > height:
                if len(tif_array.shape) == 3:
                    chip = tif_array[:, y:, x:x+crop_width]
                elif len(tif_array.shape) == 2:
                    chip = tif_array[y:, x:x+crop_width]
                tif_array_list.append(chip)

            elif x+crop_width > width and y+crop_height > height:
                if len(tif_array.shape) == 3:
                    chip = tif_array[:, y:, x:]
                elif len(tif_array.shape) == 2:
                    chip = tif_array[y:, x:]
                tif_array_list.append(chip)

    return tif_array_list


def save_array_as_tif(matrix, path, profile=None, prototype=None):
    """将 numpy 数组保存为 tif 图像
        输入: numpy 数组, 保存路径, 预设参数
    """
    assert matrix.ndim == 2 or matrix.ndim == 3
    if prototype:
        with rasterio.open(str(prototype)) as src:
            profile = src.profile
    with rasterio.open(path, mode='w', **profile) as dst:
        if matrix.ndim == 3:
            for i in range(matrix.shape[0]):
                dst.write(matrix[i], i + 1)
        else:
            dst.write(matrix, 1)


def main(big_tif_image_path:str, big_tif_mask_path:str, crop_size:tuple, stride:int):
    # 输入路径
    big_tif_image_path = big_tif_image_path
    big_tif_mask_path = big_tif_mask_path

    # 图片裁剪尺寸
    image_size = crop_size
    
    # 图片裁剪步长
    stride = stride

    # 读取tif为numpy数组
    array_image = read_tif(big_tif_image_path)
    array_mask = read_tif(big_tif_mask_path)

    # 分割numpy数组为numpy数组list
    image_tif_array_list = split_tif(array_image, stride, image_size)
    mask_tif_array_list = split_tif(array_mask, stride, image_size)

    # 输出路径
    save_image_path_root = f"images_{image_size[0]}"
    if not os.path.exists(save_image_path_root):
        os.makedirs(save_image_path_root)
    save_mask_path_root = f"masks_{image_size[0]}"
    if not os.path.exists(save_mask_path_root):
        os.makedirs(save_mask_path_root)

    # 保存tif文件(原图)
    metadata = {
        'driver': 'GTiff',
        'width': image_size[1],
        'height': image_size[0],
        'count': 4,
        'dtype': np.uint16
    }
    for index, tif_array in enumerate(image_tif_array_list):
        metadata["width"] = tif_array.shape[2]
        metadata["height"] = tif_array.shape[1]
        save_array_as_tif(tif_array, os.path.join(
            save_image_path_root, f"{index:0>4d}.tif"), metadata)
        print("\r保存进度(原图):{}/{}:{}".format(index+1, len(image_tif_array_list), "▋"*(int((index+1)/len(image_tif_array_list)*100)//2)),
              "%{:.1f}".format((index+1)/len(image_tif_array_list)*100),
              flush=True,
              end="")
        
    # 保存tif文件(标签)
    metadata = {
        'driver': 'GTiff',
        'width': image_size[1],
        'height': image_size[0],
        'count': 1,
        'dtype': np.uint8
    }
    for index, tif_array in enumerate(mask_tif_array_list):
        metadata["width"] = tif_array.shape[1]
        metadata["height"] = tif_array.shape[0]
        save_array_as_tif(tif_array, os.path.join(
            save_mask_path_root, f"{index:0>4d}.tif"), metadata)
        print("\r保存进度(标签):{}/{}:{}".format(index+1, len(mask_tif_array_list), "▋"*(int((index+1)/len(mask_tif_array_list)*100)//2)),
              "%{:.1f}".format((index+1)/len(mask_tif_array_list)*100),
              flush=True,
              end="")


if __name__ == "__main__":
    # 输入路径
    big_tif_image_path = "tif_image\splited_image_and_mask\image_part1.tif"
    big_tif_mask_path = "tif_image\splited_image_and_mask\mask_part1.tif"

    # 图片裁剪尺寸
    crop_size = (256, 256)
    
    # 图片裁剪步长
    stride = 256

    main(big_tif_image_path=big_tif_image_path,     # 原图路径
         big_tif_mask_path=big_tif_mask_path,       # 标签路径
         crop_size=crop_size,                       # 裁剪尺寸
         stride=stride)                             # 裁剪步长


