import os
import rasterio
import numpy as np
import time

"""用于将切割好的小图合成原来的大图
    
"""

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


def get_tif_array_list(tif_path:str):
    """将路径下的tif图片弄到一个列表里
        输入:tif图片根目录
        返回:numpy数组的列表
    """
    tif_name_list = os.listdir(tif_path)
    tif_array_list = []

    for tif_name in tif_name_list:
        tif_array = read_tif(os.path.join(tif_path, tif_name))
        tif_array_list.append(tif_array)
    
    return tif_array_list


def merge_tif(tif_array_list: list, height:int, width:int):
    """将图片合成大图
        输入: 小图转化的numpy数组列表, 大图的高, 大图的宽
        返回: 合成好的numpy数组
    """

    # 先拼横条, 再把横条合成块
    bar_shape_tif_array_list = []

    for index, tif_array in enumerate(tif_array_list):
        if index == 0:
            bar_shape_tif_array = tif_array
        else:
            if len(bar_shape_tif_array.shape) == 2:     # 二维数组(H, W)
                if bar_shape_tif_array.shape[1] < width:
                    bar_shape_tif_array = np.concatenate((bar_shape_tif_array, tif_array), axis=1)
                    if index == len(tif_array_list)-1:
                        bar_shape_tif_array_list.append(bar_shape_tif_array)
                else:
                    bar_shape_tif_array_list.append(bar_shape_tif_array)
                    bar_shape_tif_array = tif_array

            elif len(tif_array.shape) == 3:   # 三维数组(C, H, W)
                if bar_shape_tif_array.shape[1] < width:
                    bar_shape_tif_array = np.concatenate((bar_shape_tif_array, tif_array), axis=2)
                    if index == len(tif_array_list)-1:
                        bar_shape_tif_array_list.append(bar_shape_tif_array)
                else:
                    bar_shape_tif_array_list.append(bar_shape_tif_array)
                    bar_shape_tif_array = tif_array
            else:
                print("不对劲，这输入的数组不是二维也不是三维")

    for index, bar_shape_tif_array in enumerate(bar_shape_tif_array_list):
        if len(bar_shape_tif_array.shape) == 2:     # 二维数组(H, W)
            if index == 0:
                big_tif_array = bar_shape_tif_array
            else:
                big_tif_array = np.concatenate((big_tif_array, bar_shape_tif_array), axis=0)

        elif len(tif_array.shape) == 3:   # 三维数组(C, H, W)
            if index == 0:
                big_tif_array = bar_shape_tif_array
            else:
                big_tif_array = np.concatenate((big_tif_array, bar_shape_tif_array), axis=1)

    return big_tif_array



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


def main(tif_dir:str, height:int, width:int):
    

    tif_array_list = get_tif_array_list(tif_dir)
    big_tif_array = merge_tif(tif_array_list, height=height, width=width)
    print(f"保存图片的(H, W):{big_tif_array.shape}")

    tif_name = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())

    # 保存tif文件(原图)
    metadata = {
        'driver': 'GTiff',
        'width': width,
        'height': height,
        'count': 1,
        'dtype': np.uint8
    }
    save_array_as_tif(matrix=big_tif_array,
                      path=f"./{tif_name}.tif",
                      profile=metadata)

    # for index, tif_array in enumerate(image_tif_array_list):
    #     metadata["width"] = tif_array.shape[2]
    #     metadata["height"] = tif_array.shape[1]
    #     save_array_as_tif(tif_array, os.path.join(
    #         save_image_path_root, f"{index}.tif"), metadata)
    #     print("\r保存进度(原图):{}/{}:{}".format(index+1, len(image_tif_array_list), "▋"*(int((index+1)/len(image_tif_array_list)*100)//2)),
    #           "%{:.1f}".format((index+1)/len(image_tif_array_list)*100),
    #           flush=True,
    #           end="")
        
    # # 保存tif文件(标签)
    # metadata = {
    #     'driver': 'GTiff',
    #     'width': image_size[1],
    #     'height': image_size[0],
    #     'count': 1,
    #     'dtype': np.uint8
    # }
    # for index, tif_array in enumerate(mask_tif_array_list):
    #     metadata["width"] = tif_array.shape[1]
    #     metadata["height"] = tif_array.shape[0]
    #     save_array_as_tif(tif_array, os.path.join(
    #         save_mask_path_root, f"{index}.tif"), metadata)
    #     print("\r保存进度(标签):{}/{}:{}".format(index+1, len(mask_tif_array_list), "▋"*(int((index+1)/len(mask_tif_array_list)*100)//2)),
    #           "%{:.1f}".format((index+1)/len(mask_tif_array_list)*100),
    #           flush=True,
    #           end="")


if __name__ == "__main__":

    
    main(tif_dir="masks_512",
         height=7239,
         width=13812)

    # print(len(tif_array_list))

    # a = np.array([[0, 0, 1, 1],[2, 2, 3, 3]])
    # b = np.array([[1, 1, 2, 2],[3, 3, 4, 4]])

    # print(a.shape, b.shape, np.concatenate((a, b), axis=1))



    pass


