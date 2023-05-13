import os
import rasterio
import numpy as np
import math

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



def main(tif_dir:str):

    name_list = os.listdir(tif_dir)

    mean_dict = dict()
    std_dict = dict()
    
    for index, name in enumerate(name_list):
        tif_path = os.path.join(tif_dir, name)
        tif_array = read_tif(tif_path)
        for channel in range(0, tif_array.shape[0]):
            mean = np.mean(tif_array[channel, :, :])
            if channel in mean_dict.keys():
                mean_dict[channel] = (mean_dict[channel]+mean)/2
            else:
                mean_dict[channel] = mean            
        print("\r mean:{}/{}:{}".format(index+1, len(name_list), "▋"*(int((index+1)/len(name_list)*100)//2)),
              "%{:.1f}".format((index+1)/len(name_list)*100),
              flush=True,
              end="")
        
    for index, name in enumerate(name_list):
        tif_path = os.path.join(tif_dir, name)
        tif_array = read_tif(tif_path)
        for channel in range(0, tif_array.shape[0]):            
            square_subtraction = ((tif_array[channel, :, :] - mean_dict[channel])**2).sum()
            if channel in std_dict.keys():
                std_dict[channel] = std_dict[channel]+square_subtraction
            else:
                std_dict[channel] = square_subtraction
        print("\r std:{}/{}:{}".format(index+1, len(name_list), "▋"*(int((index+1)/len(name_list)*100)//2)),
              "%{:.1f}".format((index+1)/len(name_list)*100),
              flush=True,
              end="")
        
    for channel in range(0, tif_array.shape[0]):            
        std_dict[channel] = np.sqrt(std_dict[channel]/(tif_array[channel, :, :].size*len(name_list)))
            

    return mean_dict, std_dict



if __name__ == "__main__":
    
    dict_tuple = main(tif_dir="images_256")

    print(f"\n mean:{dict_tuple[0]} \n std:{dict_tuple[1]}")

    # a = list(range(1, 37))
    # a = np.asarray(a)
    # a = a.reshape(4, 3, 3)
    # # a = a[0,:,:]
    # print(np.shape(a)[0])

    