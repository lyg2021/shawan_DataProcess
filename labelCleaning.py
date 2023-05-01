import os
import numpy as np
import rasterio


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
    # print(f"shape:{big_tif_array.shape}, max:{big_tif_array.max()}, \nunique:{np.unique(big_tif_array)}")
    return big_tif_array


def is_empty_mask(image_array:np.ndarray, percentage: float=0.9):
    """numpy数组判断是否含0量超标
        image_array: 图像转为的numpy数组
        percentage: 背景像素占比高于这个比例认为含0量超标, 属于 empty_mask
    """
    background_num = np.sum(image_array == 0)   # 统计像素值为0的像素的个数, 背景
    image_size = image_array.size  # 统计图像的像素总个数
    threshold_num = int(image_size * percentage)  # 像素个数阈值，高于这个值就认为标签无效

    if background_num >= threshold_num:
        return True
    else:
        return False
    

def main(image_path:str, mask_path:str, percentage:float):
    """输入:原图路径和标签路径,删除标签含0量超过 percentage*总像素数量的标签和关联的原图"""

    # 输入路径
    image_path = image_path
    mask_path = mask_path

    # 含0量超过这个比例就删了这标签
    percentage = percentage

    # 删了多少个
    delete_num = 0

    name_list = os.listdir(path=mask_path)

    for index, name in enumerate(name_list):
        mask_array = read_tif(os.path.join(mask_path, name))
        if is_empty_mask(image_array=mask_array, percentage=percentage):
            os.remove(os.path.join(mask_path, name))
            os.remove(os.path.join(image_path, name))
            delete_num += 1
        print("\r删除中:{}/{}:{}".format(index+1, len(name_list), "▋"*(int((index+1)/len(name_list)*100)//2)),
              "%{:.1f}".format((index+1)/len(name_list)*100),
              flush=True,
              end=f":删除了{delete_num}个含0量较高的图像和标签")
        

if __name__ == "__main__":

    # 输入路径
    image_path = "images_512"
    mask_path = "masks_512"

    # 含0量超过这个比例就删了这标签
    percentage = 0.95

    main(image_path=image_path, mask_path=mask_path, percentage=percentage)


 

