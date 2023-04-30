import os

"""
命名太单一不利于后续加入新的训练数据,
所以要求大家命名为210724-02-1 至210724-02-N
(命名规则：这张图的时间-分辨率-图片序号), 
分辨率指遥感图像的空间分辨率, 02 代表 2m 的空间分辨率
裁剪的尺寸在数据集文件夹上注明

只适用于一级目录
"""

def rename(dir_path:str, new_first_name, suffix_name):
    """输入: 需要重命名的文件所在路径, 新的固定名称, 文件后缀名
        操作:对该路径下所有文件以 {new_first_name}-index{suffix_name} 的形式命名
    """
    name_list = os.listdir(dir_path)
    for index, name in enumerate(name_list):
        os.rename(os.path.join(dir_path, name), 
                  os.path.join(dir_path, f"{new_first_name}-{index:0>4d}{suffix_name}"))
        print("\r重命名{}/{}:{}".format(index+1, len(name_list), "▋"*(int((index+1)/len(name_list)*100)//2)),
              "%{:.1f}".format((index+1)/len(name_list)*100),
              flush=True,
              end="")


if __name__ == "__main__":
    """
    命名太单一不利于后续加入新的训练数据,
    所以要求大家命名为210724-02-1 至210724-02-N
    (命名规则：这张图的时间-分辨率-图片序号), 
    分辨率指遥感图像的空间分辨率, 02 代表 2m 的空间分辨率
    裁剪的尺寸在数据集文件夹上注明

    只适用于一级目录
    """

    dir_path1 = "images_256"
    dir_path2 = "masks_256"

    new_first_name = "210724-02"
    suffix_name = ".tif"

    rename(dir_path=dir_path1, new_first_name=new_first_name, suffix_name=suffix_name)
    rename(dir_path=dir_path2, new_first_name=new_first_name, suffix_name=suffix_name)