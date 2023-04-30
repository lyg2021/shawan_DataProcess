# ├── data
# │   ├── my_dataset
# │   │   ├── img_dir
# │   │   │   ├── train
# │   │   │   │   ├── xxx{img_suffix}
# │   │   │   │   ├── yyy{img_suffix}
# │   │   │   │   ├── zzz{img_suffix}
# │   │   │   ├── val
# │   │   ├── ann_dir
# │   │   │   ├── train
# │   │   │   │   ├── xxx{seg_map_suffix}
# │   │   │   │   ├── yyy{seg_map_suffix}
# │   │   │   │   ├── zzz{seg_map_suffix}
# │   │   │   ├── val

import os
import shutil
import random


def make_dir(dataset_root):
    """建立目录
        输入: 要建立的数据集的根目录
        返回: 建立的目录的列表
    """
    dir_list_list = [["img_dir", "train"], ["ann_dir", "train"], 
                ["img_dir", "val"], ["ann_dir", "val"], 
                ["img_dir", "test"], ["ann_dir", "test"]]
    return_list = []
    for dir_list in dir_list_list:
        full_dir = os.path.join(dataset_root, dir_list[0], dir_list[1])
        return_list.append(full_dir)
        if not os.path.exists(full_dir):
            os.makedirs(full_dir)
    return return_list
    



def allot(image_path: str, mask_path, train_percentage: float, path_list: list):
    """
        输入:原图路径,标签路径,训练集所占百分比,目标路径的列表
        操作:直接打乱顺序复制过去,按照比例分配数据
    """
    name_list = os.listdir(image_path)

    # 打乱下
    random.shuffle(name_list)

    # 直接开始copy
    for index, name in enumerate(name_list):
        if index < len(name_list)*train_percentage:
            shutil.copyfile(src=os.path.join(image_path, name), 
                            dst=os.path.join(path_list[0], name))
            shutil.copyfile(src=os.path.join(mask_path, name), 
                            dst=os.path.join(path_list[1], name))
        else:
            shutil.copyfile(src=os.path.join(image_path, name), 
                            dst=os.path.join(path_list[2], name))
            shutil.copyfile(src=os.path.join(mask_path, name), 
                            dst=os.path.join(path_list[3], name))
        print("\r allot{}/{}:{}".format(index+1, len(name_list), "▋"*(int((index+1)/len(name_list)*100)//2)),
              "%{:.1f}".format((index+1)/len(name_list)*100),
              flush=True,
              end="")
    
        


if __name__ == "__main__":
    # 输入路径
    image_path = r"images_256"
    mask_path = r"masks_256"

    # 训练集的百分比
    train_percentage = 0.6

    # 输出的数据集根目录
    dataset_root = r"shawan_4fenlei_256"

    path_list = make_dir(dataset_root=dataset_root)

    allot(image_path=image_path,    # 原图路径
          mask_path=mask_path,      # 标签路径
          train_percentage=train_percentage,    # 训练集百分比
          path_list=path_list)      # 输出路径列表(固定的)
