from tif_split_to_tif import main as main_split
from labelCleaning import main as main_labelCleaning
from rename import rename
from allot import allot
from allot import make_dir

if __name__ == '__main__':
    num = 9  # 有多少张图片
    # 图片裁剪尺寸
    crop_size = (256, 256)
    # 图片裁剪步长
    stride = 128
    image_path = "images_256"
    mask_path = "masks_256"
    # 输出的数据集根目录
    dataset_root = r"shawan_6fenlei_256"

    #以下两个路径需要在for循环里进行修改
    #big_tif_image_path = f"image/image_part{i + 1}.tif"  # TODO 需要原图片路径
    #big_tif_mask_path = f"mask/part{i + 1}.tif"  # TODO 需要修改mask路径

    for i in range(num):
        # 1、图片切割
        # 输入路径
        big_tif_image_path = f"tif_image/splited_image_and_mask/image_part{i + 1}.tif"  #TODO 需要原图片路径
        big_tif_mask_path = f"tif_image/splited_image_and_mask/mask_part{i + 1}.tif"     #TODO 需要修改mask路径

        main_split(big_tif_image_path=big_tif_image_path,  # 原图路径
             big_tif_mask_path=big_tif_mask_path,  # 标签路径
             crop_size=crop_size,  # 裁剪尺寸
             stride=stride)  # 裁剪步长

        # 2、 数据清洗
        # 输入路径
        image_path = image_path
        mask_path = mask_path
        # 含0量超过这个比例就删了这标签
        percentage = 0.95
        main_labelCleaning(image_path=image_path, mask_path=mask_path, percentage=percentage)

        # 3、重命名
        """
        只适用于一级目录
        """
        dir_path1 = image_path
        dir_path2 = mask_path
        new_first_name = f"210724-02-{i + 1}"
        suffix_name = ".tif"
        rename(dir_path=dir_path1, new_first_name=new_first_name, suffix_name=suffix_name)
        rename(dir_path=dir_path2, new_first_name=new_first_name, suffix_name=suffix_name)

    # 4、最后重新命名回来
    dir_path1 = image_path
    dir_path2 = mask_path

    new_first_name = "210724-02"
    suffix_name = ".tif"

    rename(dir_path=dir_path1, new_first_name=new_first_name, suffix_name=suffix_name)
    rename(dir_path=dir_path2, new_first_name=new_first_name, suffix_name=suffix_name)

    # 5、制作成mmsegmentation需要的数据集

    # 输入路径
    image_path = image_path
    mask_path = mask_path

    # 训练集的百分比
    train_percentage = 0.6

    path_list = make_dir(dataset_root=dataset_root)

    allot(image_path=image_path,  # 原图路径
          mask_path=mask_path,  # 标签路径
          train_percentage=train_percentage,  # 训练集百分比
          path_list=path_list)  # 输出路径列表(固定的)
