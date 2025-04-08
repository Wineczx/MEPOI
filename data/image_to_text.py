# Author: T_Xu(create), S_Sun(modify)

import os
import argparse
import json
import time
import logging

def image_to_text(skip_img_file, photo_dir_, write_filename_, cuda_no='cuda:3'):
    from PIL import Image
    from transformers import AutoProcessor, Blip2ForConditionalGeneration
    import torch
    processor = AutoProcessor.from_pretrained("/data/CaiZhuaoXiao/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("/data/CaiZhuaoXiao/blip2-opt-2.7b", torch_dtype=torch.float16)
    device = cuda_no if torch.cuda.is_available() else "cpu"
    model.to(device)

    prompt = ''

    # 将图像转换为文本，封装为一个函数
    def image2text(image_filename):
        try:
            image = Image.open(image_filename).convert('RGB')
        except Image.UnidentifiedImageError:
            logger.error(f'转换失败, UnidentifiedImageError, 图片名称为:{image_filename}')
            return None
        except OSError as e:
            logger.error(f'转换失败, {repr(e)}, 图片名称为:{image_filename}')
            return None
        inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text

    
    # 记录信息
    total_num = 0 # 转换的总数
    count = 1
    time_start = time.time()

    # 加载已转换记录
    skip_img_set = None
    with open(skip_img_file, 'a+') as f:
        f.seek(0)
        skip_img_set = set(f.read().splitlines())
        f.close()

    # 图片字典，key为row_i+gmap_id，value为对应图片集paths
    images_dict = dict()
    # 生成图片字典
    for filename_ in os.listdir(photo_dir_):
        # 文件名（gmap_1_0x7c00456eecad3111-0x8217f9600c51f33_1.png）按'_'分割字符串
        filename_split = filename_.split('~')
        # 组织dict的key
        key = filename_split[0] 

        # 跳过已转换的图片j
        if key in skip_img_set:
            continue

        # 组织对应key的value
        if images_dict.get(key) is None:
            images_dict[key] = [filename_]
        else:
            images_dict[key].append(filename_)
    # 更新总数
    total_num = len(images_dict)

    # 遍历图片字典，将每个key对应的图片集转为文字集
    for key in images_dict:
        # 文字集
        image_text_list = []
        # 遍历图片集，将每个图片转为文字
        image_filename_list= images_dict[key]
        for image_filename in image_filename_list:
            # 转换
            image_text = image2text(photo_dir_ + image_filename)
            if image_text is not None: # 异常的先不加入，最后处理
                image_text_list.append(image_text)
        
        # 更新转换结果，格式与review_summary.json一致，key为row_i+gmap_id，value为des
        with open(write_filename_, 'a+') as f:
            f.write(json.dumps({key: image_text_list}) + '\n')
            f.flush()
            f.close()
        # 更新已保存记录
        with open(skip_img_file, 'a+') as f:
            f.write(key + '\n')
            f.flush()
            f.close()
        
        # 计算完成循环的剩余时间
        time_end = time.time()
        time_left = (time_end - time_start) / (count + 1) * (total_num - count)
        # 转换为时分秒
        time_left = time.strftime("%H:%M:%S", time.gmtime(time_left))

        print(count, ' / ', total_num, ', time left: ', time_left, ' : ', image_text_list)
        count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', type=str, default='NYC', help='the region name of datasets(e.g. California)')
    parser.add_argument('--dataset_path', type=str, default='/data/CaiZhuaoXiao/yelp/FL/', help='the index of the cuda')
    parser.add_argument('--cuda', type=str, default='0', help='the index of the cuda')
    args, _ = parser.parse_known_args()

    parent_path = os.path.join(args.dataset_path)

    # 初始化日志
    logfilename = 'transform_img_to_text.log'
    logfilepath = os.path.join(parent_path, logfilename)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    fh = logging.FileHandler(logfilepath, 'a', 'utf-8')
    fh.setLevel(logging.INFO)
    logging.basicConfig(
        level=logging.INFO,
        handlers = [sh, fh]
    )
    logger = logging.getLogger()

    # 图片转文字表述
    image_to_text(parent_path + 'skip_img_file.json', # 用于断点续传的跳过文件，每个地区一个
     parent_path + 'photos/', # 已下载的meta-xxx.json的（每个POI的）图片集
     parent_path + 'image_description.json', # 输出文件
     'cuda:' + args.cuda) # 使用的GPU