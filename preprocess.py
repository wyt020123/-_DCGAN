import os
import cv2
from tqdm import tqdm

input_folder = "images"
output_folder = "processed_images"

os.makedirs(output_folder, exist_ok=True)

for img_name in tqdm(os.listdir(input_folder)):
    img = cv2.imread(os.path.join(input_folder, img_name))
    img = cv2.resize(img, (64, 64))
    cv2.imwrite(os.path.join(output_folder, img_name), img)

print("数据预处理完成！")
