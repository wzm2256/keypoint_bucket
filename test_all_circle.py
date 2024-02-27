import os
import pdb


folder_all = ['YOLODataset/images/train', 'YOLODataset/images/val']
# folder = r'G:\Mycode\bucket\image1'
for folder in folder_all:
    all_file = os.listdir(folder)
    # pdb.set_trace()
    for f in all_file:
        if f.endswith('png'):
            # print(f'testing file {f}')
            os.system(f'python find_circle.py {os.path.join(folder, f)} --save_folder predicted_circle')
