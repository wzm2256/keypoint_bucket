import os
import numpy as np
import pdb

def process(file='Receive/a.jpg'):
    file_name = os.path.basename(file).split('.')[0]
    # predict bound
    os.system(f'python detect.py --data {file}')
    # pdb.set_trace()
    # get  bound
    with open(f'Result/labels/keypoint_{file_name}.txt', 'r') as f:
        bound = f.readline()

    # predict circle
    os.system(f'python find_circle.py {file} --save_folder Result')
    # get circle
    with open(f'Result/labels/circle_{file_name}.txt', 'r') as f:
        circle = f.readline()


    c_x, c_y = circle.split(' ')
    c_x = float(c_x) / 512.
    c_y = float(c_y) / 512.

    b = bound.split(' ')
    b_x = float(b[0])
    b_y = float(b[1])

    angle = np.arctan2(-b_y + c_y, b_x - c_x) / np.pi * 180

    with open(f'Result/labels/angle_{file_name}.txt', 'w') as f:
        f.write(f'{angle}\n')

    return angle, c_x, c_y


if __name__ == '__main__':
    import argparse
    agsps = argparse.ArgumentParser()
    agsps.add_argument('--path', type=str, default='Receive/a.png')

    args = agsps.parse_args()
    process(args.path)