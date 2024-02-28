import os
import numpy as np
import pdb
import detect
import find_circle
import argparse

def process(file='Receive/a.png'):
    try:
        file_name = os.path.basename(file).split('.')[0]
        # predict bound
        agsps1 = argparse.ArgumentParser()
        agsps1.add_argument('--size', type=int, default=227)
        agsps1.add_argument('--model_weight', type=str, default='best.pt')
        agsps1.add_argument('--data', type=str, default='')
        agsps1.add_argument('--save_path', type=str, default='Result')
        args1 = agsps1.parse_args()
        args1.data = file
        detect.main(args1)

        # os.system(f'python detect.py --data {file}')


        # get  bound
        with open(f'Result/labels/keypoint_{file_name}.txt', 'r') as f:
            bound = f.readline()

        # predict circle
        agsps2 = argparse.ArgumentParser()
        # agsps2.add_argument('--filename', type=str)
        agsps2.add_argument('--minr', type=int, default=135)
        agsps2.add_argument('--maxr', type=int, default=150)
        agsps2.add_argument('--save_folder', type=str, default='')
        args2 = agsps2.parse_args()
        find_circle.hough_circle(file, args2.minr, args2.maxr, 'Result')
        # os.system(f'python find_circle.py {file} --save_folder Result')


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

        return angle, c_x, c_y, b, True

    except:
        return 0., 0., 0., 0, False


if __name__ == '__main__':
    
    agsps = argparse.ArgumentParser()
    agsps.add_argument('--path', type=str, default='Receive/a.png')

    args = agsps.parse_args()
    process(args.path)