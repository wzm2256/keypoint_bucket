import sys
import cv2 as cv
import numpy as np
import pdb
from pathlib import Path
import os

def hough_circle(filename, minr, maxr, save_folder, show=1):
    
    # default_file = 'smarties.png'
    # filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    # pdb.set_trace()
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    src = cv.resize(src, (512, 512))
    # Check if image is loaded fine
    # if src is None:
    #     print ('Error opening image!')
    #     print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
    #     return -1
    
    
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=minr, maxRadius=maxr)
    
    p = Path(filename)
    txt_file_name = os.path.join(save_folder, 'labels', f"circle_{p.name.split('.')[0]}.txt")
    img_file_name = os.path.join(save_folder, f"circle_{p.name}")

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (255, 0, 255), 3)
            # circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 3)
            print(f'File {filename}; \t Center {center}; \t Radius {radius}')
            with open(txt_file_name, 'w') as f:
                f.write(f'{center[0]} {center[1]}')
            cv.imwrite(img_file_name, src)
            print(f'Circle saved to {txt_file_name}')
            print(f'Image with circle saved to {img_file_name}')
            # cv.imshow("detected circles", src)
            # cv.waitKey(0)
            return
    else:
        raise ValueError('Cannot find any circle!')
    # if show == 1:
        
        
    
    # return 0
    # print(circles.shape)
    # pdb.set_trace()
    # print(circles[0, :, 0])
    # return center, radius

if __name__ == "__main__":
    import argparse
    import pdb
    agsps = argparse.ArgumentParser()
    agsps.add_argument('filename', type=str)
    agsps.add_argument('--minr', type=int, default=135)
    agsps.add_argument('--maxr', type=int, default=150)
    agsps.add_argument('--save_folder', type=str, default='')

    args = agsps.parse_args()
    hough_circle(args.filename, args.minr, args.maxr, args.save_folder)