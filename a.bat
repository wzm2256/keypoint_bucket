:: conda activate torch22

:: 0. Convert to yolo dataset
python labelme2yolo.py --json_dir raw_image

:: 1. Train model
python train.py

:: 2. evaluate on all data
python test.py
:: All performance are recorded in all_test.txt and all_train.txt


:: 2. detect key point
::python detect.py --data "/home/wzm/MyCode/yolo/YOLODataset/images/train/"

::Result will be saved to Result/a.png and Result/labels/a.txt

::3. setup server

::python server.py

