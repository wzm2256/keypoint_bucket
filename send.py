import requests

with open(r'G:\Mycode\bucket\data\image1new\1_3.bmp', 'rb') as f:
    r = requests.post('http://127.0.0.1:8000', files={'file':f})


