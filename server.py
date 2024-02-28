from flask import Flask, request
import os
app = Flask(__name__)
import numpy as np
import process_all
import requests
from waitress import serve

out_address = 'http://localhost:8000'
in_port = 8000


@app.route('/', methods=['POST'])
def index():
   if request.method == 'POST':
      # receive and save file
      f = request.files['file'] 
      f.save('Receive/a.png')
      print('Received an image...')
      angle, c_x, c_y, b, result = process_all.process()
      print('Sending results... ')
      # r = requests.post(out_address, json={'angle':angle, 'X': c_x, 'Y': c_y})
      r = requests.post(out_address, json={'result': result, 'R':b, 'X': c_x, 'Y': c_y})
      return 'OK'
   else:
      return 'Use POST requests'

if __name__ == '__main__':
   app.debug = True
   #  app.run(host='0.0.0.0', port=in_port)
   serve(app, host='0.0.0.0', port=in_port)