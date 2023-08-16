from flask import Flask, render_template, request,session,redirect,url_for
import pandas as pd
import keras
import cv2
import numpy as np
import pickle
import os


app = Flask(__name__)
app.secret_key='capstone-d'
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    # # rendering it to the html file
    return render_template("index.html")


@app.route('/upload' , methods=['GET','POST'])
def upload_route():
    if 'image' not in request.files:
        return "No file part"

    file = request.files['image']

    if file.filename == '':
        return "No selected file"

    file_path = f"{app.config['UPLOAD_FOLDER']}/{file.filename}"
    file.save(file_path)
    session['uploaded_image']=file_path
    uploaded_image_url=url_for('static',filename=file.filename)
    # app.logger.debug(f"File saved to: {file_path}")
    return render_template('index.html',uploaded_image_url=uploaded_image_url)
     
@app.route('/extract_text', methods=['POST','GET'])
def extract_text():
     # back end logic
     if 'uploaded_image' in session:
         img_path=session['uploaded_image']
     image_data=extract_image(img_path)
     with open('tokenizer.pickle', 'rb') as handle:
         loaded_tokenizer = pickle.load(handle)
    
     extract_model=keras.models.load_model('text_extractor.keras')
     pred=extract_model.predict(image_data)
     output=decode_predictions(pred,loaded_tokenizer)
     text_output=str(output[0])
     return text_output[1:-1]



def extract_image(image_path):
    cur_image=[]
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img,(200, 50), interpolation = cv2.INTER_AREA)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) / 255
    img = np.expand_dims(img, axis = 2)
    cur_image.append(img)
    cur_image=np.array(cur_image)
    
    return cur_image

def decode_predictions(pred,tokenizer):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :34
    ]
    output_text = []
    for res in results:
        decoded = tokenizer.sequences_to_texts([res.numpy()])
        output_text.append(decoded)
    return output_text

if __name__ == "__main__":
     app.run(debug=True,threaded=True)