import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import torch
import pandas as pd
import CNN
from CropRecommendation import predict_crop


disease_info = pd.read_csv('./static/data/disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('./static/data/supplement_info.csv', encoding='cp1252')
fruit_info = pd.read_csv('./static/data/fruit_info.csv', encoding='cp1252')
crop_info = pd.read_csv('./static/data/crop_info.csv', encoding='cp1252')

model = CNN.CNN(39)    
model.load_state_dict(torch.load("./static/models/plant_disease_model.pt"))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index


app = Flask(__name__)

@app.route('/')
def home_page():
    fruits = fruit_info.to_dict(orient='records')
    return render_template('home.html', fruits=fruits)

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/disease-detection')
def disease_detection_page():
    return render_template('disease-detection.html')

@app.route('/crop-recommendation')
def crop_recommendation_page():
    return render_template('crop-recommendation.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description =disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('detect.html' , title = title , desc = description , prevent = prevent , 
                               image_url = image_url , pred = pred ,sname = supplement_name , simage = supplement_image_url , buy_link = supplement_buy_link)
    
@app.route("/recommend", methods=['GET', 'POST'])
def recommend():
  nitrogen = float(request.form["nitrogen"])
  phosphorus = float(request.form["phosphorus"])
  potassium = float(request.form["potassium"])
  temperature = float(request.form["temperature"])
  humidity = float(request.form["humidity"])
  ph = float(request.form["ph"])
  rainfall = float(request.form["rainfall"])
  features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
  predicted_crop = predict_crop(features)
  filtered_data = crop_info[crop_info['Crop'] == predicted_crop]
  title = filtered_data['Title'].values[0]
  image_url = filtered_data['Image_URL'].values[0]
  return render_template("recommend.html", title=title, result=predicted_crop.capitalize(), image_url=image_url)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
