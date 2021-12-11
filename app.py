from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import PIL

app = Flask(__name__)

class_names = {0: 'Ayam Bakar',
  1: 'Ayam Geprek',
 2: 'Mie Goreng',
 3: 'Nasi Goreng',
 4: 'Sate'}

model = load_model('Gambar_retrain.h5')

model.make_predict_function()

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(300,300))
	i = image.img_to_array(i)/255.0
	i = np.expand_dims(i, 0)
	p = model.predict(i)
	return class_names[np.argmax(p)]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)

if __name__ =='__main__':
	app.debug = True
	app.run()
