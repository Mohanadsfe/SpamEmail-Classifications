from flask import Flask,render_template,request
import pickle
from model_m2 import feature_extraction
app = Flask(__name__)
from model_m2 import process_email

pipe = pickle.load(open("LR_Model.pkl","rb"))

@app.route('/', methods=["GET","POST"])
def main_function():

    if request.method == "POST":
        text = request.form
        emails = text['email']
        
        # Reshape the input data into a 2D array
        processed_email = process_email(emails)
        input_data_features = feature_extraction.transform([processed_email])
            
        # Making prediction
        output = pipe.predict(input_data_features)[0]
        return render_template("show.html", prediction=output)
    
    else:
        return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True,port=5001)