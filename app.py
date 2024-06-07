from flask import Flask, render_template, request
from app_model import Model

app = Flask(__name__)

@app.before_request
def setup():
    # Code to be executed before the first request is handled
    global model
    urls = ["https://patents.google.com/patent/GB2478972A/en?q=(phone)&oq=phone",
            "https://patents.google.com/patent/US9980046B2/en?oq=US9980046B2",
            "https://patents.google.com/patent/US9634864B2/en?oq=US9634864B2"]
    model = Model(urls)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():

    number = request.form['number']
    if not number.isdigit() or int(number) > 30:  # Server-side validation to ensure input is numeric
        return render_template('index.html', output=False)
    
    # Call the task_2.py script with the input number
    try:
        model.model_KMeansW2V(number)
        result = model.get_result()
        output = result 
        return render_template('index.html', output=output)
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True, port=5004)
