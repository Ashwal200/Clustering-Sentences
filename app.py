from flask import Flask, render_template, request
from app_model import Model

app = Flask(__name__)

@app.before_request
def setup():
    """
    Code to be executed before the first request is handled.
    Initializes the model with a list of URLs to patent documents.
    """
    global model
    urls = [
        "https://patents.google.com/patent/GB2478972A/en?q=(phone)&oq=phone",
        "https://patents.google.com/patent/US9980046B2/en?oq=US9980046B2",
        "https://patents.google.com/patent/US9634864B2/en?oq=US9634864B2"
    ]
    model = Model(urls)

@app.route('/')
def index():
    """
    Route to handle the home page.
    Renders the index.html template.
    """
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    """
    Route to handle form submissions.
    Validates the input and calls the model to perform KMeans clustering.
    
    Returns:
        Renders the index.html template with the output of the model or an error message.
    """
    number = request.form['number']
    
    # Server-side validation to ensure input is numeric and less than or equal to 30
    if not number.isdigit() or int(number) > 30:
        return render_template('index.html', output=False)
    
    try:
        # Call the model's method to perform KMeans clustering
        model.model_KMeansW2V(int(number))
        
        # Get the result from the model
        result = model.get_result()
        
        # Render the result in the output of the template
        return render_template('index.html', output=result)
    except Exception as e:
        # Return the error message in case of an exception
        return str(e)

if __name__ == '__main__':
    # Run the Flask application with debug mode enabled on port 5004
    app.run(debug=True, port=5005)
