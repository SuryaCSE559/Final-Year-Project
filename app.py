from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template('page.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    args = [x for x in request.form.values()]
    l = list(args)
    input_data = vectorizer.transform(l)
    output = model.predict(input_data)
    accr = 'Accuracy score: {}'.format(98.47533632286996)
    prec = 'Precision score: {}'.format(94.20289855072463)
    rec = 'Recall score: {}'.format(93.5251798561151)
    f1 = 'F1 score: {}'.format(93.86281588447652)
    if output.item() == 1:
       return render_template('page.html', pred='Your message is containing Spam content', accuracy=accr, precision=prec, recall=rec, f1score=f1)
    else:
       return render_template('page.html', pred='Your message is Genuine',accuracy=accr, precision=prec, recall=rec, f1score=f1)

if __name__ == '__main__':
    app.run(debug=True)