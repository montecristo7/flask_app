from flask import Flask,escape, request,render_template,jsonify
import flask
import time

app = Flask(__name__)

'''
def model(i):
    a = 10000
    while a > 0:
        print(f"Running epoch {a}")
        a -= 10
    if i == "binary":
        return "Binary model starts running..."
    elif i == "class":
        return "Class model starts running..."
    elif i == "species":
        return "Species model starts running..."
    else:
        return "ERROR: No model selected"
'''

@app.route('/')
def hello():
    return render_template('gui.html')

@app.route('/model',methods=['GET','POST'])
def choose_type():
    if request.method == "POST":
        text = request.form['submit_button']
         
        
        def inner():
            if text == "binary":
                yield " Binary model starts running... \n "
            elif text == "class":
                yield "Class model starts running... \n"
            elif text == "species":
                yield "Species model starts running... \n "
            for x in range(10):
                time.sleep(1)
                yield f"Running currently {x}s \n "
                
        return flask.Response(inner(), mimetype='text/html')

if __name__ == '__main__':
    app.run(debug=True)