#-*- coding: utf-8 -*-
from app_to_model import Model
from tkinter import dialog
from flask import Flask, render_template, request, url_for

#Initialize
app = Flask(__name__)
model = Model('./model/wozdebugging0.1.pt')
#Define a route for url

@app.route('/')
def form():
	return render_template('form_submit.html')

#form action
@app.route('/hello', methods=['POST'] )
def action():
    dialog = request.form['dialog']
    result = model.get_result(dialog)
    return render_template('form_action.html', result=result)

#Run the app
if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=True)