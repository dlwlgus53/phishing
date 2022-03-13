#-*- coding: utf-8 -*-
from app_to_model import Model
from tkinter import dialog
from flask import Flask, render_template, request, url_for

#Initialize
app = Flask(__name__)
model = Model('./model/wozphishing1.0.pt')
#Define a route for url

@app.route('/')
def form():
	return render_template('form_submit.html')

#form action
@app.route('/hello', methods=['POST'] )
def action():
    dialog = request.form['dialog']
    result = model.get_result(dialog)
    result = {
        '검찰,경찰,금감원 사칭 - 연루범죄 ' : '통장사기',
        '검찰,경찰,금감원 사칭 - 기관' : '검찰',
        '검찰,경찰,금감원 사칭 - 이름' : '김영진',
        '대출상담 - 직급' : '해당없음',
        '대출상담 - 회사' : '해당없음',
        '대출상담 - 이름' : '해당없음',
        '납치 - 요구사항' : '해당없음',
        '납치 - 계좌' : '해당없음',
        '납치 - 은행' : '해당없음',
        '납치 - 피해자와의 관계' : '해당없음',
        '가족지인사칭 - 요구사항' : '해당없음',
        '가족지인사칭 - 피해자와의 관계' : '해당없음',
        '가족지인사칭 - 계좌번호' : '해당없음',
        '가족지인사칭 - 은행' : '해당없음',

    }
    return render_template('form_action.html', result=result)

#Run the app
if __name__ == '__main__':
	app.run(host='0.0.0.0', port = 9000, debug=True)