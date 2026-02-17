from flask import Flask, request, render_template

# Flaskアプリケーションのインスタンスを作成
app = Flask(__name__)

# ルート '/' にアクセスした際に 'index.html' を表示
@app.route('/')
def index():
    return render_template('index.html')

# ルート '/submit' にPOSTリクエストが来たときの処理
@app.route('/submit', methods=['POST'])
def submit():
    # フォームから送信された 'name' と 'email' を取得
    name = request.form.get('name')
    email = request.form.get('email')

    # サーバ側のバリデーション
    if not name or not email:
        return "Error: All fields are required!"
    if "@" not in email:
        return "Error: Invalid email address!"

    # 取得したデータを 'submit.html' テンプレートに渡して表示
    return render_template('submit.html', name=name, email=email)

# アプリケーションをデバッグモードで実行
if __name__ == '__main__':
    app.run(debug=True)