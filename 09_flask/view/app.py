from flask import Flask, render_template

app = Flask(__name__)

@app.route('/<user_name>')
def index(user_name):
    item_list = ["Apple", "Banana", "Cherry"]
    return render_template('index.html', name=user_name, items=item_list)

if __name__ == '__main__':
    app.run(debug=True)