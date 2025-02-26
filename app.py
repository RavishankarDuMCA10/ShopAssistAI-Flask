from flask import Flask, redirect, url_for, render_template, request

app = Flask(__name__)

output = 'Please input your name'

@app.route("/")
def default_func():
    global output
    return render_template("index_invite.html", name = output)

@app.route("/end_conv", methods = ['POST'])
def end_conv():
    global output
    output = 'Please input your name'
    return redirect(url_for('default_func'))

@app.route("/invite", methods = ['POST'])
def invite():
    global output
    name = request.form["user_input_message"]
    if name == 'Nikhil':
        output = 'Bye, you are not invited to the event, ' + name
    else:
        output = 'Hello, you are invited to the event, ' + name
    return redirect(url_for('default_func'))

if __name__ == '__main__':
    app.run(debug=True)
