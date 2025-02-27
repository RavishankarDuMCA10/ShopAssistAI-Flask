from flask import Flask, redirect, url_for, render_template, request

app = Flask(__name__)

conversation = []
conversation.append({'bot':'Please input your name'})

@app.route("/")
def default_func():
    global conversation
    return render_template("index_invite.html", name = conversation)

@app.route("/end_conv", methods = ['POST'])
def end_conv():
    print("end_conv called.")    
    conversation = []
    conversation.append({'bot':'Please input your name'})
    print("conversation cleared and added initial bot message.")   
    return render_template("index_invite.html", name = conversation) 

@app.route("/invite", methods = ['POST'])
def invite():
    global conversation
    name = request.form["user_input_message"]
    conversation.append({'user': name})
    if name == 'Nikhil':
        output = 'Bye, you are not invited to the event, ' + name
    else:
        output = 'Hello, you are invited to the event, ' + name

    conversation.append({'bot': output})
    return redirect(url_for('default_func'))

if __name__ == '__main__':
    app.run(debug=True)
