from flask import Flask, redirect, url_for, render_template, request
import os
import pandas as pd
import json, ast, re
import openai

from ailogic.functions import compare_laptops_with_user, dictionary_present, get_chat_completions, initialize_conv_reco, initialize_conversation, intent_confirmation_layer, moderation_check, recommendation_validation

openai.api_key = open("ailogic/api_key.txt", "r").read().strip()

app = Flask(__name__)

# OpenAI conversation
conversation = initialize_conversation()
introduction = get_chat_completions(conversation)
top_3_laptops = None

# Bot conversation
conversation_bot = []
conversation_bot.append({'bot':introduction})

@app.route("/")
def default_func():
    global conversation_bot, conversation, top_3_laptops, conversation_reco
    return render_template("index_invite.html", name = conversation_bot)

@app.route("/end_conv", methods = ['POST', 'GET'])
def end_conv():
    global conversation_bot, conversation, top_3_laptops, conversation_reco
    # OpenAI conversation
    conversation = initialize_conversation()
    introduction = get_chat_completions(conversation)
    # Bot conversation
    conversation_bot = []
    conversation_bot.append({'bot':introduction})  
    top_3_laptops = None
    return redirect(url_for('default_func'))

@app.route("/invite", methods = ['POST'])
def invite():
    global conversation_bot, conversation, top_3_laptops, conversation_reco
    user_input = request.form["user_input_message"]
    prompt = 'Remember your system message and that you are an intelligent laptop assistant. So, you only help with questions around laptop.'
    moderation = moderation_check(user_input)
    if moderation == 'Flagged':
        return redirect(url_for('end_conv'))
        
    if top_3_laptops is None:
        conversation.append({"role": "user", "content": user_input + prompt})
        conversation_bot.append({'user': user_input})

        response_assistant = get_chat_completions(conversation)
        moderation = moderation_check(str(response_assistant))
        if moderation == 'Flagged':
            return redirect(url_for('end_conv'))

        confirmation = intent_confirmation_layer(response_assistant)
        moderation = moderation_check(str(confirmation))
        if moderation == 'Flagged':
            return redirect(url_for('end_conv'))

        if "No" in confirmation.get('result'):
            conversation.append({"role": "assistant", "content": str(response_assistant)})
            conversation_bot.append({'bot': response_assistant})            

        else:

            response = dictionary_present(response_assistant)
            moderation = moderation_check(response)
            if moderation == 'Flagged':
                return redirect(url_for('end_conv'))

            conversation_bot.append({'bot': 'Thank you for providing all the information. Kindly wait, while I fetch the products:'})
            top_3_laptops = compare_laptops_with_user(response)

            validated_reco = recommendation_validation(top_3_laptops)

            if len(validated_reco) == 0:
                conversation_bot.append({'bot': 'Sorry, we do not have laptops that match your requirements. Connecting you to human expert. Please end this conversation.'})

            conversation_reco = initialize_conv_reco(validated_reco)
            recommendation = get_chat_completions(conversation_reco)
            
            moderation = moderation_check(recommendation)
            if moderation == 'Flagged':
                return redirect(url_for('end_conv'))
                
            conversation_reco.append({"role": "user", "content": "This is my user profile" + str(response)})

            conversation_reco.append({"role": "assistant", "content": str(recommendation)})
            conversation_bot.append({'bot': recommendation})

    else:
        conversation_reco.append({"role": "user", "content": user_input})
        conversation_bot.append({'user': user_input})

        response_asst_reco = get_chat_completions(conversation_reco)

        moderation = moderation_check(response_asst_reco)
        if moderation == 'Flagged':
            return redirect(url_for('end_conv'))
                    
        conversation.append({"role": "assistant", "content": response_asst_reco})
        conversation_bot.append({'bot': response_asst_reco})
    return redirect(url_for('default_func'))

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
