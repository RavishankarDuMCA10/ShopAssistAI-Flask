import os
import pandas as pd
import json, ast, re
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt

def initialize_conversation():
    '''
    Returns a list [{"role": "system", "content": system_message}]
    '''

    delimiter = "####"

    example_user_dict = {"GPU intensity": "high",
                        "Display quality": "high",
                        "Portability": "low",
                        "Multitasking": "high",
                        "Processing speed": "high",
                        "Budget": "150000"
                        }

    system_message = f"""
        You are an intelligent laptop gadget expert and your goal is to find the best laptop for a user.
        You need to ask relevent questions and understand the user profile by analysing the user's responses.
        Your final objective is to fill the values for the different keys ('GPU intensity', 'Display quality', 'Portability', 'Multitasking', 'Processing speed', 'Budget').
        These key value pairs define the user's profile.
        The python dictionary looks like this {{'GPU intensity':'values', 'Display quality':'values', 'Portability':'values', 'Multitasking':'values', 'Processing speed':'values', 'Budget':'values'}}
        The values for all keys, except 'Budget', should be 'low', 'medium', or 'high' based on the importance of the corresponding keys, as stated by user.
        The value for 'Budget' should be a numerical value extracted from the user's response.
        The values currently in the dictionary are only representive values.
        {delimiter}
        Here are some instructions around the values for the different keys. If you do not follow this, you'll be heavily penalised.
        - The value for all keys, except 'Budget', should strictly be 'low', 'medium', or 'high' based on the importance of the corresponding keys, as stated by user.
        - The value for 'Budget' should be a numerical value extracted from the user's response.
        - 'Budget' value needs to be greater than or equal to 25000 INR. If the user says less than that, please mention that there are no laptops in that range.
        - Do not randomly assign values to any of the keys. The values need to be inferred from the user's response.
        {delimiter}

        To fill the dictionary, you need to have the following chain of thoughts:
        {delimiter} Thought 1: Ask a question to understand the user's profile and requirements. \n
        If their primary use for the laptop is unclear. Ask another question to comprehend their needs.
        You are trying to fill the values of all the keys ('GPU intensity', 'Display quality', 'Portability', 'Multitasking', 'Processing speed', 'Budget') in the python dictionary by understanding the user requirements.
        Identify the keys for which you can fill the values confidently using the understanding. \n
        Remember the instructions around the values for the different keys.
        Answer "Yes" or "No" to indicate if you uderstand the requirements and have updated the values for the relevant keys. \n
        If yes, proceed to next step. Otherwise, rephrase the question to capture their profile. \n{delimiter}

        {delimiter} Thought 2: Now, you are trying to fill values for the rest of the keys which you couldn't in the previous step.
        Remember the instructions around the values for the different keys. Ask questions you might have for all the keys to strengthen your understanding of the user's profile.
        Answer "Yes" or "No" to indicate if you understood all the values for the keys and are confident about the same.
        If yes, move to the next Thought. If no, ask question on the keys whose values you are unsure of. \n
        It is a good practice to ask question with a sound logic as opposed to directly citing the key you want to understand value for.{delimiter}

        {delimiter}Thought 3: Check if you have correctly updated the values for the different keys in the python dictionary.
        If you are not confident about any of the values, ask clarifying questions.{delimiter}

        Follow the above chain of thoughts and only output the final updated python dictionay. \n

        {delimiter}Here is a sample conversation between the user and assistant:
        User: "Hi, I am an editor."
        Assistant: "Great! As an editor, you likely require a laptop that can handle demanding tasks. Hence, the laptop should have high multitasking capability. You would also need a high end display for better visuals and editing. May I know what kind of work do you primarily focus on? Are you more involved in video editing, photo editing, or both? Understanding the specific type of editing work will help me tailor my recommendations accordingly. Let me know if my understanding is correct until now."
        User: "I primarily work with After Effects."
        Assistant: "Thank you for providing that information. Working with After Effects involves working with graphics, animations, and rendering, which will require high GPU. Do you work with high-resolution media files, such as 4K videos or RAW photos? Understanding your file sizes will help determine the storage capacity and processing power needed."
        User: "Yes, sometimes I work with 4K videos as well."
        Assistant: "Thank you for the information. Processing 4K vidoes will require a good processor and high GPU. I think we have already determined earlier that you need a high GPU. To ensure I have a complete understanding of your needs, I have one more question: Are you frequently on the go and require a laptop that is lightweight and easy to carry, or do you primarily work from a stationary location?"
        User: "Yes, sometimes I travel but do not carry my laptop."
        Assistant:"Could you kindly let me know your budget for the laptop? This will help me find options that fit within your price range while meeting the specified requirements."
        User: "my max budget is 1.5lakh inr"
        Assistant: {example_user_dict}
        {delimiter}

        Start with a your introduction message as You are an intelligent laptop gadget expert and encourage the user to share their requirements.
    """
    conversation = [{"role": "system", "content": system_message}]

    return conversation


# Define a Chat Completions API call
# Retry up to 6 times with exponential backoff, starting at 1 second and maxing out at 20 second delay
def get_chat_completions(input, json_format = False):
    """
    Generate chat completions using OpenAI's GPT-3.5-turbo model.

    Parameters:
    - input: list of dictionaries representing chat messages to prompt the model.
    - json_format: boolean, specifies whether the output should be in JSON format (default: False).

    Returns:
    - output: The generated chat completion response. If json_format is True, returns a JSON object.
    """
    MODEL = 'gpt-3.5-turbo'

    response = openai.chat.completions.create(
          model= MODEL,
          messages= input,
          temperature = 0,
          max_tokens = 300,
          seed= 2345
      )
    
    return response.choices[0].message.content
    
# Define a function called moderation_check that takes user_input as a parameter.
def moderation_check(user_input):
    """
    Perform moderation check on user input using the OpenAI API.

    Parameters:
    - user_input (str): The text input to be checked for moderation.

    Returns:
    - str: "Flagged" if the input is flagged by the moderation system,
           "Not Flagged" otherwise.
    """
    # Call the OpenAI API to perform moderation on the user's input.
    response = openai.moderations.create(input=user_input)

    # Extract the moderation result from the API response.
    moderation_output = response.results[0].flagged

    # Check if the input was flagged by the moderation system. If flagged, return "Flagged" else return "Not Flagged"
    if moderation_output == True:
      return "Flagged"
    else:
      return "Not Flagged"

def iterate_llm_response(funct, debug_response, num=10):
    """
    Calls a specified function repeatedly and prints the results.

    This function is designed to test the consistency of a response from a given function.
    It calls the function multiple times (default is 10) and prints out the iteration count,
    the function's response(s).

    Args:
        funct (function): The function to be tested. This function should accept a single argument
                          and return the response value(s).
        debug_response (dict): The input argument to be passed to 'funct' on each call.
        num (int, optional): The number of times 'funct' will be called. Defaults to 10.

    Returns:
        This function only returns the results to the console.

    Example usage: Test the consistency of responses from 'intent_confirmation_layer' function
    iterate_llm_response(get_chat_completions, messages)
    """

    for i in range(1, num + 1):
      response = funct(debug_response)
      print(f"Iteration {i}: {response}\n")
      print('-' * 50)


    
def intent_confirmation_layer(response_assistant):
    """
    This function takes in a response from an AI assistant and evaluates it based on specific criteria.

    Args:
    - response_assistant (str): The input text containing user requirements captured through 6 keys:
      'GPU intensity', 'Display quality', 'Portability', 'Multitasking', 'Processing speed', and 'Budget'.

    Returns:
    - dict: A JSON-formatted dictionary with keys 'result' and, if applicable, 'reason'.
      - 'result' (str): Either 'Yes' or 'No', indicating if the input meets the criteria.
      - 'reason' (str, optional): If 'result' is 'No', this key provides a reason for the evaluation.

    Key Instructions:
    1. The input should include values for the specified keys.
    2. Allowed values for 'GPU intensity', 'Display quality', 'Portability', 'Multitasking', and 'Processing speed'
       are 'low', 'medium', and 'high'.
    3. 'Budget' should only have a numerical value.
    4. Only a one-word string in JSON format ('Yes' or 'No') should be output at the key 'result'.
    5. If 'result' is 'No', include the reason in the key 'reason'.
    6. Use the AI model 'gpt-3.5-turbo' to process the evaluation.

    Example:
    >>> input_text = "{'GPU intensity': 'high', 'Display quality': 'medium', 'Portability': 'low',
                      'Multitasking': 'medium', 'Processing speed': 'high', 'Budget': 1200}"
    >>> intent_confirmation_layer(input_text)
    {'result': 'Yes'}

    Note: Modify the input text to match the expected format before passing it to this function.
    """

    MODEL = 'gpt-3.5-turbo'

    delimiter = "####"

    allowed_values = {'low', 'medium', 'high'}

    prompt = f"""
    You are senior evaluator who has an eye for detail. The input text will contain a user requirement captured through 6 keys.
    You are provided an input. You need to evaluate if the input text has the following keys:
    {{
      'GPU intensity': 'values',
      'Display quality': 'values',
      'Portability': 'values',
      'Multitasking': 'values',
      'Processing speed': 'values',
      'Budget': 'number'
    }}
    The values for the keys should only be from the allowed values: {allowed_values}.
    The 'Budget' key can take only a numerical value.
    Next you need to evaluate if the keys have the the values filled correctly.
    Only output a one-word string in JSON format at the key 'result' - Yes/No.
    Thought 1 - Output a string 'Yes' if the values are correctly filled for all keys, otherwise output 'No'.
    Thought 2 - If the answer is No, mention the reason in the key 'reason'.
    Thought 3 - Think carefully before the answering.

    You are an intelligent laptop gadget expert and encourage the user to share their requirements. You do not need to say that you will need to update dictionary and about thought process.
    """

    messages = [{"role":"system", "content":prompt},
                {"role": "user", "content":f"""Here is the input: {response_assistant}"""}]

    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type":"json_object"},
        seed = 1234
    )

    return json.loads(response.choices[0].message.content)


def dictionary_present(response):
    delimiter = "####"

    user_req = {'GPU intensity': 'high',
                'Display quality': 'high',
                'Portability': 'medium',
                'Multitasking': 'high',
                'Processing speed': 'high',
                'Budget': '200000'}

    prompt = f"""You are a python expert. You are provided an input.
            You have to check if there is a python dictionary present in the string.
            It will have the following format {user_req}.
            Your task is to just extract the relevant values from the input and return only the python dictionary in JSON format.
            The output should match the format as {user_req}.

            {delimiter}
            Make sure that the value of budget is also present in the user input. ###
            The output should contain the exact keys and values as present in the input.
            Ensure the keys and values are in the given format:
            {{
              'GPU intensity': 'low/ medium/ high',
              'Display quality': 'low/ medium/ high',
              'Portability': 'low/ medium/ high',
              'Multitasking': 'low/ medium/ high',
              'Processing speed': 'low/ medium/ high',
              'Budget': 'numerical value'
            }}
            Here are some sample input output pairs for better understanding:
            {delimiter}
            input 1: - GPU intensity: low - Display quality: high - Portability: low - Multitasking: high - Processing speed: medium - Budget: 50,000 INR
            output 1: {{'GPU intensity': 'low', 'Display quality': 'high', 'Portability': 'low', 'Multitasking': 'high', 'Processing speed': 'medium', 'Budget': '50000'}}

            input 2: {{'GPU intensity':     'low', 'Display quality':     'low', 'Portability':    'medium', 'Multitasking': 'medium', 'Processing speed': 'low', 'Budget': '90,000'}}
            output 2: {{'GPU intensity': 'low', 'Display quality': 'low', 'Portability': 'medium', 'Multitasking': 'medium', 'Processing speed': 'low', 'Budget': '90000'}}

            input 3: Here is your user profile 'GPU intensity': 'high','Display quality': 'high','Portability': 'medium','Multitasking': 'high','Processing speed': 'high','Budget': '200000 INR'
            output 3: {{'GPU intensity': 'high','Display quality': 'high','Portability': 'medium','Multitasking': 'high','Processing speed': 'high','Budget': '200000'}}
            {delimiter}
            """
    messages = [{"role": "system", "content":prompt },
                {"role": "user", "content":f"""Here is the user input: {response}""" }]

    confirmation = get_chat_completions(messages)

    return confirmation

def product_map_layer(laptop_description):
    delimiter = "#####"

    lap_spec = {
          "GPU intensity":"(Type of the Graphics Processor)",
          "Display quality":"(Display Type, Screen Resolution, Display Size)",
          "Portability":"(Laptop Weight)",
          "Multitasking":"(RAM Size)",
          "Processing speed":"(CPU Type, Core, Clock Speed)"
    }

    values = {'low', 'medium', 'high'}

    prompt=f"""
    You are a Laptop Specifications Classifier whose job is to extract the key features of laptops as per their requirements.
    To analyze each laptop, perform the following steps:
    Step 1: Extract the laptop's primary features from the description {laptop_description}
    Step 2: Store the extracted features in {lap_spec} \
    Step 3: Classify each of the items in {lap_spec}into {values} based on the following rules: \
    {delimiter}
    GPU Intensity: low: < for Integrated graphics or entry-level dedicated graphics like Intel UHD > , \n
    medium: < if Mid-range dedicated graphics like M1, AMD Radeon, Intel Iris > , \n
    high: < High-end dedicated graphics like Nvidia > , \n

    Display Quality: low: < if resolution below Full HD (e.g., 1366x768). > , \n
    medium: < if Full HD resolution (1920x1080) or higher. > , \n
    high: < if High-resolution display (e.g., 4K, Retina) with excellent color accuracy and features like HDR support. > \n

    Portability: low: < if laptop weight is less than 1.51 kg > , \n
    medium: < if laptop weight is between 1.51 kg and 2.51 kg> , \n
    high: <if laptop weight is greater than 2.51 kg> \n

    Multitasking: low: < If RAM size is 8GB, 12GB > , \n
    medium: < if RAM size is 16GB > , \n
    high: < if RAM size is 32GB, 64GB> \n

    Processing Speed: low: < if entry-level processors like Intel Core i3, AMD Ryzen 3> , \n
    medium: < if Mid-range processors like Intel Core i5, AMD Ryzen 5 > , \n
    high: < if High-performance processors like Intel Core i7, AMD Ryzen 7 or higher > \n
    {delimiter}

    {delimiter}
    Here are some input output pairs for few-shot learning:
    input1: "The Dell Inspiron is a versatile laptop that combines powerful performance and affordability. It features an Intel Core i5 processor clocked at 2.4 GHz, ensuring smooth multitasking and efficient computing. With 8GB of RAM and an SSD, it offers quick data access and ample storage capacity. The laptop sports a vibrant 15.6" LCD display with a resolution of 1920x1080, delivering crisp visuals and immersive viewing experience. Weighing just 2.5 kg, it is highly portable, making it ideal for on-the-go usage. Additionally, it boasts an Intel UHD GPU for decent graphical performance and a backlit keyboard for enhanced typing convenience. With a one-year warranty and a battery life of up to 6 hours, the Dell Inspiron is a reliable companion for work or entertainment. All these features are packed at an affordable price of 35,000, making it an excellent choice for budget-conscious users."
    output1: {{'GPU intensity': 'medium','Display quality':'medium','Portability':'medium','Multitasking':'high','Processing speed':'medium'}}

    input2: "The Lenovo ThinkPad X1 Carbon is a sleek and lightweight laptop designed for professionals on the go. It is equipped with an Intel Core i7 processor running at 2.6 GHz, providing strong processing capabilities for multitasking and productivity. With 16GB of RAM and an SSD, it offers fast and efficient performance along with ample storage capacity. The laptop features a 14" IPS display with a resolution of 2560x1440, delivering sharp visuals and accurate colors. It comes with Intel UHD integrated graphics for decent graphical performance. Weighing just 1.13 kg, it is extremely lightweight and highly portable. The laptop features an IR camera for face unlock, providing convenient and secure login options. With a three-year warranty and an impressive battery life of up to 12 hours, the Lenovo ThinkPad X1 Carbon ensures reliability and long-lasting productivity. Priced at 130,000, it offers top-notch performance and portability for professionals."
    output2: {{'GPU intensity': 'medium', 'Display quality': 'high', 'Portability': 'high', 'Multitasking':'high', 'Processing speed':'high'}}

    input3: "The Apple MacBook Pro is a high-end laptop that combines top-tier performance with a stunning display. It is equipped with an Intel Core i9 processor running at 2.9 GHz, providing exceptional processing power for demanding tasks and content creation. With 32GB of RAM and an SSD, it offers seamless multitasking and fast storage access for large projects. The laptop features a 16" Retina display with a resolution of 3072x1920, delivering breathtaking visuals and precise color reproduction. It comes with an AMD Radeon graphics card, ensuring smooth graphics performance for professional applications. Weighing 2.02 kg, it is relatively lightweight for its size. The laptop features a True Tone display, adjusting the color temperature to match the ambient lighting for a more natural viewing experience. With a three-year warranty and a battery life of up to 10 hours, the Apple MacBook Pro offers reliability and endurance for professionals. Priced at 280,000, it caters to users who require uncompromising performance and a superior display for their demanding workloads."
    output3: {{'GPU intensity': 'medium', 'Display quality': 'high', 'Portability': 'medium','Multitasking': 'high', 'Processing speed': 'high'}}
    {delimiter}
    """

    input = f"""Follow the prompt instructions step-by-step and output the dictionary in {lap_spec} JSON format for the following laptop {laptop_description}. You are an intelligent laptop gadget expert and encourage the user to share their requirements. You do not need to say that you will need to update dictionary and about thought process.
"""

    messages=[{"role": "system", "content":prompt },{"role": "user","content":input}]

    response = get_chat_completions(messages)

    return response


def extract_dictionary_from_string(string):
  regex_pattern = r"\{[^{}]+\}"

  dictionary_matches = re.findall(regex_pattern, string)

  #Extract the first dictionary match and convert it to lower case
  if dictionary_matches:
    dictionary_string = dictionary_matches[0]
    dictionary_string = dictionary_string.lower()

    # Convert the dictionary string to a dictionary object using ast.literal_eval()
    dictionary = ast.literal_eval(dictionary_string)

  return dictionary

def compare_laptops_with_user(user_req_string):
    # Extracting user requirements from the input string (assuming it's a dictionary)
    laptop_df = pd.read_csv('updated_laptop.csv')
    user_requirements = extract_dictionary_from_string(user_req_string)
    budget = int(user_requirements.get('budget', '0').replace(',','').split()[0])

    filtered_laptops = laptop_df.copy()
    filtered_laptops['Price'] = filtered_laptops['Price'].str.replace(',','').astype(int)
    filtered_laptops = filtered_laptops[filtered_laptops['Price'] <= budget].copy()

    # Create a scoring mechanism for comparing the laptops in the laptop descriptions against the user_req_string containing the user's persona
    mapping_scores = {
        'low': 0,
        'medium': 1,
        'high': 2
    }

    filtered_laptops['Score'] = 0
    for index, row in filtered_laptops.iterrows():
      user_product_match_str = row['laptop_feature']
      laptop_values = extract_dictionary_from_string(user_product_match_str)
      score = 0

      for key, user_value in user_requirements.items():
        if key.lower() == 'budget':
          continue
        laptop_value = laptop_values.get(key, None)
        laptop_mapping = mapping_scores.get(laptop_value.lower(), -1)
        user_mapping = mapping_scores.get(user_value.lower(), -1)
        if laptop_mapping >= user_mapping:
          score += 1

      filtered_laptops.loc[index, 'Score'] = score

    # Sorting laptops by score in descending order and selecting the top 3 products
    top_laptops = filtered_laptops.drop('laptop_feature', axis=1)
    top_laptops = top_laptops.sort_values('Score', ascending=False).head(3)

    # top_laptops
    return top_laptops.to_json(orient='records')


def recommendation_validation(laptop_recommendation):
    """
    Validate a list of laptop recommendations based on a score threshold.

    Parameters:
    - laptop_recommendation (str): JSON string containing a list of laptop recommendations,
      each with a 'Score' key representing its rating.

    Returns:
    - list: A filtered list of laptop recommendations where each item has a 'Score' greater than 2.
    """
    data = json.loads(laptop_recommendation)
    data_1 = []
    for i in range(len(data)):
      if data[i]['Score'] > 2:
        data_1.append(data[i])

    return data_1


def initialize_conv_reco(products):
    system_message = f"""
    You are an intelligent laptop gadget expert and you are tasked with the objective to \
    solve the user queries about any product from the catalogue in the user message \
    You should keep the user profile in mind while answering the questions.\

    Start with a brief summary of each laptop in the following format, in decreasing order of price of laptops:
    1. <Laptop Name> : <Major specifications of the laptop>, <Price in Rs>
    2. <Laptop Name> : <Major specifications of the laptop>, <Price in Rs>

    """
    user_message = f""" These are the user's products: {products}"""
    conversation = [{"role": "system", "content": system_message },
                    {"role":"user","content":user_message}]
    # conversation_final = conversation[0]['content']
    return conversation


