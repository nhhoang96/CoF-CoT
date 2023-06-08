# ---- Just need to update processing code to read sentences from a file ----#
# Prompt still requires populating the results from GPT models in the previous steps (still left as blank for now) #

# python write_prompt # Independent steps (w/o any intent conditioning)
# Python write_prompt --type_condition codition # Conditioned based on "potential intent types"
import json
import os
import time
from tqdm import tqdm
import argparse
import re
import openai

#openai.api_key = "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"
openai.api_key = "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"
model_name = "gpt-3.5-turbo"

def call_chatgpt(input_prompt):
    success = False
    while success == False:
        try:
            response = openai.ChatCompletion.create(
                model = model_name,
                messages=[
                    {"role": "user", "content": input_prompt}
                ]
            )
            success = True
        except:
            time.sleep(1)
    response = response["choices"][0]['message']['content'].strip()
    return response

# ----- Output file definition ------#
type_info=['intent', 'slot']
for t in type_info:
    input_test_file = "nlu_data/mtop_flat_simple/" + t + "_vocab.txt"
    write_output= "nlu_data/mtop_flat_simple/" + t + "_vocab_map.jsonl"

    writer = open(write_output, 'w')

    with open(input_test_file, 'r') as file:
        content = file.read()

    gen_step_1b = 'Given the following label, generate the label description between 5-10 words: \n'


    content = content.split("\n")
    for example in tqdm(content):
        label = example.strip()

        # --- Step 1b: Get AMR Graph
        step_1b_prompt = gen_step_1b + 'Label: ' + label + '\n'
        
        step_1b_prompt += 'Description: ' + '\n'
        write_output = call_chatgpt(step_1b_prompt)
        print("STEP 1b: Get AMR Graph", step_1b_prompt)
 
        writer.write(json.dumps({"label": label,  "label_description": write_output}) + '\n'
            )
