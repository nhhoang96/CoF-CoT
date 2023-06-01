# ---- Just need to update processing code to read sentences from a file ----#
# Prompt still requires populating the results from GPT models in the previous steps (still left as blank for now) #

# python write_prompt # Independent steps (w/o any intent conditioning)
# Python write_prompt --type_condition codition # Conditioned based on "potential intent types"
import json
import os
import time
import argparse
import re
import openai

openai.api_key = os.getenv("openai_key")
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
input_test_file = "nlu_data/mtop_flat_simple/en/demo_100.txt"
write_output= 'nlu_data/mtop_flat_simple/en/demo_100_label.txt'

with open(input_test_file, 'r') as file:
    content = file.read()

gen_step_1b = 'Given the sentence, generate one and only one Abstract Meaning Representation (AMR) graph representation in the textual Neo-Davidsonian format \n'

content = content.split("\n")
for example in content[0:1]:
    utterance, logical_form, _, _, tag = example.split("\t")
    output = example.split('\t')
    utterance=output[0]

    # --- Step 1b: Get AMR Graph
    step_1b_prompt = gen_step_1b + 'Sentence: ' + utterance + '\n'
    
    step_1b_prompt += 'AMR Graph: ' + '\n'
    amr_graph = call_chatgpt(step_1b_prompt)
    #print("STEP 1b: Get AMR Graph", step_1b_prompt)

    output.append(amr_graph)
    write_output = '\t'.join(output)
    
    #writer.write(json.dumps({"utterance": utterance, "intent": intent, "AMR Graph": amr_graph, "key_phrase":
    #    key_phrases, "slot_type": slot_type, "pred_lf": pred_lf, "gold_lf": logical_form}) + '\n')
