# ---- Just need to update processing code to read sentences from a file ----#
# Prompt still requires populating the results from GPT models in the previous steps (still left as blank for now) #

# python write_prompt # Independent steps (w/o any intent conditioning)
# Python write_prompt --type_condition codition # Conditioned based on "potential intent types"
import json
import os
import time
import argparse
import re
from collections import Counter
import numpy as np
import copy
import pandas as pd



structure_map={'amr': 'Abstract Meaning Representation (AMR) Graph in the textual Neo-Davidson format', 'dp': 'Dependency Parsing Graph', 'cp': 'Constituency Parsing Graph'}


def get_intent_slot_vob(dataset):
    if dataset == "MTOP":
        intent_vocab, slot_vocab = [], []
        intent_file = './nlu_data/mtop_flat_simple/intent_vocab.txt'
        slot_file = './nlu_data/mtop_flat_simple/slot_vocab.txt'

        for line in open(intent_file, 'r'):
            intent_vocab.append(line.strip())

        for line in open(slot_file, 'r'):
            slot_vocab.append(line.strip())



    elif dataset == "MASSIVE":
        # ----- MASSIVE -----#

        intent_vocab, slot_vocab = [], []
        intent_file = './nlu_data/massive_data_full/intent_vocab.txt'
        slot_file = './nlu_data/massive_data_full/slot_vocab.txt'

        for line in open(intent_file, 'r'):
            intent_vocab.append(line.strip())

        for line in open(slot_file, 'r'):
            slot_vocab.append(line.strip())

    return intent_vocab, slot_vocab


def call_openai(args, que_promp, output_num, temperature):
    success = False
    while success == False:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": que_promp}
                ],
                n = output_num,
                temperature = temperature,
            )
            success = True
        except:
            time.sleep(1)
    #print ("Response", response)
    if (output_num == 1):
        return response['choices'][0]['message']['content']
    else: #TODO: Multiple outputs for future consistency/ majority voting performance

        predictions = get_generated_list(response['choices'])
        if (args.voting_method == 'major'):
            output = find_majority(predictions)
        elif ('complex' in args.voting_method):
            output = find_most_complex(predictions)
        
        return output
        #return get_generated_list(response['choices'])

def get_generated_list(response_list):
    que_list = []
    for que in response_list:
        que_list.append(que['message']['content'])
    return que_list


def find_majority(inputs):
    counter = Counter(inputs)
    majority = counter.most_common(1)
    return majority[0][0]

def find_most_complex(inputs):
    #print ("All inputs", inputs)
    lengths = [len(x) for x in inputs]
    np_len = np.array(lengths)
    idx_np = np.argsort(np_len)[::-1][0]
    longest_ans = np.array(inputs)[idx_np]
    #print ("INs", longest_ans)
    return longes_ans
    


def parse_lf(lf):
    #print ("LF", lf)
    intent_slot = re.sub(r'[\[\]]',' ',lf).strip()
    #print ("Type check", type(intent_slot))
    #print ("Check item", repr(intent_slot))
    intent_slot = re.sub('\s\s+', ' ', intent_slot)
    intent_slot = intent_slot.split(' ')
    slots = []
    slot_pairs=[]
    intent=''
    slot_val=''
    slot_vals=[]
    for item in intent_slot:
        item = item.replace('[','')
        item = item.replace(']','')
        if (item.startswith('IN:')):
            intent = item.split(':')[-1].lower()
        elif (item.startswith('SL:')): # slot type
            if (len(slot_pairs) >= 1):
                slot_pairs.append(slot_val.strip())
                slot_vals.append(slot_val)
                slots.append(tuple(slot_pairs))  
                slot_pairs= []
            slot_pairs.append(item.split(':')[-1].lower())

            #reset
            slot_val = ''
        else:
            slot_val+= ' ' + item

    # Final slots
    slot_pairs.append(slot_val.strip())
    slot_vals.append(slot_val.strip())
    slots.append(tuple(slot_pairs))
    #print ("Extracted Intent: %s \t Slots: %s \n"%(intent, slots))
    return intent, slots, slot_vals

def select_fs_ex(demo_dict_ex, num_sample, criteria):
    if (criteria == 'length'):

        sorted_idx = copy.deepcopy(np.array(demo_dict_ex['utt_length']))
        sorted_idx = np.argsort(sorted_idx)
        top_num= sorted_idx[::-1][:num_sample] #idx of max_elements

        selected_demo_dict_ex={'utt':[], 'intent':[], 'key_phrase':[], 'pair':[], 'AMR':[], 'lf':[]}

        for k,v in demo_dict_ex.items():
            v = copy.deepcopy(np.array(v))
            selected_v = v[top_num]
            selected_demo_dict_ex[k] = list(selected_v)
    elif (criteria == 'random'):
        sorted_idx = np.arange(len(demo_dict_ex['utt_length']))
        np.random.shuffle(sorted_idx)
        top_num = sorted_idx[:num_sample]

        selected_demo_dict_ex={'utt':[], 'intent':[], 'key_phrase':[], 'pair':[], 'AMR':[], 'lf':[]}

        for k,v in demo_dict_ex.items():
            v = copy.deepcopy(np.array(v))
            selected_v = v[top_num]
            selected_demo_dict_ex[k] = list(selected_v)
    elif (criteria == 'type'):
        demos = pd.DataFrame(demo_dict_ex)
        type_sample_num = int(num_sample / demos.groupby('domain').ngroups)
        #print ("Group by", demos.groupby('domain').ngroups)
        #print ("Type check", type_sample_num)
        type_random_demos = demos.groupby('domain').apply(lambda x: x.sample(type_sample_num))
        type_random_demos.reset_index(drop=True, inplace=True)
        selected_demo_dict_ex = type_random_demos.to_dict(orient='list')
        #print ("Top num", top_num)
    return selected_demo_dict_ex
# ----- Output file definition ------#
input_test_file = "nlu_data/mtop_flat_simple/en/eval.txt"
with open(input_test_file, 'r') as file:
    content = file.read()


    
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="MTOP", choices=["MTOP", "MASSIVE"], type=str,
                    help='Type of dataset')
parser.add_argument("--type_condition", default='control', type=str, help='Kind of conditioning: (none, control, control_filter)')
parser.add_argument("--add_demo", choices=['true','false'], default='false', type=str)
parser.add_argument("--output_for", choices=['api','test'], default='test', type=str)

parser.add_argument("--voting_method", default='major', type=str)
parser.add_argument("--structure_rep",  choices=['amr','dp','cp'], default='amr', type=str)
parser.add_argument("--number_output",  default=2, type=int)
parser.add_argument("--number_demo",  default=11, type=int)
parser.add_argument('--demo_select_criteria', default='random', type=str)
parser.add_argument("--temperature",  default=1, type=float)

args = parser.parse_args()

#------ Prepare Demos -----#
#input_fs_file = "nlu_data/mtop_flat_simple/en/fs.txt"
input_fs_file = "nlu_data/mtop_flat_simple/en/demo_100_label.txt"

demo_ex=[]
demo_dict_ex={'utt':[], 'intent':[], 'key_phrase':[], 'pair':[], 'AMR':[], 'lf':[], 'utt_length':[], 'domain':[]}
ex_counter=0
for line in open(input_fs_file, 'r'):
    ex = line.strip().split('\t')
    utt = ex[0]
    lf = ex[1]
    domain=ex[3]
    amr_info = ex[-1] #when generated
    #print ("AMR INFO", amr_info)
    intent, slot_pairs, slot_vals = parse_lf(lf)
    demo_dict={'utt':'', 'intent':'', 'key_phrase':'', 'pair':'', 'AMR':'', 'lf':''}
    demo_dict['utt'] = utt
    demo_dict['intent'] = intent
    demo_dict['pair'] = ','.join(map(str,slot_pairs))
    demo_dict['key_phrase'] = ','.join(slot_vals)
    demo_dict['AMR'] = amr_info
    demo_dict['lf'] = lf
    

    demo_dict_ex['utt'].append(utt)
    demo_dict_ex['intent'].append(intent)
    demo_dict_ex['key_phrase'].append(','.join(slot_vals))
    demo_dict_ex['pair'].append(','.join(map(str, slot_pairs)))
    demo_dict_ex['AMR'].append(amr_info)
    demo_dict_ex['lf'].append(lf)

    demo_dict_ex['domain'].append(domain)
    demo_dict_ex['utt_length'].append(len(utt.split(' ')))


    #print ("Demo dict", demo_dict)
    demo_ex.append(demo_dict)
    ex_counter += 1
    #break

#---- Conduct selection
selected_demo_dict_ex=select_fs_ex(demo_dict_ex, args.number_demo, args.demo_select_criteria)
#print ("Top num", top_num)

print ("Selected key-len %d \t Num samples:%d"%(len(selected_demo_dict_ex), len(selected_demo_dict_ex['utt'])))
print ("Ex counter", ex_counter)

#
## OpenAPI api 
if (args.output_for == 'api'):
    import openai
    key_file = open('./key.txt', 'r')
    key = [k.strip() for k in key_file][0]
    openai.api_key =  str(key)
    model_name = "gpt-3.5-turbo"

# ---- Generic Structure
intent_vocab, slot_vocab = get_intent_slot_vob(args.dataset)
intent_str = ','.join(intent_vocab)
slot_str = ','.join(slot_vocab)
gen_step_1 = 'Given the intent vocabulary and sentence, choose 1 of the following as the intent type for the sentence: \n'
#gen_step_1c = 'Given the intent vocabulary and sentence, choose the top 3 of the following as the potential intent types for the sentence with numeric confidence scores. Return the list of intent types separated by commas, followed by the list of numeric confidence scores separated by commas: \n'

gen_step_1c = 'Given the intent vocabulary and sentence, choose the top 3 of the following as the potential intent types for the sentence. Return the list of intent types separated by commas: \n' 

gen_step_1c_filter = 'Given the intent vocabulary and sentence, choose at least 1 of the following whose confidence score is greater than or equal to 0.8 as the potential intent types for the sentence. Return the list of intent types separated by commas: \n' 

# Update different Structured Rep 
gen_step_1b = 'Given the sentence, generate one and only one ' + structure_map[args.structure_rep] + '\n'

gen_step_1bc = 'Given the sentence and its potential intent types, generate one and only one ' + structure_map[args.structure_rep] + '\n'

gen_step_2 = 'Based on the sentence and its ' + structure_map[args.structure_rep] + ' , identify a list of key phrases for the sentence. Key phrases can be made up from multiple AMR concepts. Each word in key phrases must exist in the given sentence. Return a list of key phrases separated by commas \n'

gen_step_2c = 'Based on the sentence, its potential intents and its ' + structure_map[args.structure_rep] + ' , identify a list of key phrases for the sentence. Key phrases can be made up from multiple AMR concepts. Each word in key phrases must exist in the given sentence. Return a list of key phrases separated by commas \n'

gen_step_3 = 'Given the slot vocabulary, the sentence and its key phrases, identify the corresponding slot type for each key phrases. Return the list of key phrases and their corresponding slot types in the following format: (slot_type, key_phrase) separated by commas \n'
gen_step_3 += 'Slot Vocabulary: ' + slot_str + '\n'

gen_step_3c = 'Given the slot vocabulary, the sentence, its potential intents, its ' + structure_map[args.structure_rep] + ' and its key phrases, identify the corresponding slot type for each key phrases. Return the list of key phrases and their corresponding slot types in the following format: (slot_type, key_phrase) separated by commas \n'
gen_step_3c += 'Slot Vocabulary: ' + slot_str + '\n'

gen_step_4 = 'Given the sentence, its potential intent types, its slot type and slot value pairs in (slot_type, slot_value) format, generate the logic form in the format: [IN:___ [SL:___] [SL:____]] where IN: is followed by an intent type and SL: is followed by a slot type and slot value pair separated by white space. The number of [SL: ] is unlimited. The number of [IN: ] is limited to 1 \n'

direct_prompt = "New Session: Given the intent type vocabulary, slot type vocabulary and sentence, generate logic form of the sentence in the format of [IN:___ [SL:____] [SL:___]] where IN: is followed by an intent type and SL: is followed by a slot type and slot value pair separated by white space. The number of [SL: ] is unlimited. \n"
direct_prompt += "Intent Type: " + intent_str + "\n"
direct_prompt += "Slot Type: " + slot_str + "\n"

type_prompt = "chain_of"
add_demo, condition_type = args.add_demo, args.type_condition
add_voting = args.voting_method
result_output_file = f'./result/{type_prompt}_demo_{add_demo}_condition_{condition_type}_voting_{add_voting}_dialogue.jsonl'
writer = open(result_output_file, 'w')
slot_type=''
intent=''
amr_graph=''
key_phrases=''

content = content.split("\n")
for example in content[0:100]:
    utterance, logical_form, _, _, tag = example.split("\t")
    # --- Directly prompt
    if type_prompt == "direct":
        direct_prompt += "Sentence: " + utterance + "\n"
        direct_prompt += "Just generate the Logic Form: "
        print ("Direct prompt", direct_prompt)
        pred_lf = call_openai(args,direct_prompt, args.number_output, args.temperature)
        writer.write(json.dumps({"utterance": utterance, "pred_lf": pred_lf, "gold_lf": logical_form}) + '\n')
    else:
        # --- Step 1a: Get Intent
        if (args.type_condition == 'none'):
            step_1a_prompt = gen_step_1 + 'Intent Vocabulary: ' + intent_str + '\n'

        elif (args.type_condition == 'control_single'):
            step_1a_prompt = gen_step_1 + 'Intent Vocabulary: ' + intent_str + '\n'
        elif (args.type_condition == 'control'):
            step_1a_prompt = gen_step_1c + 'Intent Vocabulary: ' + intent_str + '\n'
        elif (args.type_condition == 'control_filter'):
            step_1a_prompt = gen_step_1c_filter + 'Intent Vocabulary: ' + intent_str + '\n'
        step_1a_prompt += 'Sentence: ' + utterance + '\n'
        step_1a_prompt += 'Intent type: ' + '\n'

        # --- Demo Step 1a
        if (args.add_demo == 'true'):
            if (args.type_condition == 'none'):
                demo_1 = gen_step_1 + 'Intent Vocabulary: ' + intent_str + '\n'

            elif (args.type_condition == 'control_single'):
                demo_1 = gen_step_1 + 'Intent Vocabulary: ' + intent_str + '\n'
            elif (args.type_condition == 'control'):
                demo_1 = gen_step_1c + 'Intent Vocabulary: ' + intent_str + '\n'
            elif (args.type_condition == 'control_filter'):
                demo_1 = gen_step_1c_filter + 'Intent Vocabulary: ' + intent_str + '\n'

            iter_demo=0
            for idx in range (len(selected_demo_dict_ex['utt'])): 
            #for dem in demo_ex:
                demo_utt = selected_demo_dict_ex['utt'][idx]
                demo_intent = selected_demo_dict_ex['intent'][idx]
                #print ("Demo ex", utt)

                demo_1 += 'Sentence: ' + demo_utt + '\n'
                demo_1 += 'Intent type: ' + demo_intent + '\n'
                iter_demo += 1
            assert iter_demo == args.number_demo
            step_1a_prompt = demo_1 + '\n' + step_1a_prompt
        
        if (args.output_for == 'api'):
            intent = call_openai(args,step_1a_prompt, args.number_output, args.temperature)
            print("STEP 1a: Get Intent")
        else:
            print("STEP 1a: Get Intent \n", step_1a_prompt)
            intent =''
        
        # --- Step 1b: Get AMR Graph
        if (args.type_condition == 'none'):
            step_1b_prompt = gen_step_1b + 'Sentence: ' + utterance + '\n'
        else:
            step_1b_prompt = gen_step_1bc + 'Sentence: ' + utterance + '\n'
            step_1b_prompt += 'Potential Intent Types: ' + intent + '\n'
        
        step_1b_prompt += structure_map[args.structure_rep] + ': ' + '\n'
        print ("Step 1b prompt", step_1b_prompt)

        #--- Demo 1b
        if (args.add_demo == 'true'):
            iter_demo = 0
            for idx in range (len(selected_demo_dict_ex['utt'])): 
                demo_utt = selected_demo_dict_ex['utt'][idx]
                demo_intent = selected_demo_dict_ex['intent'][idx]
                demo_amr = selected_demo_dict_ex['AMR'][idx]
                if (args.type_condition == 'none'):
                    demo_1b = gen_step_1b + 'Sentence: ' + demo_utt + '\n'
                else:
                    demo_1b = gen_step_1bc + 'Sentence: ' + demo_utt + '\n'
                if (args.type_condition != 'none'):
                    demo_1b += 'Potential Intent Types: ' + demo_intent + '\n'
                
                demo_1b += structure_map[args.structure_rep] + ': ' + demo_amr + '\n'
                iter_demo += 1

            assert iter_demo == args.number_demo

            step_1b_prompt  = demo_1b + '\n' + step_1b_prompt 

        if (args.output_for == 'api'):
            amr_graph = call_openai(args,step_1b_prompt, args.number_output, args.temperature)
            
            print("STEP 1b: Get AMR Graph")
            print ("AMR", amr_graph)
        else:
            print("STEP 1b: Get AMR Graph", step_1b_prompt)
            amr_graph=''

        # --- Step 2: Get Key Phrases
        if (args.type_condition == 'none'):
            step_2_prompt = gen_step_2 + 'Sentence: ' + utterance + '\n'
        else:
            step_2_prompt = gen_step_2c + 'Sentence: ' + utterance + '\n'
            step_2_prompt += 'Potential Intents: ' + intent + '\n'

        #print ("AMR Graph", type(structure_map[args.structure_rep]), type(amr_graph), amr_graph)
        step_2_prompt += structure_map[args.structure_rep] + ': ' + amr_graph + '\n'
        step_2_prompt += 'Key phrases: \n'

        # -- Step 2 Demo
        if (args.add_demo == 'true'):
            iter_demo = 0
            for idx in range (len(selected_demo_dict_ex['utt'])): 
                demo_utt = selected_demo_dict_ex['utt'][idx]
                demo_intent = selected_demo_dict_ex['intent'][idx]
                demo_kp = selected_demo_dict_ex['key_phrase'][idx]
                demo_amr = selected_demo_dict_ex['AMR'][idx]

                if (args.type_condition == 'none'):
                    demo_2 = gen_step_2 + 'Sentence: ' + demo_utt + '\n'
                else:
                    demo_2 = gen_step_2c + 'Sentence: ' + demo_utt + '\n'
            
                #for dem in demo_ex:

                if (args.type_condition != 'none'):
                    demo_2 += 'Potential Intents: ' +  demo_intent + '\n'
                demo_2 += structure_map[args.structure_rep] + ': ' + demo_amr + '\n'
                demo_2 += 'Key phrases: ' + demo_kp + '\n'
                iter_demo += 1

            assert iter_demo == args.number_demo

            step_2_prompt  = demo_2 + '\n' + step_2_prompt 

        if (args.output_for == 'api'):
            key_phrases = call_openai(args,step_2_prompt, args.number_output, args.temperature)
            print("STEP 2: Get Key Phrases")
        else:
            print("STEP 2: Get Key Phrases", step_2_prompt)
            key_phrases=''

        # --- Step 3: Get Slot Type
        if (args.type_condition == 'none'):
            step_3_prompt = gen_step_3 + 'Sentence: ' + utterance + '\n'
            step_3_prompt += 'Key phrases: ' + key_phrases + '\n'
            #step_3 = gen_step_3 + 'Key phrases: ' + key_phrases + '\n'
            #step_3 += 'Sentence: ' + utterance + '\n'
        else:
            step_3_prompt = gen_step_3c + 'Sentence: ' + utterance + '\n'
            step_3_prompt += 'Potential Intent Types: ' + intent + '\n'
            step_3_prompt += structure_map[args.structure_rep] + ': ' + amr_graph + '\n'
            step_3_prompt += 'Key phrases: ' + key_phrases + '\n'
            #step_3 = gen_step_3 + 'Key phrases: ' + '\n'
            #step_3 += 'Sentence: ' + utterance + '\n'

        step_3_prompt += 'Slot Type, Key phrase  pairs: \n'


        if (args.add_demo == 'true'):
            if (args.type_condition == 'none'):
                demo_3 = gen_step_3 + 'Sentence: ' + demo_utt + '\n'
            else:
                demo_3 = gen_step_3c + 'Sentence: ' + demo_utt + '\n'
       
            iter_demo = 0
            for idx in range (len(selected_demo_dict_ex['utt'])): 
            #for dem in demo_ex:
                demo_utt = selected_demo_dict_ex['utt'][idx]
                demo_intent = selected_demo_dict_ex['intent'][idx]
                demo_kp = selected_demo_dict_ex['key_phrase'][idx]
                demo_pair = selected_demo_dict_ex['pair'][idx]
                if (args.type_condition != 'none'):
                    demo_3 += 'Potential Intents: ' + demo_intent + '\n'
                demo_3 += structure_map[args.structure_rep] + ': ' + demo_amr + '\n'
                demo_3 += 'Key phrases: ' + demo_kp + '\n'

                demo_3 += 'Slot Type, Key phrase pairs: ' + demo_pair + '\n'
                iter_demo += 1

            assert iter_demo == args.number_demo
            step_3_prompt  = demo_3 + '\n' + step_3_prompt 

        if (args.output_for == 'api'):
            slot_type = call_openai(args,step_3_prompt, args.number_output, args.temperature)
            print("STEP 3: Get Slot Type")
        else:
            slot_type=''
            print("STEP 3: Get Slot Type", step_3_prompt)
        

        # --- Step 4: Get Logic Form
        step_4_prompt = gen_step_4 + 'Sentence: ' + utterance + '\n'
        step_4_prompt += "Intent: " + intent + '\n'
        step_4_prompt += "Slot Type, Slot Value pairs: " + slot_type + '\n'
        step_4_prompt += 'Logic Form: \n'


        #--- Add Demo Step 4
        if (args.add_demo == 'true'):
            demo_4 = gen_step_4 + 'Sentence: ' + demo_utt + '\n'
        
            iter_demo=0
            for idx in range (len(selected_demo_dict_ex['utt'])): 
            #for dem in demo_ex:
                demo_utt = selected_demo_dict_ex['utt'][idx]
                demo_intent = selected_demo_dict_ex['intent'][idx]
                demo_kp = selected_demo_dict_ex['key_phrase'][idx]
                demo_pair = selected_demo_dict_ex['pair'][idx]
                demo_lf = selected_demo_dict_ex['lf'][idx]
                demo_4 += "Intent: " + demo_intent + '\n'
                demo_4 += "Slot Type, Slot Value pairs: " + demo_pair + '\n'
                demo_4 += 'Logic Form: ' + demo_lf + '\n'
                iter_demo += 1

            assert iter_demo == args.number_demo

            step_4_prompt  = demo_4 + '\n' + step_4_prompt 
        if (args.output_for == 'api'):
            pred_lf = call_openai(args,step_4_prompt, args.number_output, args.temperature)
            print("STEP 4: Get Logic Form")
        else:
            pred_lf =''
            print("STEP 4: Get Logic Form", step_4_prompt)
        writer.write(json.dumps({"utterance": utterance, "intent": intent, "AMR Graph": amr_graph, "key_phrase":
            key_phrases, "slot_type": slot_type, "pred_lf": pred_lf, "gold_lf": logical_form}) + '\n')
