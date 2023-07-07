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

structure_map={'amr': 'Abstract Meaning Representation (AMR) Graph in the textual Neo-Davidson format', 'dp': 'Dependency Parsing Graph', 'cp': 'Constituency Parsing Graph', 'none':'none'}


def get_intent_slot_vob(dataset):
    if dataset == "MTOP":
        intent_vocab, slot_vocab = [], []
        intent_file = './nlu_data/mtop_flat_simple/intent_vocab.txt'
        slot_file = './nlu_data/mtop_flat_simple/slot_vocab.txt'

        for line in open(intent_file, 'r'):
            intent_vocab.append(line.strip())

        for line in open(slot_file, 'r'):
            slot_vocab.append(line.strip())

        
        intent_map_file = './nlu_data/mtop_flat_simple/intent_vocab_map.jsonl'
        intent_vocab, intent_descr=[],[]
        intent_map={}
        intent_rev_map={}
        for line in open(intent_map_file, 'r'):
            line_output = json.loads(line)
            name, description = line_output['label'].strip(), line_output['label_description'].strip()
            intent_map[name] = description
            intent_vocab.append(name)
            intent_descr.append(description)
            intent_rev_map[description] = name

        

        slot_map={}
        slot_map_file = './nlu_data/mtop_flat_simple/slot_vocab_map.jsonl'

        slot_vocab, slot_descr=[],[]
        slot_rev_map={}
        for line in open(slot_map_file, 'r'):
            line_output = json.loads(line)
            name, description = line_output['label'].strip(), line_output['label_description'].strip()
            slot_map[name] = description
            slot_vocab.append(name)
            slot_descr.append(description)
            slot_rev_map[description] = name

        #print ("Check intent", intent_vocab)
        #print ("Intent map", intent_map)

    elif dataset == "MASSIVE":
        # ----- MASSIVE -----#

        intent_vocab, slot_vocab = [], []
        intent_file = './nlu_data/massive_data_full/intent_vocab.txt'
        slot_file = './nlu_data/massive_data_full/slot_vocab.txt'

        for line in open(intent_file, 'r'):
            intent_vocab.append(line.strip())

        for line in open(slot_file, 'r'):
            slot_vocab.append(line.strip())


        intent_map_file = './nlu_data/massive_data_full/intent_vocab_map.jsonl'
        intent_vocab, intent_descr=[],[]
        intent_map={}
        intent_rev_map={}
        for line in open(intent_map_file, 'r'):
            line_output = json.loads(line)
            name, description = line_output['label'], line_output['label_description']
            intent_map[name] = description
            intent_vocab.append(name)
            intent_descr.append(description)
            intent_rev_map[description] = name
        

        slot_map={}
        slot_map_file = './nlu_data/massive_data_full/slot_vocab_map.jsonl'

        slot_vocab, slot_descr=[],[]
        slot_rev_map={}
        for line in open(slot_map_file, 'r'):
            line_output = json.loads(line)
            name, description = line_output['label'], line_output['label_description']
            #name = name.replace('_',' ')
            slot_map[name] = description
            slot_vocab.append(name)
            slot_descr.append(description)
            slot_rev_map[description] = name

    return intent_vocab, intent_descr, intent_map, intent_rev_map, slot_vocab, slot_descr, slot_map, slot_rev_map


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
    idx_np = np.argsort(np_len)[::-1][:5]
    longest_ans = np.array(inputs)[idx_np]
    output = find_majority(longest_ans)
    #print ("INs", longest_ans)
    return output
    


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

def condition_intent_info(args, current_prompt,intent_info, intent_map):
    output_prompt = current_prompt
    if (args.condition_on == 'descr'):
        multiple_intents = intent_info.split(',')
        #print ("Multiple intents", multiple_intents)
        for m in multiple_intents:
            #if (m in intent_map):
            #    output_prompt += 'Potential Intent Types: ' + intent_map[m]
            #else:
            if (True):
                output_prompt += 'Potential Intent Types: ' + m
        output_prompt += '\n'
    else:
        output_prompt += 'Potential Intent Types: ' + intent_info + '\n'
    return output_prompt


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="MTOP", choices=["MTOP", "MASSIVE"], type=str,
                    help='Type of dataset')

parser.add_argument("--demo_file", default='demo_5_label', type=str, help='Kind of conditioning: (none, control, control_filter)')

parser.add_argument("--type_prompt", default='chain_of', type=str, help='Direct Prompting or CoT')
parser.add_argument('--seed', default=111, type=int, choices=[111,222,333])

parser.add_argument("--test_file", default='test', type=str, help='Kind of conditioning: (none, control, control_filter)')
parser.add_argument("--type_condition", default='none', type=str, help='Kind of conditioning: (none, control, control_filter)')
parser.add_argument("--add_demo", choices=['true','false'], default='false', type=str)
parser.add_argument("--output_for", choices=['api','test'], default='test', type=str)

parser.add_argument("--voting_method", default='major', type=str)
parser.add_argument("--structure_rep",  choices=['amr','dp','cp','none'], default='amr', type=str)
parser.add_argument("--number_output",  default=1, type=int)
parser.add_argument("--number_demo",  default=11, type=int)
parser.add_argument('--demo_select_criteria', default='random', type=str)
parser.add_argument("--temperature",  default=0.7, type=float)
parser.add_argument('--condition_on', default='label', type=str)

parser.add_argument('--add_domain', default='false', type=str)

parser.add_argument('--write_output', default='false', type=str)

args = parser.parse_args()

#------ Prepare Demos -----#
#input_fs_file = "nlu_data/mtop_flat_simple/en/fs.txt"
if (args.dataset == 'MTOP'):
    data_root = './nlu_data/mtop_flat_simple/en/'
else:
    data_root = './nlu_data/massive_data_full/en/'

#input_fs_file = "nlu_data/mtop_flat_simple/en/demo_100_label.txt"
input_fs_file = os.path.join(data_root, args.demo_file + '.txt')

# ----- Output file definition ------#
#input_test_file = "nlu_data/mtop_flat_simple/en/eval.txt"
input_test_file = os.path.join(data_root, args.test_file + '_'  + str (args.seed) + '.txt')
print ("Input test", input_test_file)
#with open(input_test_file, 'r') as file:
#    content = file.read()
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
if (args.add_demo == 'true'):
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
intent_vocab, intent_descr, intent_map, intent_rev_map, slot_vocab,slot_map, slot_descr,slot_rev_map = get_intent_slot_vob(args.dataset)
intent_str = ','.join(intent_vocab)
intent_descr_str=','.join(intent_descr)
slot_str = ','.join(slot_vocab)


#--- Instruction Prompt
# TODO: add intent description based on args.condition_on
#gen_step_1 = 'Given the intent vocabulary and corresponding description in the format (intent vocabulary label, itent description) and the input sentence, choose 1 of the following as the intent type for the sentence: \n'
gen_step_1 = 'Given the intent vocabulary ' 
gen_step_1c = 'Given the intent vocabulary '
gen_step_1c_filter = 'Given the intent vocabulary '

if (args.condition_on == 'descr'):
    #gen_step_1 += ', additional intent explanations in the paired format as (intent type, intent_explanations), '
    #gen_step_1c += ', additional intent explanations, '
    #gen_step_1c_filter += ', intent description, '
    pass

if (args.add_domain == 'true'):
    gen_step_1 += ' domain name, '
    gen_step_1c += ' domain name, '
    gen_step_1c_filter += ' domain name, '

gen_step_1 += 'and the input sentence, choose 1 of the following intent types as the intent type for the sentence. Return the exact match in the list. \n'
gen_step_1c += ' and the input sentence, choose the top 3 of the following intent types as the potential intent types for the sentence. Return the list of intent types separated by commas: \n' 
gen_step_1c_filter += 'and input sentence, choose all of the intent types whose confidence scores are greater than or equal to 0.65 as the potential intent types for the sentence. Return the list of intent types separated by commas: \n' 

if (args.condition_on == 'descr'):
    pass
    #gen_step_1 += ' Do not return the intent explanations.'
    #gen_step_1c += ' Do not return the intent explanations.'
    #gen_step_1c_filter += ' Do not return the intent explanations.'

gen_step_1 += '\n'
gen_step_1c += '\n'
gen_step_1c_filter += '\n'
#gen_step_1 += 'Intent Vocabulary: ' + intent_str + '\n'
#gen_step_1c += 'Intent Vocabulary: ' + intent_str + '\n'
#gen_step_1c_filter += 'Intent Vocabulary: ' + intent_str + '\n'

if (args.condition_on == 'descr'):
    #gen_step_1 += 'Intent Description: ' + intent_descr_str + '\n'
    gen_step_1 += 'Intent Vocabulary: \n'
    gen_step_1c += 'Intent Vocabulary: \n'
    gen_step_1c_filter += 'Intent Vocabulary: \n'

    #gen_step_1 += intent_descr_str + '\n'
    #gen_step_1c += intent_descr_str + '\n'
    #gen_step_1c_filter += intent_descr_str + '\n'
    for idx in range (len(intent_vocab)):
        gen_step_1 += '(' + intent_vocab[idx] + ',' + intent_descr[idx] + '),'
        gen_step_1c += '(' + intent_vocab[idx] + ',' + intent_descr[idx] + '),'
        gen_step_1c_filter += '(' + intent_vocab[idx] + ',' + intent_descr[idx] + '),'
    gen_step_1c += 'Intent Description: ' + intent_descr_str + '\n'
    gen_step_1c_filter += 'Intent Description: ' + intent_descr_str + '\n'

else:
    gen_step_1 += 'Intent Vocabulary: ' + intent_str + '\n'

    gen_step_1c += 'Intent Vocabulary: ' + intent_str + '\n'

    gen_step_1c_filter += 'Intent Vocabulary: ' + intent_str + '\n'



gen_step_1_flip = 'Given the intent vocabulary, sentence and its AMR graph, choose 1 of the following as the intent type for the sentence: \n'
gen_step_1_flip += 'Intent Vocabulary: ' + intent_str + '\n'
#gen_step_1c = 'Given the intent vocabulary and sentence, choose the top 3 of the following as the potential intent types for the sentence with numeric confidence scores. Return the list of intent types separated by commas, followed by the list of numeric confidence scores separated by commas: \n'

#gen_step_1c = 'Given the intent vocabulary, intent description and sentence, choose the top 3 of the following as the potential intent types for the sentence. Return the list of intent types separated by commas: \n' 
#gen_step_1c = 'Given the intent vocabulary and the input sentence, choose the top 3 of the following as the potential intent types for the sentence. Return the list of intent types separated by commas: \n' 
#gen_step_1c += 'Intent Description: ' + intent_descr_str + '\n'
#gen_step_1c += 'Intent Vocabulary: ' + intent_str + '\n'

#gen_step_1c_filter = 'Given the intent vocabulary and sentence, choose at least 1 of the following whose confidence score is greater than or equal to 0.8 as the potential intent types for the sentence. Return the list of intent types separated by commas: \n' 

#gen_step_1c_filter += 'Intent Vocabulary: ' + intent_str + '\n'

# Update different Structured Rep 
gen_step_1b = 'Given the sentence, generate one and only one ' + structure_map[args.structure_rep]
if (args.structure_rep == 'amr'):
    gen_step_1b += '. The format includes :ARG and :op. Eacch word in the leaf node must exist in the given sentence '
gen_step_1b += '\n'

gen_step_1bc = 'Given the sentence and its potential intent types, generate one and only one ' + structure_map[args.structure_rep]

if (args.structure_rep == 'amr'):
    gen_step_1bc += '. The format includes :ARG and :op. Each word in the leaf node must exist in the given sentence '

gen_step_1bc += '. No explanation is needed. \n'

if (args.structure_rep == 'amr'):
    gen_step_1b += '. The format includes :ARG and :op. Eacch word in the leaf node must exist in the given sentence \n'

#gen_step_2 = 'Based on the sentence and its ' + structure_map[args.structure_rep] + ' , identify a list of key phrases for the sentence. Key phrases can be made up from multiple AMR concepts. Each word in key phrases must exist in the given sentence. Each unique word can only appear in one key phrase. Key phrases need to contain consecutive words in the given sentence. Return a list of key phrases separated by commas \n'


gen_step_2 = 'Based on the following information'

gen_step_2_nostruct = 'Based on the sentence, identify a list of key phrases for the sentence. Each word in key phrases must exist in the given sentence and might only appear once in the returned list. Key phrases need to contain consecutive words in the given sentence. Return a list of key phrases separated by commas \n'

gen_step_2c = 'Based on the sentence, its potential intents and its ' + structure_map[args.structure_rep]

if (args.add_domain == 'true'):
    gen_step_2 += ',its domain name'
    gen_step_2c += ',its domain name'

gen_step_2 += ', identify a list of key '

gen_step_2c += ', identify a list of key '
if (args.structure_rep == 'amr'):

    gen_step_2+= 'noun '
    gen_step_2c += 'noun '


gen_step_2 += 'phrases for the sentence. Key phrases can be made up from multiple concepts. Each word in key phrases must exist in the given sentence. Each word in the sentence appears in only one key phrase. Key phrases need to contain consecutive words in the given sentence. Key phrases do not need to cover all words in the sentence. Return a list of key phrases separated by commas \n'

gen_step_2c += 'phrases for the sentence. Key phrases can be made up from multiple AMR concepts. Each word in key phrases must exist in the given sentence. Each word in the sentence must appear in at most one and only one key phrase. Key phrases need to contain consecutive words in the given sentence. Key phrases do not need to cover all words in the sentence. Return a list of key phrases separated by commas \n'


gen_step_3 = 'Given the slot vocabulary'

gen_step_3c = 'Given the slot vocabulary'

if (args.condition_on == 'descr'):
    gen_step_3 += ', slot explanations in the paired format (slot type, slot explanations)'
    gen_step_3c += ', slot explanations in the paired format (slot type, slot explanations)'


gen_step_3 +=  ', the sentence, its key phrases'
gen_step_3c+= ', the sentence, its potential intents, its ' + structure_map[args.structure_rep] +', its key phrases'
if (args.add_domain == 'true'):
    gen_step_3 += ', and domain name'
    gen_step_3c += ', and domain name'

gen_step_3 +=', identify the corresponding slot type as one of the types in the slot vocabulary for each key phrase. Return the list of key phrases and their corresponding slot types in the following format: (slot_type, key_phrase) separated by commas. If none of the slot types in the vocabulary fits, return the slot type as O' 
gen_step_3c += ', identify one of the following in the slot vocabulary as the slot type for each key phrase. Return the list of key phrases and their corresponding slot types in the following format: (slot_type, key_phrase) separated by commas. If none of the slot types in the vocabulary fits, return the slot type as O.'
if (args.condition_on == 'descr'):
    gen_step_3c += 'Do not return slot explanations. '
    gen_step_3 += ' Do not return slot explanations. '

gen_step_3c += '\n'
gen_step_3 += '\n'

#print ("Slot check", slot_vocab, slot_descr)
if (args.condition_on == 'label'):
    gen_step_3 += 'Slot Vocabulary: ' + slot_str + '\n'
    gen_step_3c += 'Slot Vocabulary: ' + slot_str + '\n'
else:
    gen_step_3 += ' Slot Vocabulary: ' + '\n'
    gen_step_3c += ' Slot Vocabulary: ' + '\n'

    for k,v in slot_descr.items():
        gen_step_3 += '(' + k + ',' + v + '),'
        gen_step_3c += '(' + k + ',' + v + '),'
    #for idx in range (len(slot_vocab)):
    #    gen_step_3 += '(' + slot_vocab[idx] + ',' + slot_descr[idx] + '),'
    #    gen_step_3c += '(' + slot_vocab[idx] + ',' + slot_descr[idx] + '),'

gen_step_4 = 'Given the sentence, its potential intent types, its slot type and slot value pairs in (slot_type, slot_value) format,'
if (args.add_domain == 'true'):
    gen_step_4 += ' domain,'

gen_step_4 += 'generate a single logic form in the format: [IN:___ [SL:___] [SL:____]] where IN: is followed by an intent type and SL: is followed by a slot type and slot value pair separated by white space. The number of [SL: ] is unlimited. The number of [IN: ] is limited to 1. Use the given information precisely \n'

new_session="New Session: "
direct_prompt = "Given the intent type vocabulary, slot type vocabulary and sentence, generate logic form of the sentence in the format of [IN:___ [SL:____] [SL:___]] where IN: is followed by an intent type and SL: is followed by a slot type and slot value pair separated by white space. The number of [SL: ] is unlimited. The number of [IN: ] is limited to 1 \n"
direct_prompt += "Intent Type: " + intent_str + "\n"
direct_prompt += "Slot Type: " + slot_str + "\n"

type_prompt,seed_val = args.type_prompt, args.seed
add_demo, condition_type = args.add_demo, args.type_condition
add_voting, numdemo, select,cond = args.voting_method, args.number_demo, args.demo_select_criteria, args.condition_on
struc_rep = args.structure_rep
num_output =args.number_output
add_domain=args.add_domain

result_output_file=f'./result_{args.dataset}/ablation_randorder.jsonl'

#if (args.add_demo == 'false'):
#    result_output_file = f'./result_{args.dataset}/zs_{type_prompt}_seed_{seed_val}_condition_{condition_type}_voting_{add_voting}_structure_{struc_rep}_criteria_{select}_condname_{cond}_numoutput_{num_output}_domain_{add_domain}_dialogue.jsonl'
#else:
#    result_output_file = f'./result_{args.dataset}/fs_{type_prompt}_seed_{seed_val}_condition_{condition_type}_voting_{add_voting}_structure_{struc_rep}_numberdemo_{numdemo}_criteria_{select}_condname_{cond}_numoutput_{num_output}_dialogue.jsonl'

if (args.write_output == 'true'):
    writer = open(result_output_file, 'w')
else:
    writer=None
slot_type=''
intent=''
amr_graph=''
key_phrases=''
sequence_order=['1', '1b', '2', '3', '4']
#sequence_order=['1b', '2', '3', '1a', '4']
sequence_order=['2','1','1b','3','4']

def gen_step_1a(args,intent_str):

    if (args.type_condition == 'none'):
        step_1a_prompt = gen_step_1 + 'Intent Vocabulary: ' + intent_str + '\n'

    elif (args.type_condition == 'control_single'):
        step_1a_prompt = gen_step_1 + 'Intent Vocabulary: ' + intent_str + '\n'
    elif (args.type_condition == 'control'):
        step_1a_prompt = gen_step_1c + 'Intent Vocabulary: ' + intent_str + '\n'
    elif (args.type_condition == 'control_filter'):
        step_1a_prompt = gen_step_1c_filter + 'Intent Vocabulary: ' + intent_str + '\n'
    #if (demo == 'false'):
    #step_1a_prompt += 'Sentence: ' + utterance + '\n'
    #step_1a_prompt += 'Intent type: ' + '\n'
    #prompt = 
    return step_1a_prompt

def gen_step_prompt(args, step_number='1a'):
    
    if (step_number == '1a'):
        if (args.type_condition == 'control'):
            out_prompt = gen_step_1c
        elif (args.type_condition == 'control_filter'):
            out_prompt = gen_step_1c_filter
        else:
            out_prompt = gen_step_1
    elif (step_number == '1c'):
        out_prompt = gen_step_1_flip
    elif (step_number == '1b'):
        if (args.type_condition == 'none'):
            out_prompt = gen_step_1b 
        else:
            out_prompt = gen_step_1bc

    elif (step_number == '2'):
        if (args.type_condition == 'none'):
            out_prompt = gen_step_2
        else:
            out_prompt = gen_step_2c

    elif (step_number == '3'):
        if (args.type_condition == 'none'):
            out_prompt = gen_step_3
        else:
            out_prompt = gen_step_3c
            step_3_prompt = gen_step_3 + 'Sentence: ' + utterance + '\n'

    elif (step_number == '4'):
        out_prompt = gen_step_4
    out_prompt += '\n'
    return out_prompt

#def gen_full_prompt(args, step_number, utterance,intent, slot):

result = []
for example in open(input_test_file, 'r'):
    #print ("Example", example)
    if (args.dataset =='MTOP'):
        utterance, logical_form, _, domain_name, tag = example.strip().split("\t")
    else:

        utterance, logical_form,domain_name, tag = example.strip().split("\t")
    print ("utt", utterance, logical_form)
    # --- Step 1a: Get Intent
    # --- Demo Step 1a
    #if (args.add_demo == 'true'):
    #    demo_1 = gen_step_prompt(args, step_number='1a')
    #    #demo_1 = gen_step_1a(args, intent_str)
    #    iter_demo=0
    #    for idx in range (len(selected_demo_dict_ex['utt'])): 
    #    #for dem in demo_ex:
    #        demo_utt = selected_demo_dict_ex['utt'][idx]
    #        demo_intent = selected_demo_dict_ex['intent'][idx]

    #        demo_1 += 'Sentence: ' + demo_utt + '\n'
    #        demo_1 += 'Intent type: ' + demo_intent + '\n##\n'
    #        iter_demo += 1
    #    assert iter_demo == args.number_demo
    #    step_1a_prompt = demo_1 + '\n'
    #    #step_1a_prompt = demo_1 + '\n' + step_1a_prompt
    #else:


    # Fine to Coarse Order
    #step_1b_prompt = gen_step_prompt(args, step_number='1b')

    #step_1b_prompt += 'Sentence: ' + utterance + '\n'
    #if (args.type_condition != 'none'):
    #    step_1b_prompt = condition_intent_info(args, step_1b_prompt,intent, intent_map)
    #
    #step_1b_prompt += structure_map[args.structure_rep] + ': ' + '\n'

    #if (args.output_for == 'api'):
    #    amr_graph = call_openai(args,step_1b_prompt, args.number_output, args.temperature)
    #    
    #    print("STEP 1b: Get AMR Graph", step_1b_prompt)
    #    print ("AMR", amr_graph)
    #else:
    #    print("STEP 1b: Get AMR Graph", step_1b_prompt)
    #    amr_graph=''


    #step_2_prompt = gen_step_prompt(args, step_number='2')

    #step_2_prompt += 'Sentence: ' + utterance + '\n'
    #if (args.type_condition != 'none'):
    #    step_2_prompt = condition_intent_info(args, step_2_prompt,intent, intent_map)
    #step_2_prompt += structure_map[args.structure_rep] + ': ' + amr_graph + '\n'


    #step_2_prompt += 'Key phrases: \n'

    #if (args.output_for == 'api'):
    #    key_phrases = call_openai(args,step_2_prompt, args.number_output, args.temperature)
    #    print("STEP 2: Get Key Phrases", step_2_prompt)
    #    print("STEP 2: Key phrases", key_phrases)
    #else:
    #    print("STEP 2: Get Key Phrases", step_2_prompt)
    #    key_phrases=''





    #step_3_prompt = gen_step_prompt(args, step_number='3')
    #step_3_prompt += 'Sentence: ' + utterance + '\n'
    #if (args.type_condition != 'none'):
    #    step_3_prompt = condition_intent_info(args, step_3_prompt,intent, intent_map)
    #    step_3_prompt += structure_map[args.structure_rep] + ': ' + amr_graph + '\n'

    #step_3_prompt += 'Key phrases: ' + key_phrases + '\n'
    #if (args.add_domain == 'true'):
    #    #step_1a_prompt += 'Domain Name: ' + domain_name + '\n'
    #    step_3_prompt += 'Domain name: ' + domain_name + '\n'
    #step_3_prompt += 'Slot Type, Key phrase  pairs: \n'
    #if (args.output_for == 'api'):
    #    slot_type = call_openai(args,step_3_prompt, args.number_output, args.temperature)
    #    print("STEP 3: Get Slot Type", step_3_prompt)
    #    print ("STEP 3: Slot Type: ", slot_type)
    #else:
    #    slot_type=''
    #    print("STEP 3: Get Slot Type", step_3_prompt)



    #step_1a_prompt = gen_step_prompt(args, step_number='1a')

    #step_1a_prompt = new_session + step_1a_prompt

    #step_1a_prompt += 'Sentence: ' + utterance + '\n'
    ##if (args.add_domain == 'true'):
    ##    step_1a_prompt += 'Domain Name: ' + domain_name + '\n'
    #step_1a_prompt += 'Intent type: ' + '\n'
    #if (args.output_for == 'api'):
    #    intent = call_openai(args,step_1a_prompt, args.number_output, args.temperature)
    #    print("STEP 1a: Get Intent ", step_1a_prompt)
    #    print ("Intent", intent)
    #    #act_intent = intent_rev_map[intent.strip()]
    #    #print ("Mapped intent", act_intent)
    #else:
    #    print("STEP 1a: Get Intent \n", step_1a_prompt)
    #    intent =''
   
    #step_4_prompt = gen_step_prompt(args, step_number='4')

    #step_4_prompt += 'Sentence: ' + utterance + '\n'

    #step_4_prompt += "Intent: " + intent + '\n'
    #step_4_prompt += "Slot Type, Slot Value pairs: " + slot_type + '\n'

    #if (args.add_domain == 'true'):
    #    step_4_prompt += 'Domain name: ' + domain_name + '\n'
    #step_4_prompt += 'Logic Form: \n'
    #if (args.output_for == 'api'):
    #    pred_lf = call_openai(args,step_4_prompt, args.number_output, args.temperature)
    #    print("STEP 4: Get Logic Form", step_4_prompt)
    #    print ("STEP 4: OUT LF ", pred_lf)
    #    print ("Target", logical_form)
    #else:
    #    pred_lf =''
    #    print("STEP 4: Get Logic Form", step_4_prompt)

    #-------------------Random Order ------------------------------------------#

    step_2_prompt = gen_step_prompt(args, step_number='2')

    step_2_prompt += 'Sentence: ' + utterance + '\n'
    if (args.type_condition != 'none'):
        step_2_prompt = condition_intent_info(args, step_2_prompt,intent, intent_map)
    step_2_prompt += structure_map[args.structure_rep] + ': ' + amr_graph + '\n'


    step_2_prompt += 'Key phrases: \n'

    if (args.output_for == 'api'):
        key_phrases = call_openai(args,step_2_prompt, args.number_output, args.temperature)
        print("STEP 2: Get Key Phrases", step_2_prompt)
        print("STEP 2: Key phrases", key_phrases)
    else:
        print("STEP 2: Get Key Phrases", step_2_prompt)
        key_phrases=''


    step_1b_prompt = gen_step_prompt(args, step_number='1b')

    step_1b_prompt += 'Sentence: ' + utterance + '\n'
    if (args.type_condition != 'none'):
        step_1b_prompt = condition_intent_info(args, step_1b_prompt,intent, intent_map)
    
    step_1b_prompt += structure_map[args.structure_rep] + ': ' + '\n'

    if (args.output_for == 'api'):
        amr_graph = call_openai(args,step_1b_prompt, args.number_output, args.temperature)
        
        print("STEP 1b: Get AMR Graph", step_1b_prompt)
        print ("AMR", amr_graph)
    else:
        print("STEP 1b: Get AMR Graph", step_1b_prompt)
        amr_graph=''


        amr_graph=''


    step_1a_prompt = gen_step_prompt(args, step_number='1a')

    #step_1a_prompt = new_session + step_1a_prompt

    step_1a_prompt += 'Sentence: ' + utterance + '\n'
    #if (args.add_domain == 'true'):
    #    step_1a_prompt += 'Domain Name: ' + domain_name + '\n'
    step_1a_prompt += 'Intent type: ' + '\n'
    if (args.output_for == 'api'):
        intent = call_openai(args,step_1a_prompt, args.number_output, args.temperature)
        print("STEP 1a: Get Intent ", step_1a_prompt)
        print ("Intent", intent)
        #act_intent = intent_rev_map[intent.strip()]
        #print ("Mapped intent", act_intent)
    else:
        print("STEP 1a: Get Intent \n", step_1a_prompt)
        intent =''

    step_3_prompt = gen_step_prompt(args, step_number='3')
    step_3_prompt += 'Sentence: ' + utterance + '\n'
    if (args.type_condition != 'none'):
        step_3_prompt = condition_intent_info(args, step_3_prompt,intent, intent_map)
        step_3_prompt += structure_map[args.structure_rep] + ': ' + amr_graph + '\n'

    step_3_prompt += 'Key phrases: ' + key_phrases + '\n'
    if (args.add_domain == 'true'):
        #step_1a_prompt += 'Domain Name: ' + domain_name + '\n'
        step_3_prompt += 'Domain name: ' + domain_name + '\n'
    step_3_prompt += 'Slot Type, Key phrase  pairs: \n'
    if (amr_graph != ''):
        step_3_prompt += "AMR Graph: " + amr_graph
    if (args.output_for == 'api'):
        slot_type = call_openai(args,step_3_prompt, args.number_output, args.temperature)
        print("STEP 3: Get Slot Type", step_3_prompt)
        print ("STEP 3: Slot Type: ", slot_type)
    else:
        slot_type=''
        print("STEP 3: Get Slot Type", step_3_prompt)

 
    step_4_prompt = gen_step_prompt(args, step_number='4')

    step_4_prompt += 'Sentence: ' + utterance + '\n'

    step_4_prompt += "Intent: " + intent + '\n'
    step_4_prompt += "Slot Type, Slot Value pairs: " + slot_type + '\n'

    if (args.add_domain == 'true'):
        step_4_prompt += 'Domain name: ' + domain_name + '\n'
    step_4_prompt += 'Logic Form: \n'
    if (args.output_for == 'api'):
        pred_lf = call_openai(args,step_4_prompt, args.number_output, args.temperature)
        print("STEP 4: Get Logic Form", step_4_prompt)
        print ("STEP 4: OUT LF ", pred_lf)
        print ("Target", logical_form)
    else:
        pred_lf =''
        print("STEP 4: Get Logic Form", step_4_prompt)
    print ("=====================================================================")
    result.append({"utterance": utterance, "intent": intent, "AMR Graph": amr_graph, "key_phrase":
            key_phrases, "slot_type": slot_type, "pred_lf": pred_lf, "gold_lf": logical_form})
print ("Output file name",  result_output_file)
if (args.write_output == 'true'):
    json.dump(result, writer, indent=4)
