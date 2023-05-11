import json
import top_metrics
import re

file_loc = './gpt_output/'
#file_name='direct_dialogue'

#file_name='chain_of_dialogue'
#file_name='fixed_direct'
file_name='fixed_chain_of_dialogue'

file_full_loc = file_loc + file_name + '.jsonl'

#with open( file_full_loc, 'r') as json_file:
#    json_list = list(json_file)
#
#json_lis
#for line in json_list:
golds, preds=[],[]
texts=[]
for line in open(file_full_loc, 'r'):
    line_output = json.loads(line)
    print ("Line output", line_output)
    utt, gold_lf, pred_lf = line_output['utterance'], line_output['gold_lf'], line_output['pred_lf']

    # Fix pred_lf format
    if ("Logic Form" in pred_lf):
        pred_lf = pred_lf.split('Logic Form:')[-1].strip()

    pred_lf = pred_lf.replace(']', ' ]')
    pred_lf = pred_lf.replace('IN: ', 'IN:')

    print ("Pred lf", pred_lf)
    info_split = pred_lf.split('[')
    intent=''
    slot_name =[]
    for item in info_split:

        item = re.sub(r'[\[\]]',' ',item).strip()
        if (item.startswith('IN:')): #intent info
            name_intent = item.split(':')[-1]
            name_intent = name_intent.upper()
            intent = 'IN:' + name_intent
        elif (item.startswith('SL:')):
            item = item.replace('SL: ', 'SL:')
            slot_label = item.split(':')[-1].split(' ')[0] # slot_label

            slot_label = slot_label.upper()
            slot_names = item.split(':')[-1].split(' ')[1:] # slot_label
            slot_item_name = ' '.join(slot_names).strip()
            slot_full = 'SL:' + slot_label + ' ' + slot_item_name
            #slot_name.append(item)

            slot_name.append(slot_full)


    new_pred_lf = '['
    if (intent == ''):
        new_pred_lf += ' '
    else:
        new_pred_lf += intent 

    if (len(slot_name) == 0): # no slot 
        new_pred_lf += ' [ ]'
    else:
        for s in slot_name:

            new_pred_lf += ' [' + s + ' ]' 

    # closing
    new_pred_lf += ' ]'
    print ("Intent info:", intent)
    print ("Slot info:", slot_name)
    print ("INfo", info_split)
    print ("New pred lf", new_pred_lf)
    #intent_info = re.findall(r'(?=\[)[a-zA-Z](?<=\s)', pred_lf)

    #print ("Intent search", intent_info)
    truncate_lf = re.sub(r'[\[\]]',' ',pred_lf)
    truncate_lf = re.sub(r'\s\s+', ' ', truncate_lf)
    truncate_lf = truncate_lf.strip()
    print ("Truncate LF", truncate_lf)
    #preds.append(truncate_lf)

    items = truncate_lf.split(' ')
    print ("Items",items)
    print ("\n")
    golds.append(gold_lf)
    #preds.append(pred_lf)
    preds.append(new_pred_lf)
    texts.append(utt)


    #print ("Utt check", utt, gold_lf, pred_lf)
    #print ("\n")

eval_results=top_metrics.top_metrics(golds, preds, texts)
print ("Eval results", eval_results)
