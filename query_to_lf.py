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

openai.api_key = "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"
model_name = "gpt-3.5-turbo"


def get_intent_slot_vob(dataset):
    if dataset == "MTOP":
        # ---- MTOP ------#
        intent_vocab = ['add_time_timer', 'add_to_playlist_music', 'answer_call', 'cancel_call', 'cancel_message',
                        'create_alarm', 'create_call', 'create_playlist_music', 'create_reminder', 'create_timer',
                        'delete_alarm', 'delete_playlist_music', 'delete_reminder', 'delete_timer', 'dislike_music',
                        'disprefer', 'end_call', 'fast_forward_music', 'follow_music', 'get_age', 'get_airquality',
                        'get_alarm', 'get_attendee_event', 'get_availability', 'get_call', 'get_call_contact',
                        'get_call_time', 'get_category_event', 'get_contact', 'get_contact_method',
                        'get_date_time_event', 'get_details_news', 'get_education_degree', 'get_education_time',
                        'get_employer', 'get_employment_time', 'get_event', 'get_gender', 'get_group',
                        'get_info_contact', 'get_info_recipes', 'get_job', 'get_language', 'get_life_event',
                        'get_life_event_time', 'get_location', 'get_lyrics_music', 'get_major', 'get_message',
                        'get_message_contact', 'get_mutual_friends', 'get_recipes', 'get_reminder',
                        'get_reminder_amount', 'get_reminder_date_time', 'get_reminder_location', 'get_stories_news',
                        'get_sunrise', 'get_sunset', 'get_timer', 'get_track_info_music', 'get_undergrad',
                        'get_weather', 'help_reminder', 'hold_call', 'ignore_call', 'is_true_recipes', 'like_music',
                        'loop_music', 'merge_call', 'pause_music', 'pause_timer', 'play_media', 'play_music', 'prefer',
                        'previous_track_music', 'question_music', 'question_news', 'remove_from_playlist_music',
                        'repeat_all_music', 'repeat_all_off_music', 'replay_music', 'restart_timer', 'resume_call',
                        'resume_music', 'resume_timer', 'rewind_music', 'send_message', 'set_available',
                        'set_default_provider_calling', 'set_default_provider_music', 'set_rsvp_interested',
                        'set_rsvp_no', 'set_rsvp_yes', 'set_unavailable', 'share_event', 'silence_alarm',
                        'skip_track_music', 'snooze_alarm', 'start_shuffle_music', 'stop_music', 'stop_shuffle_music',
                        'subtract_time_timer', 'switch_call', 'unloop_music', 'update_alarm', 'update_call',
                        'update_method_call', 'update_reminder', 'update_reminder_date_time',
                        'update_reminder_location', 'update_reminder_todo', 'update_timer']


        slot_vocab = ['music_radio_id', 'alarm_name', 'recipes_diet', 'recipes_meal', 'group', 'attendee',
                      'music_artist_name', 'location', 'recipes_unit_nutrition', 'recipes_cooking_method',
                      'music_album_title', 'music_track_title', 'title_event', 'weather_temperature_unit',
                      'music_playlist_title', 'attendee_event', 'music_type', 'recipes_source', 'user_attendee_event',
                      'todo', 'method_timer', 'recipes_type_nutrition', 'phone_number', 'amount',
                      'method_retrieval_reminder', 'type_relation', 'category_event', 'date_time', 'job', 'news_topic',
                      'contact_method', 'age', 'recipes_attribute', 'person_reminded', 'recipes_included_ingredient',
                      'timer_name', 'recipes_type', 'major', 'news_reference', 'ordinal', 'school', 'news_type',
                      'sender', 'O', 'music_genre', 'recipes_qualifier_nutrition', 'contact_related',
                      'music_album_modifier', 'recipes_dish', 'recipient', 'recipes_excluded_ingredient', 'news_source',
                      'music_rewind_time', 'recipes_cuisine', 'recipes_time_preparation', 'contact', 'content_exact',
                      'music_provider_name', 'recipes_unit_measurement', 'news_category', 'life_event', 'type_content',
                      'employer', 'similarity', 'contact_added', 'contact_removed', 'recipes_rating', 'type_contact',
                      'weather_attribute', 'attribute_event', 'period', 'name_app', 'music_playlist_modifier',
                      'education_degree', 'method_recipes', 'gender']

    elif dataset == "MASSIVE":
        # ----- MASSIVE -----#
        intent_vocab = ['qa_currency', 'iot_wemo_off', 'general_greet', 'play_music', 'iot_coffee', 'play_radio',
                        'lists_createoradd', 'audio_volume_down', 'music_dislikeness', 'takeaway_order',
                        'music_likeness', 'weather_query', 'audio_volume_mute', 'alarm_query', 'email_sendemail',
                        'takeaway_query', 'cooking_recipe', 'lists_query', 'email_querycontact', 'iot_cleaning',
                        'social_post', 'transport_query', 'qa_stock', 'play_audiobook', 'qa_maths', 'play_game',
                        'audio_volume_other', 'music_query', 'alarm_remove', 'qa_factoid', 'general_joke',
                        'iot_hue_lightdim', 'calendar_remove', 'iot_hue_lightoff', 'play_podcasts', 'transport_ticket',
                        'email_query', 'iot_hue_lightchange', 'news_query', 'transport_taxi', 'cooking_query',
                        'general_quirky', 'lists_remove', 'calendar_set', 'recommendation_movies', 'iot_wemo_on',
                        'recommendation_locations', 'datetime_convert', 'transport_traffic', 'alarm_set',
                        'calendar_query', 'qa_definition', 'audio_volume_up', 'iot_hue_lightup', 'iot_hue_lighton',
                        'music_settings', 'recommendation_events', 'datetime_query', 'email_addcontact', 'social_query']

        slot_vocab = ['alarm_type', 'app_name', 'artist_name', 'audiobook_author', 'audiobook_name', 'business_name',
                      'business_type', 'change_amount', 'coffee_type', 'color_type', 'cooking_type', 'currency_name',
                      'date', 'definition_word', 'device_type', 'drink_type', 'email_address', 'email_folder',
                      'event_name', 'food_type', 'game_name', 'game_type', 'general_frequency', 'house_place',
                      'ingredient', 'joke_type', 'list_name', 'meal_type', 'media_type', 'movie_name', 'movie_type',
                      'music_album', 'music_descriptor', 'music_genre', 'news_topic', 'order_type', 'person',
                      'personal_info', 'place_name', 'player_setting', 'playlist_name', 'podcast_descriptor',
                      'podcast_name', 'radio_name', 'relation', 'song_name', 'sport_type', 'time', 'time_zone',
                      'timeofday', 'transport_agency', 'transport_descriptor', 'transport_name', 'transport_type',
                      'weather_descriptor']

    return intent_vocab, slot_vocab


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

# ----- Output file definition ------#
input_test_file = "nlu_data/mtop_flat_simple/en/eval.txt"
with open(input_test_file, 'r') as file:
    content = file.read()


input_fs_file = "nlu_data/mtop_flat_simple/en/fs.txt"

demo_ex=[]
ex_counter=0
for line in open(input_fs_file, 'r'):
    ex = line.strip().split('\t')
    utt = ex[0]
    lf = ex[1]
    domain=ex[3]
    amr_info = ex[-1] #when generated
    print ("AMR INFO", amr_info)
    intent, slot_pairs, slot_vals = parse_lf(lf)
    demo_dict={'utt':'', 'intent':'', 'key_phrase':'', 'pair':'', 'AMR':'', 'lf':''}
    demo_dict['utt'] = utt
    demo_dict['intent'] = intent
    demo_dict['pair'] = ','.join(map(str,slot_pairs))
    demo_dict['key_phrase'] = ','.join(slot_vals)
    demo_dict['AMR'] = amr_info
    demo_dict['lf'] = lf

    print ("Demo dict", demo_dict)
    demo_ex.append(demo_dict)
    ex_counter += 1
    #break


    
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="MTOP", choices=["MTOP", "MASSIVE"], type=str,
                    help='Type of dataset')
parser.add_argument("--type_condition", default='control', type=str, help='Kind of conditioning: (none, control, control_filter)')
parser.add_argument("--add_demo", choices=['true','false'], default='false', type=str)
parser.add_argument("--output_for", choices=['api','test'], default='test', type=str)

args = parser.parse_args()

# ---- Generic Structure
intent_vocab, slot_vocab = get_intent_slot_vob(args.dataset)
intent_str = ','.join(intent_vocab)
slot_str = ','.join(slot_vocab)
gen_step_1 = 'Given the intent vocabulary and sentence, choose 1 of the following as the intent type for the sentence: \n'
#gen_step_1c = 'Given the intent vocabulary and sentence, choose the top 3 of the following as the potential intent types for the sentence with numeric confidence scores. Return the list of intent types separated by commas, followed by the list of numeric confidence scores separated by commas: \n'

gen_step_1c = 'Given the intent vocabulary and sentence, choose the top 3 of the following as the potential intent types for the sentence. Return the list of intent types separated by commas: \n' 

gen_step_1c_filter = 'Given the intent vocabulary and sentence, choose at least 1 of the following whose confidence score is greater than or equal to 0.8 as the potential intent types for the sentence. Return the list of intent types separated by commas: \n' 

gen_step_1b = 'Given the sentence, generate one and only one Abstract Meaning Representation (AMR) graph representation in the textual Neo-Davidsonian format \n'

gen_step_1bc = 'Given the sentence and its potential intent types, generate one and only one Abstract Meaning Representation (AMR) graph representation in the textual Neo-Davidsonian format \n'

gen_step_2 = 'Based on the sentence and its AMR graph, identify a list of key phrases for the sentence. Key phrases can be made up from multiple AMR concepts. Each word in key phrases must exist in the given sentence. Return a list of key phrases separated by commas \n'

gen_step_2c = 'Based on the sentence, its potential intents and its AMR graph, identify a list of key phrases for the sentence. Key phrases can be made up from multiple AMR concepts. Each word in key phrases must exist in the given sentence. Return a list of key phrases separated by commas \n'

gen_step_3 = 'Given the slot vocabulary, the sentence and its key phrases, identify the corresponding slot type for each key phrases. Return the list of key phrases and their corresponding slot types in the following format: (slot_type, key_phrase) separated by commas \n'
gen_step_3 += 'Slot Vocabulary: ' + slot_str + '\n'

gen_step_3c = 'Given the slot vocabulary, the sentence, its potential intents, its AMR graph and its key phrases, identify the corresponding slot type for each key phrases. Return the list of key phrases and their corresponding slot types in the following format: (slot_type, key_phrase) separated by commas \n'
gen_step_3c += 'Slot Vocabulary: ' + slot_str + '\n'

gen_step_4 = 'Given the sentence, its potential intent types, its slot type and slot value pairs in (slot_type, slot_value) format, generate the logic form in the format: [IN:___ [SL:___] [SL:____]] where IN: is followed by an intent type and SL: is followed by a slot type and slot value pair separated by white space. The number of [SL: ] is unlimited. The number of [IN: ] is limited to 1 \n'

direct_prompt = "New Session: Given the intent type vocabulary, slot type vocabulary and sentence, generate logic form of the sentence in the format of [IN:___ [SL:____] [SL:___]] where IN: is followed by an intent type and SL: is followed by a slot type and slot value pair separated by white space. The number of [SL: ] is unlimited. \n"
direct_prompt += "Intent Type: " + intent_str + "\n"
direct_prompt += "Slot Type: " + slot_str + "\n"

type_prompt = "chain_of"
add_demo, condition_type = args.add_demo, args.type_condition
result_output_file = f'./result/{type_prompt}_demo_{add_demo}_condition_{condition_type}_dialogue.jsonl'
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
        pred_lf = call_chatgpt(direct_prompt)
        writer.write(json.dumps({"utterance": utterance, "pred_lf": pred_lf, "gold_lf": logical_form}) + '\n')
    else:
        # --- Step 1a: Get Intent
        if (args.type_condition == 'none'):
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
            else:
                demo_1 = gen_step_1c + 'Intent Vocabulary: ' + intent_str + '\n'
            for dem in demo_ex:
                demo_utt = dem['utt']
                demo_intent = dem['intent']
                #print ("Demo ex", utt)

                demo_1 += 'Sentence: ' + demo_utt + '\n'
                demo_1 += 'Intent type: ' + demo_intent + '\n'

            step_1a_prompt = demo_1 + '\n' + step_1a_prompt
        
        if (args.output_for == 'api'):
            intent = call_chatgpt(step_1a_prompt)
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
        
        step_1b_prompt += 'AMR Graph: ' + '\n'

        #--- Demo 1b
        if (args.add_demo == 'true'):
            if (args.type_condition == 'none'):
                demo_1b = gen_step_1b + 'Sentence: ' + demo_utt + '\n'
            else:
                demo_1b = gen_step_1bc + 'Sentence: ' + demo_utt + '\n'
            for dem in demo_ex:
                demo_utt = dem['utt']
                demo_intent = dem['intent']
                if (args.type_condition != 'none'):
                    demo_1b += 'Potential Intent Types: ' + demo_intent + '\n'
                
                demo_1b += 'AMR Graph: ' + '\n'

            step_1b_prompt  = demo_1b + '\n' + step_1b_prompt 

        if (args.output_for == 'api'):
            amr_graph = call_chatgpt(step_1b_prompt)
            print("STEP 1b: Get AMR Graph")
        else:
            print("STEP 1b: Get AMR Graph", step_1b_prompt)
            amr_graph=''

        # --- Step 2: Get Key Phrases
        if (args.type_condition == 'none'):
            step_2_prompt = gen_step_2 + 'Sentence: ' + utterance + '\n'
        else:
            step_2_prompt = gen_step_2c + 'Sentence: ' + utterance + '\n'
            step_2_prompt += 'Potential Intents: ' + intent + '\n'


        step_2_prompt += 'AMR Graph: ' + amr_graph + '\n'
        step_2_prompt += 'Key phrases: \n'

        # -- Step 2 Demo
        if (args.add_demo == 'true'):
            if (args.type_condition == 'none'):
                demo_2 = gen_step_2 + 'Sentence: ' + demo_utt + '\n'
            else:
                demo_2 = gen_step_2c + 'Sentence: ' + demo_utt + '\n'
        
            for dem in demo_ex:
                demo_utt = dem['utt']
                demo_intent = dem['intent']
                demo_kp = dem['key_phrase']
                demo_amr = dem['AMR']
                if (args.type_condition != 'none'):
                    demo_2 += 'Potential Intents: ' +  demo_intent + '\n'
                demo_2 += 'AMR Graph: ' + demo_amr + '\n'
                demo_2 += 'Key phrases: ' + demo_kp + '\n'

            step_2_prompt  = demo_2 + '\n' + step_2_prompt 

        if (args.output_for == 'api'):
            key_phrases = call_chatgpt(step_2_prompt)
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
            step_3_prompt += 'AMR Graph: ' + amr_graph + '\n'
            step_3_prompt += 'Key phrases: ' + key_phrases + '\n'
            #step_3 = gen_step_3 + 'Key phrases: ' + '\n'
            #step_3 += 'Sentence: ' + utterance + '\n'

        step_3_prompt += 'Slot Type, Key phrase  pairs: \n'


        if (args.add_demo == 'true'):
            if (args.type_condition == 'none'):
                demo_3 = gen_step_3 + 'Sentence: ' + demo_utt + '\n'
            else:
                demo_3 = gen_step_3c + 'Sentence: ' + demo_utt + '\n'
        
            for dem in demo_ex:
                demo_utt = dem['utt']
                demo_intent = dem['intent']
                demo_kp = dem['key_phrase']
                demo_pair = dem['pair']
                if (args.type_condition != 'none'):
                    demo_3 += 'Potential Intents: ' + demo_intent + '\n'
                demo_3 += 'AMR Graph: ' + demo_amr + '\n'
                demo_3 += 'Key phrases: ' + demo_kp + '\n'

                demo_3 += 'Slot Type, Key phrase pairs: ' + demo_pair + '\n'

            step_3_prompt  = demo_3 + '\n' + step_3_prompt 

        if (args.output_for == 'api'):
            slot_type = call_chatgpt(step_3_prompt)
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
        
            for dem in demo_ex:
                demo_utt = dem['utt']
                demo_intent = dem['intent']
                demo_kp = dem['key_phrase']
                demo_pair = dem['pair']

                demo_4 += "Intent: " + demo_intent + '\n'
                demo_4 += "Slot Type, Slot Value pairs: " + demo_pair + '\n'
                demo_4 += 'Logic Form: ' + dem['lf'] + '\n'


            step_4_prompt  = demo_4 + '\n' + step_4_prompt 
    if (args.output_for == 'api'):
        pred_lf = call_chatgpt(step_4_prompt)
        print("STEP 4: Get Logic Form")
    else:
        pred_lf =''
        print("STEP 4: Get Logic Form", step_4_prompt)
        writer.write(json.dumps({"utterance": utterance, "intent": intent, "AMR Graph": amr_graph, "key_phrase":
            key_phrases, "slot_type": slot_type, "pred_lf": pred_lf, "gold_lf": logical_form}) + '\n')
