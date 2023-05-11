# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#			http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Metrics for TOP and MTOP parses."""


#from .top_utils import *
from top_utils import * 
from seqeval.metrics import f1_score, precision_score, recall_score


def _safe_divide(x, y):
		return x / y if y != 0 else 0.0


def convert_slots_to_bio(input_text, slot_types, slot_values):
	text_list = input_text.split(' ')
	slot_output = ['O']* len(text_list)
	for idx in range (len(slot_types)):
		cur_type, cur_value = slot_types[idx], slot_values[idx]
		cur_slot_type = cur_type.split(':')[-1]
		value_list = cur_value.split(' ')
		try:
			start_idx = text_list.index(value_list[0])

			slot_output[start_idx] = 'B-' + cur_slot_type
			if (len(value_list) > 1):
				end_idx = text_list.index(value_list[-1])
				slot_output[start_idx +1: end_idx + 1] = ['I-' + cur_slot_type] * (end_idx - start_idx)
		except:
				pass

	#print ("Output", slot_output)
			
	return slot_output
			
		
def top_metrics(targets, predictions, input_texts):
                """Returns eval metrics for TOP and MTOP datasets."""
                num_correct = 0
                num_total = 0
                num_invalid = 0

                num_intent_correct = 0
                num_frame_correct = 0
#print ("PRedictions", predictions)
                target_bio, predicted_bio = [], []
                counter = 0
                for target, predicted, text_input in zip(targets, predictions, input_texts):
                                if target == predicted:
                                                num_correct += 1
                                                #print ("Correct case",  target, predicted)
                                num_total += 1
                                target = target.replace('""', '')
                                target_lf = deserialize_top(target)
                                predicted_lf = deserialize_top(predicted)


                                if (predicted.startswith('[')):
                                    print ("Target-Pred-Text", target, predicted, text_input)
                                    #print ("Check label lf", target, target_lf)
                                    #print ("Check pred lf", predicted, predicted_lf)
                                    print ("\n")

                                #print ("Target", target, target_lf)
                                #print ("Predict", predicted, predicted_lf)
                                assert target_lf is not None
                                if not predicted_lf:
                                                num_invalid += 1
                                                continue
                                #print ("Pass here?")
                                target_frame, target_slot_types, target_slot_values = get_frame_top(target_lf)
                                predicted_frame, predicted_slot_types, predicted_slot_values = get_frame_top(predicted_lf)
                                target_intent = target_frame.split("-")[0]
                                predicted_intent = predicted_frame.split("-")[0]
                                
                                #text_input = seq_datasets[counter]['text_in']
                        

                                #print ("Text", text_input) 
                                #print ("Target", target, target_frame)
                                #print ("Target Info", target_slot_types, target_slot_values)
                                #print ("Predict", predicted, predicted_frame)
                                #print ("Predict Info", predicted_slot_types, predicted_slot_values)
                                # Given text input, slot_type, slot_values, convert to B-I-O format 

                                target_bio_tags = convert_slots_to_bio(text_input, target_slot_types, target_slot_values)
                                target_bio.append(target_bio_tags)


                                predicted_bio_tags = convert_slots_to_bio(text_input, predicted_slot_types, predicted_slot_values)
                                predicted_bio.append(predicted_bio_tags)

                                #print ("BIO (pred-target)", predicted_bio_tags , '\t', target_bio_tags)
                                #print ("\n")

                 

                                num_intent_correct += int(predicted_intent == target_intent)
                                num_frame_correct += int(predicted_frame == target_frame)
                                counter += 1
                                #print ("Frame", predicted_frame, '\t', target_frame)
                                #print ("Intent", predicted_intent, '\t', target_intent)
                print ("Num invalid, total", num_invalid, num_total)
                f1 = round(f1_score(target_bio, predicted_bio),4)
                precision = round(precision_score(target_bio, predicted_bio),4)
                recall= round(recall_score(target_bio, predicted_bio),4)
                #print ("ori score", num_frame_correct, num_correct)
                return dict(
                                num_total=num_total,
                                slot_f1 = f1,
                                slot_precision = precision,
                                slot_recall = recall,
                                exact_match=_safe_divide(num_correct, num_total),
                                intent_accuracy=_safe_divide(num_intent_correct, num_total),
                                semantic_frame_match=_safe_divide(num_frame_correct, num_total),
                                invalid_predictions=_safe_divide(num_invalid, num_total))
