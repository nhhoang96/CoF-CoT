file_loc='./result/'
file_name='chain_of_demo_true_condition_none_dialogue'
#file_name='chain_of_demo_true_condition_control_dialogue'
python3 cal_metric.py --eval_file ${file_name} --eval_loc ${file_loc}
