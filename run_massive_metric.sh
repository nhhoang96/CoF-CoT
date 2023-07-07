dataset_name='MASSIVE'
file_loc="./result_${dataset_name}/"
#seed_num=('222' '333')
seed_num=('111' '222' '333')
#file_name='chain_of_demo_false_condition_none_voting_major_dialogue'
#file_name='chain_of_demo_false_condition_none_voting_major_structure_cp_dialogue'
#file_name='chain_of_demo_false_condition_none_voting_major_dialogue'
#file_name='chain_of_demo_true_condition_control_dialogue'


# Simple baseline
#Direct
#file_name='direct_seed_111_demo_false_condition_control_voting_major_structure_amr_numberdemo_11_criteria_random_condname_label_dialogue'
#file_name='zs_direct_seed_111_condition_control_voting_major_structure_amr_criteria_random_condname_label_numoutput_1_dialogue'

#Direct
#for e in "${seed_num[@]}"; do
#	#ZS
#	#file_name="zs_baseline_direct_seed_${e}_voting_major_numout_1_dialogue"
#    #python3 cal_metric.py --eval_file ${file_name} --eval_loc ${file_loc}
#   #FS
#	file_name="zs_baseline_direct_seed_${e}_voting_major_numout_1_numberdemo_5_dialogue"
#	python3 cal_metric.py --eval_file ${file_name} --eval_loc ${file_loc}
#done
#
###COT baseline
## 
#for e in "${seed_num[@]}"; do
#	#----- ZS -------#
#	#file_name="zs_baseline_chain_of_seed_${e}_voting_major_numout_1_dialogue" # COT 
#	#python3 cal_metric.py --eval_file ${file_name} --eval_loc ${file_loc}
#	#file_name="zs_baseline_chain_of_seed_${e}_voting_major_numout_10_dialogue" # SC-COT
#	#python3 cal_metric.py --eval_file ${file_name} --eval_loc ${file_loc}
#	#file_name="zs_baseline_chain_of_seed_${e}_voting_complex_numout_10_dialogue" # Complex-COT
#	#python3 cal_metric.py --eval_file ${file_name} --eval_loc ${file_loc}
#
#	#----- FS -------#
#   file_name="zs_baseline_chain_of_seed_${e}_voting_major_numout_1_numberdemo_5_dialogue" # COT 
#   python3 cal_metric.py --eval_file ${file_name} --eval_loc ${file_loc}
#   file_name="zs_baseline_chain_of_seed_${e}_voting_major_numout_10_numberdemo_5_dialogue" # SC-COT
#   python3 cal_metric.py --eval_file ${file_name} --eval_loc ${file_loc}
#   file_name="zs_baseline_chain_of_seed_${e}_voting_complex_numout_10_numberdemo_5_dialogue" # Complex-COT
#   python3 cal_metric.py --eval_file ${file_name} --eval_loc ${file_loc}
#done

#---Reverse Order & Random Order
file_name="ablation_revorder"
python3 cal_metric.py --eval_file ${file_name} --eval_loc ${file_loc}
file_name="ablation_randorder"
python3 cal_metric.py --eval_file ${file_name} --eval_loc ${file_loc}


# Ablation: Conditioning Types:
#cond_type=('none' 'control_single' 'control_filter')
#cond_type=('control_filter')
#cond_type=('control_single' 'none')
#cond_type=('control_single')
#for e in "${seed_num[@]}"; do
#    for c in "${cond_type[@]}"; do
#        #file_name="zs_chain_of_seed_${e}_condition_${c}_voting_major_structure_amr_criteria_random_condname_descr_numoutput_10_domain_false_dialogue" # description based
#        #python3 cal_metric.py --eval_file ${file_name} --eval_loc ${file_loc}
#        file_name="zs_chain_of_seed_${e}_condition_${c}_voting_major_structure_amr_criteria_random_condname_label_numoutput_10_domain_true_dialogue" # label name based
#
#		file_name="fs_chain_of_seed_${e}_condition_${c}_voting_major_structure_amr_numberdemo_5_criteria_random_condname_label_numoutput_10_dialogue"
#        #file_name="zs_chain_of_seed_${e}_condition_${c}_voting_major_structure_amr_criteria_random_condname_label_numoutput_10_domain_true_dialogue" # fs
#        #file_name="zs_chain_of_seed_${e}_condition_${c}_voting_major_structure_amr_criteria_random_condname_label_numoutput_10_domain_false_dialogue" # label name based
#        python3 cal_metric.py --eval_file ${file_name} --eval_loc ${file_loc}
#    done
#done


#---- Full Model ---
#cond_type=('control_single' 'none')
#for e in "${seed_num[@]}"; do
#    for c in "${cond_type[@]}"; do
#        #file_name="zs_chain_of_seed_${e}_condition_${c}_voting_major_structure_amr_criteria_random_condname_descr_numoutput_10_domain_false_dialogue" # description based
#        #python3 cal_metric.py --eval_file ${file_name} --eval_loc ${file_loc}
#        file_name="zs_chain_of_seed_${e}_condition_${c}_voting_major_structure_amr_criteria_random_condname_label_numoutput_10_domain_true_dialogue" # label name based
#
#		#file_name="fs_chain_of_seed_${e}_condition_${c}_voting_major_structure_amr_numberdemo_5_criteria_random_condname_label_numoutput_10_dialogue"
#        #file_name="zs_chain_of_seed_${e}_condition_${c}_voting_major_structure_amr_criteria_random_condname_label_numoutput_10_domain_true_dialogue" # fs
#        #file_name="zs_chain_of_seed_${e}_condition_${c}_voting_major_structure_amr_criteria_random_condname_label_numoutput_10_domain_false_dialogue" # label name based
#		#FS
#		#file_name="fs_chain_of_seed_${e}_condition_${c}_voting_major_structure_amr_numberdemo_5_criteria_random_condname_label_numoutput_10_dialogue"
#        python3 cal_metric.py --eval_file ${file_name} --eval_loc ${file_loc}
#    done
#done



#for e in "${seed_num[@]}"; do
#	file_name="zs_direct_seed_${e}_condition_control_voting_complex_structure_amr_criteria_random_condname_label_numoutput_1_domain_false_dialogue"
#	python3 cal_metric.py --eval_file ${file_name} --eval_loc ${file_loc}
#done
#SC-COT and ComplexCOT
#for e in "${seed_num[@]}"; do
#	file_name="zs_chain_of_seed_${e}_condition_none_voting_major_structure_none_criteria_random_condname_label_numoutput_10_domain_false_dialogue"
#	python3 cal_metric.py --eval_file ${file_name} --eval_loc ${file_loc}
#
#	file_name="zs_chain_of_seed_${e}_condition_none_voting_complex_structure_none_criteria_random_condname_label_numoutput_10_domain_false_dialogue"
#	python3 cal_metric.py --eval_file ${file_name} --eval_loc ${file_loc}
#done

# CP, DP, None
#file_name='chain_of_seed_111_demo_false_condition_none_voting_major_structure_none_numberdemo_11_criteria_random_condname_label_dialogue'
#file_name='chain_of_seed_111_demo_false_condition_none_voting_major_structure_none_numberdemo_11_criteria_random_condname_label_numoutput_1_dialogue'
#file_name='chain_of_seed_111_demo_false_condition_none_voting_major_structure_none_numberdemo_11_criteria_random_condname_label_dialogue'
#file_name='chain_of_seed_111_demo_false_condition_none_voting_major_structure_cp_numberdemo_11_criteria_random_condname_label_dialogue'


#struct_types=('amr')
##struct_types=('none' 'cp' 'dp' 'amr')
#for e in "${seed_num[@]}"; do
#	for s in "${struct_types[@]}"; do
#		file_name="zs_chain_of_seed_${e}_condition_none_voting_major_structure_${s}_criteria_random_condname_label_numoutput_1_domain_false_dialogue" #DP
#		#echo ${file_name}
#		python3 cal_metric.py --eval_file ${file_name} --eval_loc ${file_loc}
#	done
#done

#file_name='zs_chain_of_seed_111_condition_none_voting_major_structure_amr_criteria_random_condname_descr_numoutput_1_domain_false_dialogue'
#file_name='zs_chain_of_seed_111_condition_none_voting_major_structure_amr_criteria_random_condname_descr_numoutput_1_domain_true_dialogue'

#for e in "${seed_num[@]}"; do
#	file_name="zs_chain_of_seed_${e}_condition_none_voting_major_structure_amr_criteria_random_condname_label_numoutput_1_domain_false_dialogue"
#	python3 cal_metric.py --eval_file ${file_name} --eval_loc ${file_loc}
#	file_name="zs_chain_of_seed_${e}_condition_none_voting_major_structure_amr_criteria_random_condname_descr_numoutput_1_domain_false_dialogue"
#	python3 cal_metric.py --eval_file ${file_name} --eval_loc ${file_loc}
#done


#---- Ablation: Condition type
#file_name='zs_chain_of_seed_111_condition_none_voting_major_structure_amr_criteria_random_condname_label_numoutput_1_dialogue'
#file_name='zs_chain_of_seed_111_condition_control_filter_voting_major_structure_amr_criteria_random_condname_label_numoutput_1_dialogue'

#file_name='zs_chain_of_seed_111_condition_none_voting_major_structure_amr_criteria_random_condname_label_numoutput_1_domain_true_dialogue' #add domain
#file_name='zs_chain_of_seed_111_condition_none_voting_major_structure_amr_criteria_random_condname_descr_numoutput_1_domain_false_dialogue'
#python3 cal_metric.py --eval_file ${file_name} --eval_loc ${file_loc}

#file_name='zs_chain_of_seed_111_condition_none_voting_major_structure_amr_criteria_random_condname_label_numoutput_1_dialogue' #AMR-conditioning
#file_name='zs_chain_of_seed_111_condition_none_voting_major_structure_cp_criteria_random_condname_label_numoutput_1_dialogue' #CP
#file_name='zs_chain_of_seed_111_condition_none_voting_major_structure_none_criteria_random_condname_label_numoutput_1_dialogue' #base

#file_name='chain_of_seed_111_demo_false_condition_none_voting_major_structure_dp_numberdemo_11_criteria_random_condname_label_dialogue'
#file_name='chain_of_seed_111_demo_false_condition_none_voting_major_structure_amr_numberdemo_11_criteria_random_condname_label_dialogue'

# Condition type
#file_name='chain_of_seed_111_demo_false_condition_control_voting_major_structure_amr_numberdemo_11_criteria_random_condname_label_numoutput_1_dialogue' #control-all
#file_name='chain_of_seed_111_demo_false_condition_control_single_voting_major_structure_amr_numberdemo_11_criteria_random_condname_label_numoutput_1_dialogue' #control-single
#file_name='chain_of_seed_111_demo_false_condition_none_voting_major_structure_amr_numberdemo_11_criteria_random_condname_label_numoutput_1_dialogue' #no intent conditioning
#file_name='chain_of_seed_111_demo_false_condition_none_voting_major_structure_amr_numberdemo_11_criteria_random_condname_label_numoutput_1_dialogue'
#file_name='chain_of_seed_111_demo_false_condition_control_filter_voting_major_structure_amr_numberdemo_11_criteria_random_condname_label_numoutput_1_dialogue' # control filter

# Using label description
#file_name='zs_chain_of_seed_111_condition_none_voting_major_structure_amr_criteria_random_condname_descr_numoutput_1_domain_false_dialogue'
#file_name='zs_chain_of_seed_111_condition_none_voting_major_structure_amr_criteria_random_condname_label_numoutput_1_domain_false_dialogue'
#python3 cal_metric.py --eval_file ${file_name} --eval_loc ${file_loc}
#Baseline: SC-COT, ComplexCOT
#file_name='zs_chain_of_seed_222_condition_none_voting_major_structure_none_criteria_random_condname_label_numoutput_1_dialogue' #sc-cot
#file_name='zs_chain_of_seed_333_condition_none_voting_major_structure_none_criteria_random_condname_label_numoutput_10_domain_false_dialogue'

#file_name='zs_chain_of_seed_333_condition_none_voting_complex_structure_none_criteria_random_condname_label_numoutput_10_domain_false_dialogue'
#python3 cal_metric.py --eval_file ${file_name} --eval_loc ${file_loc}
#file_name='chain_of_seed_111_demo_false_condition_none_voting_major_structure_amr_numberdemo_11_criteria_random_condname_label_numoutput_10_dialogue'
#file_name='chain_of_seed_111_demo_false_condition_none_voting_complex_structure_amr_numberdemo_11_criteria_random_condname_label_numoutput_10_dialogue'

