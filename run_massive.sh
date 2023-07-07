demo_file='demo_5_label'
test_file='test_out_domain_seed'  # Report results with this instead
output='api'
#test_file='test_single_seed' # this is only for testing
#output='test'
dataset='MASSIVE'
seeds=(111 222 333)
write='true'

for s in ${seeds[@]}; do
	#----- Control Conditioning
	#python query_to_lf.py --type_condition='control_filter' --add_demo='false' --output_for=${output} --number_output=10 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='true' #full model

	#python query_to_lf.py --type_condition='control_single' --add_demo='false' --output_for=${output} --number_output=10 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='true' #full model

	#python query_to_lf.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=10 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='true' #full model

	#python query_to_lf.py --type_condition='control_single' --add_demo='false' --output_for=${output} --number_output=10 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='true' --add_demo='true' --number_demo=5 --demo_file=${demo_file} 

	# Zero-shot
	python query_to_lf.py --type_condition='control_single' --add_demo='false' --output_for=${output} --number_output=10 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='true' #full model

	# Few-shot
	python query_to_lf.py --type_condition='control_single' --add_demo='false' --output_for=${output} --number_output=10 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='true' --number_demo=5 --demo_file=${demo_file}#full model

	## Structured Rep Ablation (ZS)
	#python query_to_lf.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=1 --structure_rep='none' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write}
	#python query_to_lf.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=1 --structure_rep='cp' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write}
	#python query_to_lf.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=1 --structure_rep='dp' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write}
	#python query_to_lf.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=1 --structure_rep='amr' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write}
done

