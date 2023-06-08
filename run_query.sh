demo_file='demo_16_label'
test_file='test_out_domain_seed'
dataset='MTOP'
output='api'
seeds=(111 222 333)
#python query_to_lf.py --type_condition='none' --add_demo='false' --output_for='test' --number_output=1 --structure_rep='none' --voting_method='complex' --number_demo=1 --demo_select_criteria='length' --condition_on='descr' --demo_file=${demo_file} --test_file=${test_file} --dataset=${dataset}

#python query_to_lf.py --type_condition='none' --add_demo='true' --output_for='test' --number_output=1 --structure_rep='amr' --voting_method='complex' --number_demo=16 --demo_select_criteria='length' --condition_on='descr' --demo_file=${demo_file} --test_file=${test_file} --dataset=${dataset} --type_prompt='direct'

for s in ${seeds[@]}; do
	# Direct Prompting (ZS,FS):
	python query_to_lf.py --type_prompt='direct' --number_output=1  --output_for=${output} --dataset=${dataset} --test_file=${test_file} --seed=$s
	python query_to_lf.py --type_prompt='direct' --add_demo='true' --output_for=${output} --number_demo=16 --number_output=1 --dataset=${dataset} --test_file=${test_file} --seed=$s

	# Structured Rep Ablation (ZS)
	python query_to_lf.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=1 --structure_rep='none' --dataset=${dataset} --test_file=${test_file} --seed=$s
	python query_to_lf.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=1 --structure_rep='cp' --dataset=${dataset} --test_file=${test_file} --seed=$s
	python query_to_lf.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=1 --structure_rep='dp' --dataset=${dataset} --test_file=${test_file} --seed=$s
	python query_to_lf.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=1 --structure_rep='amr' --dataset=${dataset} --test_file=${test_file} --seed=$s

	# Conditioning Ablation (ZS)
	python query_to_lf.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=1 --dataset=${dataset} --test_file=${test_file} --seed=$s
	python query_to_lf.py --type_condition='control_single' --add_demo='false' --output_for=${output} --number_output=1 --dataset=${dataset} --test_file=${test_file} --seed=$s
	python query_to_lf.py --type_condition='control' --add_demo='false' --output_for=${output} --number_output=1 --dataset=${dataset} --test_file=${test_file} --seed=$s
	python query_to_lf.py --type_condition='control_filter' --add_demo='false' --output_for=${output} --number_output=1 --dataset=${dataset} --test_file=${test_file} --seed=$s

	# Intent Name vs Intent Description Ablation (ZS):
	python query_to_lf.py --type_condition='control_single' --add_demo='false' --output_for=${output} --number_output=1 --condition_on='descr' --dataset=${dataset} --test_file=${test_file} --seed=$s
	python query_to_lf.py --type_condition='control_single' --add_demo='false' --output_for=${output} --number_output=1 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s


	# Baseline results (ZS): SC-COT, COMPLEXCOT
	python query_to_lf.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=10 --voting_method='major' --dataset=${dataset} --test_file=${test_file} --seed=$s
	python query_to_lf.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=10 --voting_method='complex' --dataset=${dataset} --test_file=${test_file} --seed=$s


	#Few-shot: HOLD
	# Baseline FS: SC-COT, COMPLEXCOT, RandomCOT (RetrieveCOT)
	#python query_to_lf.py --type_condition='none' --add_demo='true' --number_demo=16 --output_for=${output} --number_output=10 --voting_method='major' --demo_select_criteria='random' --dataset=${dataset} --test_file=${test_file} --seed=$s
	#python query_to_lf.py --type_condition='none' --add_demo='true' --number_demo=16 --output_for=${output} --number_output=10 --voting_method='complex' --demo_select_criteria='length' --dataset=${dataset} --test_file=${test_file} --seed=$s
	#python query_to_lf.py --type_condition='none' --add_demo='true' --number_demo=16 --output_for=${output} --number_output=1 --voting_method='major' --demo_select_criteria='random' --dataset=${dataset} --test_file=${test_file} --seed=$s

	##Ours
done

