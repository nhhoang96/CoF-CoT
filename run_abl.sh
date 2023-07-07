demo_file='demo_5_label'
test_file='test_out_domain_seed'  # Report results with this instead
output='api'
#test_file='test_single_seed' # this is only for testing
#output='test'
dataset='MASSIVE'
#dataset='MASSIVE'
seeds=(111 222 333)
write='true'

for s in ${seeds[@]}; do
	python query_abl.py --dataset=${dataset} --output_for=${output} --test_file=${test_file} --seed=${s} --write_output=${write} --add_demo='false'
	# Direct Prompting (ZS,FS):
	#-------ZS--------
	#python query_baseline.py --type_prompt='direct' --number_output=1 --voting_method='major' --output_for=${output} --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write}


	#python query_baseline.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=1 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='false' --voting_method='major'

	#python query_baseline.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=10 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='false' --voting_method='major'

	#python query_baseline.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=10 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='false' --voting_method='complex'

	#------ FS

	#python query_baseline.py --type_prompt='direct' --number_output=1 --voting_method='major' --output_for=${output} --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_demo='true' --number_demo=5


	#python query_baseline.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=1 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='false' --voting_method='major' --add_demo='true' --number_demo=5

	#python query_baseline.py --type_condition='none' --output_for=${output} --number_output=10 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='false' --voting_method='major' --add_demo='true' --number_demo=5 --demo_file=${demo_file}

	#python query_baseline.py --type_condition='none' --output_for=${output} --number_output=10 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='false' --voting_method='complex' --add_demo='true' --number_demo=5

	#python query_to_lf.py --type_condition='control_filter' --add_demo='false' --output_for=${output} --number_output=1 --condition_on='descr' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='false'

	#python query_baseline.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=1 --condition_on='descr' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='false' --voting_method='major'

	#python query_baseline.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=10 --condition_on='descr' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='false' --voting_method='complex'


	#python query_to_lf.py --type_prompt='direct' --number_output=10 --voting_method='complex' --output_for=${output} --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write}
	#python query_to_lf.py --type_prompt='direct' --add_demo='true' --output_for=${output} --number_demo=16 --number_output=1 --dataset=${dataset} --test_file=${test_file} --seed=$s --demo_file=${demo_file} --write_output=${write}

	## Structured Rep Ablation (ZS)
	#python query_to_lf.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=1 --structure_rep='none' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write}
	#python query_to_lf.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=1 --structure_rep='cp' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write}
	#python query_to_lf.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=1 --structure_rep='dp' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write}
	#python query_to_lf.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=1 --structure_rep='amr' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output='true'

	## Conditioning Ablation (ZS)
	#python query_to_lf.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=1 --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write}
	#python query_to_lf.py --type_condition='control_single' --add_demo='false' --output_for=${output} --number_output=1 --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write}
	#python query_to_lf.py --type_condition='control' --add_demo='false' --output_for=${output} --number_output=1 --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write}
	#python query_to_lf.py --type_condition='control_filter' --add_demo='false' --output_for=${output} --number_output=1 --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write}
	#python query_to_lf.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=1 --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --condition_on='descr'

	## Intent Name vs Intent Description Ablation (ZS):
	#python query_to_lf.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=1 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write}

	#python query_to_lf.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=1 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='true'

	#---Best Model?
	#python query_to_lf.py --type_condition='control_single' --output_for=${output} --number_output=1 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='true' --add_demo='true' --demo_file=${demo_file} --number_demo=5
	#python query_to_lf.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=1 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='false'
	#python query_to_lf.py --type_condition='control_filter' --add_demo='false' --output_for=${output} --number_output=1 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='false'
	#python query_to_lf.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=1 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='false'

	#python query_to_lf.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=10 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='false'


	#python query_to_lf.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=10 --condition_on='descr' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='false'

	#python query_baseline.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=10 --condition_on='descr' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='false' --voting_method='major'

	#python query_baseline.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=10 --condition_on='descr' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='false' --voting_method='complex'


	#python query_to_lf.py --type_condition='control_single' --add_demo='false' --output_for=${output} --number_output=1 --condition_on='descr' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='false'

	#python query_to_lf.py --type_condition='control_filter' --add_demo='false' --output_for=${output} --number_output=1 --condition_on='descr' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='false'



	## Baseline results (ZS): SC-COT, COMPLEXCOT
	#python query_to_lf.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=10 --voting_method='major' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --structure_rep='amr'
	#python query_to_lf.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=10 --voting_method='complex' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --structure_rep='amr'


	#Few-shot: HOLD
	# Baseline FS: SC-COT, COMPLEXCOT, RandomCOT (RetrieveCOT)
	#python query_to_lf.py --type_condition='none' --add_demo='true' --number_demo=16 --output_for=${output} --number_output=10 --voting_method='major' --demo_select_criteria='random' --dataset=${dataset} --test_file=${test_file} --seed=$s --demo_file=${demo_file}
	#python query_to_lf.py --type_condition='none' --add_demo='true' --number_demo=16 --output_for=${output} --number_output=10 --voting_method='complex' --demo_select_criteria='length' --dataset=${dataset} --test_file=${test_file} --seed=$s --demo_file=${demo_file}
	#python query_to_lf.py --type_condition='none' --add_demo='true' --number_demo=16 --output_for=${output} --number_output=1 --voting_method='major' --demo_select_criteria='random' --dataset=${dataset} --test_file=${test_file} --seed=$s

	##Ours
done

