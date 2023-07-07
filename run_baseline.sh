demo_file='demo_5_label'
#test_file='test_out_domain_seed'  # Report results with this instead
output='api'
test_file='test_single_seed' # this is only for testing
#output='test'
#dataset='MTOP'
dataset='MASSIVE'
seeds=(111 222 333)
#write='true'
write='false'

for s in ${seeds[@]}; do
	#-------ZS--------
	python query_baseline.py --type_prompt='direct' --number_output=1 --voting_method='major' --output_for=${output} --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write}
	python query_baseline.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=1 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='false' --voting_method='major'

	python query_baseline.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=10 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='false' --voting_method='major' --add_domain='true'

	python query_baseline.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=10 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='false' --voting_method='complex'

	#------ FS---------
	python query_baseline.py --type_prompt='direct' --number_output=1 --voting_method='major' --output_for=${output} --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_demo='true' --number_demo=5 --demo_file=${demo_file}
	python query_baseline.py --type_condition='none' --add_demo='false' --output_for=${output} --number_output=1 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='false' --voting_method='major' --add_demo='true' --number_demo=5 --demo_file=${demo_file}

	python query_baseline.py --type_condition='none' --output_for=${output} --number_output=10 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='false' --voting_method='major' --add_demo='true' --number_demo=5 --demo_file=${demo_file}

	python query_baseline.py --type_condition='none' --output_for=${output} --number_output=10 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='false' --voting_method='complex' --add_demo='true' --number_demo=5 --demo_file=${demo_file}

done

