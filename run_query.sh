demo_file='demo_5_label'
test_file='test_out_domain_seed'  # Report results with this instead
output='api'
#test_file='test_single_seed' # this is only for testing
#output='test'
dataset='MTOP'
seeds=(111 222 333)
write='true'

for s in ${seeds[@]}; do
	python query_to_lf.py --type_condition='control_single' --add_demo='false' --output_for=${output} --number_output=10 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='true' --add_demo='true' --number_demo='5'

	python query_to_lf.py --type_condition='control_single' --add_demo='false' --output_for=${output} --number_output=10 --condition_on='label' --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='true'
done

