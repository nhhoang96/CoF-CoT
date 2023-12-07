demo_file='demo_5_label'
test_file='test_out_domain_seed'  # Report results with this instead
output='api'
test_file='test_single_seed' # this is only for testing 2 samples (comment out and update $seeds to run the complete experiments)
dataset='MTOP'
#seeds=(111 222 333)
seeds=(111)
write='false'


for s in ${seeds[@]}; do
	python query_to_lf.py --type_condition='control_single' --add_demo='false' --output_for=${output} --number_output=10 --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='true' --model_type='palm'
done

