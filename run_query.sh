demo_file='demo_5_label'
test_file='test_out_domain_seed'  # Report results with this instead
output='api'
test_file='test_single_seed' # this is only for testing 2 samples (comment out and update $seeds to run the complete experiments)
dataset=${1-'MTOP'}
model_type=${2-'palm'}
add_demo=${3-'false'}
#seeds=(111 222 333)
seeds=(111)
write='false'


for s in ${seeds[@]}; do
	python query_to_lf.py --add_demo=${add_demo} --output_for=${output} --dataset=${dataset} --test_file=${test_file} --seed=$s --write_output=${write} --add_domain='true' --model_type=${model_type}
done

