# llm_nlu

**NEW UPDATE**: Please create the key.txt file with your OPENAI KEY. query_to_lf.py will read the key from that file (one-line) before running ChatCompletion

Put the evaluating file under ./gpt_output and change the variable file_name in cal_metric.py to run the evaluation

# Additional Note (May 25th)
(1) Run exp for CoT (no conditioning)
```
python query_to_lf.py --type_condition='none' --add_demo='false'
```

(2) Run exp for CoT (conditioning)
```
python query_to_lf.py --type_condition='control' --add_demo='false'
```

(3) Run exp for CoT (with demos-no conditioning)
```
python query_to_lf.py --type_condition='none' --add_demo='true'
```

(4) Run exp for CoT (with demos - with conditioning)
```
python query_to_lf.py --type_condition='control' --add_demo='true'
```

# Note (May 30th)
(1) Run exp for CoT (with demos - with conditioning) - fix prompt to remove the confidence score in the output, add AMR graph conditioning (main prompt)
```
python query_to_lf.py --type_condition='control' --add_demo='true' --output_for='api'
```

(2) Run exp for CoT (with demos - with conditioning & 0.8 filtering)
```
python query_to_lf.py --type_condition='control_filter' --add_demo='true' --output_for='api'
```

**DONE** ~~(3) Generate AMR labels to prepare for future Demonstration Selection (use training dataset). Write output to data directory, naming as train_amr.txt, one more column than train.txt. It is possible to try out only <=100 examples for now.~~
```
python query_amr.py
```

# Note (June 5th)

(1) Run exp with different structured rep  (Exp with base CoT w/o Demos)
```
python query_to_lf.py --type_condition='none' --add_demo='false' --output_for='api' --number_output=1 --structure_rep=${rep}
```

where rep ```${rep}```: Structured representation as either **'dp'**: for Dependency Parsing, **'cp'**: Constituency Parsing, **'amr'**: AMR Graph


(2) Run exp with CoT conditioning & CoT conditioning-filter (fixed from note May 30th) 
```
python query_to_lf.py --type_condition='control_single' --add_demo='false' --output_for='api' --number_output=1
python query_to_lf.py --type_condition='control' --add_demo='false' --output_for='api' --number_output=1
python query_to_lf.py --type_condition='control_filter' --add_demo='false' --output_for='api' --number_output=1
```

**DONE** ~~(3) Generate Label Definitions and output as a JSON file for future loading. The output should be generated as ./nlu_data/mtop_flat_simple/intent_vocab_map.jsonl and ./nlu_data_mtop_flat_simple/slot_vocab_map.jsonl~~
```
python query_label_desc.py
```

(4) Run baseline: SC-CoT, ComplexCoT
```
python query_to_lf.py --type_condition='none' --add_demo='false' --output_for='api' --number_output=10 --voting_method='major'
python query_to_lf.py --type_condition='none' --add_demo='false' --output_for='api' --number_output=10 --voting_method='complex'
```

(5) Run demo selection method, including: 'length', 'type', 'random' (NOTE: num_samples = 11 = # domains)
```
python query_to_lf.py --type_condition='none' --add_demo='true' --output_for='api' --number_output=1 --number_demo=11 --demo_select_criteria='length'
python query_to_lf.py --type_condition='none' --add_demo='true' --output_for='api' --number_output=1 --number_demo=11 --demo_select_criteria='type'
python query_to_lf.py --type_condition='none' --add_demo='true' --output_for='api' --number_output=1 --number_demo=11 --demo_select_criteria='random'
```


(6) Run conditioning on intent description (control-single w/o demos for now)

```
python query_to_lf.py --type_condition='control_single' --add_demo='false' --output_for='api' --number_output=1 --condition_on='descr'
```

# Final Experiments (June 7th)

See run_query.sh for the completed sets of scripts to run (all arguments)

(1) Main Experiments:

_Zero-shot Experiments_

(a1) Direct Prompting
```
python query_to_lf.py --type_prompt='direct' --number_output=1
```
(b1) CoT baselines (no demonstrations): SC-CoT, ComplexCoT
```
python query_to_lf.py --type_condition='none' --add_demo='false' --output_for='api' --number_output=10 --voting_method='major'
python query_to_lf.py --type_condition='none' --add_demo='false' --output_for='api' --number_output=10 --voting_method='complex'
```

(2) Ablation 1: No structured Rep, CP, DP, AMR
```
python query_to_lf.py --type_condition='none' --add_demo='false' --output_for='api' --number_output=1 --structure_rep='none'
python query_to_lf.py --type_condition='none' --add_demo='false' --output_for='api' --number_output=1 --structure_rep='cp'
python query_to_lf.py --type_condition='none' --add_demo='false' --output_for='api' --number_output=1 --structure_rep='dp'
python query_to_lf.py --type_condition='none' --add_demo='false' --output_for='api' --number_output=1 --structure_rep='amr'
```

(3) Ablation 2: Type of conditioning
```
python query_to_lf.py --type_condition='none' --add_demo='false' --output_for='api' --number_output=1
python query_to_lf.py --type_condition='control_single' --add_demo='false' --output_for='api' --number_output=1
python query_to_lf.py --type_condition='control' --add_demo='false' --output_for='api' --number_output=1
python query_to_lf.py --type_condition='control_filter' --add_demo='false' --output_for='api' --number_output=1
```

(4) Ablation 3: Intent Description vs Intent Name
```
python query_to_lf.py --type_condition='control_single' --add_demo='false' --output_for='api' --number_output=1 --condition_on='descr'
python query_to_lf.py --type_condition='control_single' --add_demo='false' --output_for='api' --number_output=1 --condition_on='label'
```

_Few-shot Experiments(In-context learning)_

(a2) Direct Prompting
```
python query_to_lf.py --type_prompt='direct' --add_demo='true' -- output_for='api' --number_demo=16 --number_output=1
```

**NOTE: WAIT**


(b2) CoT baselines (no demonstrations): SC-CoT, ComplexCoT, RandomCoT
```
python query_to_lf.py --type_condition='none' --add_demo='true' --number_demo=16 --output_for='api' --number_output=10 --voting_method='major' --demo_select_criteria='random'
python query_to_lf.py --type_condition='none' --add_demo='false' --number_demo=16 --output_for='api' --number_output=10 --voting_method='complex' --demo_select_criteria='length'
python query_to_lf.py --type_condition='none' --add_demo='true' --number_demo=16 --output_for='api' --number_output=1 --voting_method='major' --demo_select_criteria='random'
```

(5) Our proposed model: ZS, FS
```
python query_to_lf.py --type_condition='control_filter' --add_demo='false' --output_for='api' --number_output=10 --voting_method='major' --structure_rep='amr' --condition_on='descr'
python query_to_lf.py --type_condition='control_filter' --add_demo='true' --number_demo=16 --output_for='api' --number_output=10 --voting_method='major' --structure_rep='amr' --condition_on='descr'
```


