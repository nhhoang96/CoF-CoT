# llm_nlu
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
python query_to_lf.py --type_condition='none' --add_demo='false' --output_for='api' --structure_rep=${rep}
```

where rep ```${rep}```: Structured representation as either **'dp'**: for Dependency Parsing, **'cp'**: Constituency Parsing, **'amr'**: AMR Graph


(2) Run exp with CoT conditioning & CoT conditioning-filter (fixed from note May 30th) 
```
python query_to_lf.py --type_condition='control_single' --add_demo='false' --output_for='api' 
python query_to_lf.py --type_condition='control' --add_demo='false' --output_for='api' 
python query_to_lf.py --type_condition='control_filter' --add_demo='false' --output_for='api' 
```

**DONE** ~~(3) Generate Label Definitions and output as a JSON file for future loading. The output should be generated as ./nlu_data/mtop_flat_simple/intent_vocab_map.jsonl and ./nlu_data_mtop_flat_simple/slot_vocab_map.jsonl~~
```
python query_label_desc.py
```


