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

