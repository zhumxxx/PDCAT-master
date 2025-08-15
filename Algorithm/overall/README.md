In this repository, we provide our code for overall comparison.


## Techniques


### The folder `overall` contains all the techniques code for overall experiment.

#### The folder `PDCAT-algorithm` is the code for **PDCAT: Preference-Driven Compiler Auto-Tuning**.  
It contains `constraints.txt`, `data.txt`, `tuning_flags.txt`, `PDCAT.py` and `PDCAT_c.py`.  
`constraints.txt` contains collected combined constraints.  
`data.txt` contains the initial training data. You can append more tuning data to this file.  
`constraints.txt` contains collected combined constraints. The constraint format is expressed as flag A -> flag B (indicating a dependency) or flag A and flag B (indicating a synergy). You can add new constraints using the same syntax.  
`tuning_flags.txt` contains the target optimization flags. You can modify or replace them as needed for tuning experiments.  
`PDCAT.py` is the main file to tune programs from PolyBench and `PDCAT_c.py` is the main file to tune programs from cBench.  
For PolyBench:  
If you want to use it to tune program `symm`, you can run command `python PDCAT.py --log_file=symm_pdcat.log --source_path=/data/mingxuanzhu/PDCAT-master/Benchmarks/polyBench/linear-algebra/blas/symm --gcc_path=gcc --flag_path=tuning_flags.txt --sequences_path=data.txt --constraints_path=constraints.txt --permax=2.5 --permin=0.8`.  
In this command, `--log_file` is your log file name, `--source_path` is your program path, `--gcc_path` is your compiler path, `--sequences_path` is initial tuning data, `--flag_path` is for your tuning optimization flags, and `--constraints_path` is for your constraints.   
For cBench:  
If you want to use it to tune program `automotive_bitcount`, you can run command `python PDCAT_c.py --log_file=automotive_bitcount_pdcat.log --source_path=/data/mingxuanzhu/PDCAT-master/Benchmarks/cBench/automotive_bitcount/src --gcc_path=gcc --flag_path=tuning_flags.txt --sequences_path=data.txt --constraints_path=constraints.txt --permax=2.5 --permin=0.8 --exec_param=1125000`. 
`--exec_param` is execution parameter.


#### The `CompTuner-algorithm` is the code for **Compiler Autotuning through Multiple Phase Learning**. 
It contains `tuning_flags.txt`, `CompTuner.py` and `CompTuner_c.py`.  
`tuning_flags.txt` contains the target optimization flags. You can modify or replace them as needed for tuning experiments.  
`CompTuner.py` is the main file to tune programs from PolyBench and `CompTuner_c.py` is the main file to tune programs from cBench.  
For PolyBench:  
If you want to use it to tune program `correlation`, you can run command `python CompTuner.py --log_file=correlation_comptuner.log --source_path=/data/mingxuanzhu/PDCAT-master/Benchmarks/polyBench/datamining/correlation --gcc_path=gcc --flag_path=tuning_flags.txt`.  
In this command, `--log_file` is your log file name, `--source_path` is your program path, `--gcc_path` is your compiler path, and `--flag_path` is for your tuning optimization flags.   
For cBench:  
If you want to use it to tune program `automotive_susan_c`, you can run command `python CompTuner_c.py --log_file=automotive_susan_c_compTuner.log --source_path=/data/mingxuanzhu/PDCAT-master/Benchmarks/cBench/automotive_susan_c/src --gcc_path=gcc --flag_path=tuning_flags.txt --exec_param="Benchmarks/cBench/automotive_susan_data/1.pgm output_large.corners.pgm -c"`. 
`--exec_param` is execution parameter.


#### The `RIO-algorithm` is the code for **Random Iterative Optimization**. 
It contains `tuning_flags.txt`, `RIO.py` and `RIO_c.py`.  
`tuning_flags.txt` contains the target optimization flags. You can modify or replace them as needed for tuning experiments.  
`RIO.py` is the main file to tune programs from PolyBench and `RIO_c.py` is the main file to tune programs from cBench.  
For PolyBench:  
If you want to use it to tune program `correlation`, you can run command `python RIO.py --log_file=correlation_rio.log --source_path=/data/mingxuanzhu/PDCAT-master/Benchmarks/polyBench/datamining/correlation --gcc_path=gcc --flag_path=tuning_flags.txt`.  
In this command, `--log_file` is your log file name, `--source_path` is your program path, `--gcc_path` is your compiler path, and `--flag_path` is for your tuning optimization flags.   
For cBench:  
If you want to use it to tune program `automotive_susan_c`, you can run command `python RIO_c.py --log_file=automotive_susan_c_rio.log --source_path=/data/mingxuanzhu/PDCAT-master/Benchmarks/cBench/automotive_susan_c/src --gcc_path=gcc --flag_path=tuning_flags.txt --exec_param="Benchmarks/cBench/automotive_susan_data/1.pgm output_large.corners.pgm -c"`. 
`--exec_param` is execution parameter.


#### The `CFSCA-algorithm` is the code for **Compiler Auto-tuning via Critical Flag Selection**. 
It contains `tuning_flags.txt`, `getRelated.py`, `CFSCA.py` and `CFSCA_c.py`.  
`tuning_flags.txt` contains the target optimization flags. You can modify or replace them as needed for tuning experiments.  
`getRelated.py` is the file to obtian the related optimization flags with the target programs. You firstly need run `python getRelated.py --source_path=/data/mingxuanzhu/PDCAT-master/Benchmarks/polyBench/datamining/correlation --flag_path=tuning_flags`, to obtain the related flags of the target program.
`CFSCA.py` is the main file to tune programs from PolyBench and `CFSCA_c.py` is the main file to tune programs from cBench.  
For PolyBench:  
If you want to use it to tune program `correlation`, you can run command `python CSFCA.py --log_file=correlation_cfsca.log --source_path=/data/mingxuanzhu/PDCAT-master/Benchmarks/polyBench/datamining/correlation --gcc_path=gcc --flag_path=tuning_flags.txt --related_flags=1,2,3,4,5,6,7,8,9,10`.  
In this command, `--log_file` is your log file name, `--source_path` is your program path, `--gcc_path` is your compiler path, `--flag_path` is for your tuning optimization flags, and `--related_flags` are the indexs of the related flags.   
For cBench:  
If you want to use it to tune program `automotive_susan_c`, you can run command `python CSFCA_c.py --log_file=automotive_susan_c_cfsca.log --source_path=/data/mingxuanzhu/PDCAT-master/Benchmarks/cBench/automotive_susan_c/src --gcc_path=gcc --flag_path=tuning_flags.txt --related_flags=1,2,3,4,5,6,7,8,9,10 --exec_param="Benchmarks/cBench/automotive_susan_data/1.pgm output_large.corners.pgm -c"`. 
`--exec_param` is execution parameter.


#### The `SRTuner-algorithm` is the code for **Effective Compiler Optimization Customization by Exposing Synergistic Relations**.  
For PolyBench:  
If you want to use it to tune program `correlation`, you can run command `python tune.py --log_file=correlation_srtuner.log --source_path=/data/mingxuanzhu/PDCAT-master/Benchmarks/polyBench/datamining/correlation --gcc_path=gcc --flag_path=tuning_flags.txt`.  
In this command, `--log_file` is your log file name, `--source_path` is your program path, `--gcc_path` is your compiler path, `--flag_path` is for your tuning optimization flags, and `--related_flags` are the indexs of the related flags.   
For cBench:  
If you want to use it to tune program `automotive_susan_c`, you can run command `python CSFCA_c.py --log_file=automotive_susan_c_srtuner.log --source_path=/data/mingxuanzhu/PDCAT-master/Benchmarks/cBench/automotive_susan_c/src --gcc_path=gcc --flag_path=tuning_flags.txt --exec_param="Benchmarks/cBench/automotive_susan_data/1.pgm output_large.corners.pgm -c"`. 
`--exec_param` is execution parameter.
