In this repository, we provide our data.


## Result


### The folder `overall` contains all the results for overall experiment.

The `overall_first_selection.txt` contains the results of compared techniques on the testing programs for the first selection.
In the first selection, the initial tuning programs are:
C1, C2, C5, C6, C7, C8, C10, C11, C12, C13, C14, C15, C17, C18, C19, C20, C21, C22, C24, C27, C28,
P1, P3, P5, P7, P9, P12, P13, P14, P15, P16, P17, P18, P19, P21, P22, P24, P25, P27, P28
the testing programs are: 
C3, C4, C9, C16, C23, C25, C26, P2, P4, P6, P8, P10, P11, P20, P23, P26, P29, P30

The `overall_second_selection.txt` contains the results of compared techniques on the testing programs for the second selection.
In the first selection, the initial tuning programs are:
C1, C2, C5, C6, C7, C8, C10, C11, C12, C13, C14, C15, C17, C18, C19, C20, C21, C22, C24, C25, C26, 
C27, C28, P1, P2, P3, P5, P7, P9, P12, P12, P16, P17, P18, P19, P21, P22, P23, P24, P25
the testing programs are: 
C3, C4, C9, C16, C23, P4, P6, P8, P10, P11, P14, P15, P16, P26, P27, P28, P29, P30

The `overall_third_selection.txt` contains the results of compared techniques on the testing programs for the third selection.
In the first selection, the initial tuning programs are:
C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C19, C20, C21, C22, C24, C26, C27, C28,
P1, P3, P5, P7, P9, P11, P12, P13, P14, P15, P17, P18, P19, P20, P24, P25, P26, P30
the testing programs are: 
C15, C16, C17, C18, C23, C25, C30, P4, P6, P8, P10, P16, P21, P22, P23, P27, P28, P30

### The folder `ablation` contains all the results for ablation study.

The `initial_enable_probabilities_aqc.txt` contains the results of:

- **PDCAT** 
- Variant PDCAT<sub>NoInitial</sub> (i.e., removing initial enable probabilities acquisition part of PDCAT)
- Variant PDCAT<sub>10</sub>  (i.e., using 10 initial tuning programs for initial enable probabilities acquisition)
- Variant PDCAT<sub>20</sub> (i.e., using 20 initial tuning programs for initial enable probabilities acquisition)

The `tuning_on_program.txt` contains the results of:

- **PDCAT**
- Variant PDCAT<sub>NoTuning</sub> (i.e., removing tuning on target program part of PDCAT)

The `combined_optimization_analysis.txt` contains the results of:

- **PDCAT**
-  Variant PDCAT<sub>NoComb</sub>  (i.e., removing combined optimization analysis part of PDCAT)
-  Variant CFSCA<sub>Comb</sub>  (i.e., adding combined optimization analysis part to CFSCA)
-  Variant CompTuner<sub>NoComb</sub>  (i.e., adding combined optimization analysis part to CompTuner)
-  Variant SRTuner<sub>Comb</sub>  (i.e., adding combined optimization analysis part to SRTuner)

The `common_optimization.txt` contains the results of:

- **PDCAT**
- Variant PDCAT<sub>AllExp</sub> (i.e., removing common optimizations part of PDCAT)
- Variant PDCAT<sub>O2</sub> (i.e., replacing common optimizations -O1 with -O2 of PDCAT)
- Variant PDCAT<sub>O3</sub> (i.e., replacing common optimizations -O1 with -O3 of PDCAT)
- Variant CFSCA<sub>O1</sub> (i.e., adding common optimizations to CFSCA)
- Variant CompTuner<sub>O1</sub> (i.e., adding common optimizations to CompTuner)
- Variant SRTuner<sub>O1</sub> (i.e., adding common optimizations to SRTuner)

