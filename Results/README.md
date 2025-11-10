Paper results are stored in `BendRegOCT/ExperimentName/ExperimentName.csv`. Only the raw csv results are included, 
logs for each experiment can be made available if requested. EQP initial cut results also a have padded version, which are
used for plotting results over all datasets, not just those for which each maximum split set size has a marginal benefit.
The most important columns are:

- Model
- Dataset
- Encoding
- Buckets
- Model Status
- Objective
- Bound
- Gap
- Solve Time

Other columns store the settings used, HPC runtime and memory statistics, initial cut statistics, and callback subroutine statistics 

Scripts for creating figures and tables from the results are available in `Scripts/ResultsProcessing/`. Settings to use those scripts
to create the figures and tables from the paper are available in `Figures/Settings/` and `Tables/Settings/` respectively.

