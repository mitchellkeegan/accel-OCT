We provide a number of scripts for validating and exploring experiment results. 

`ValidateResults.py` and `ValidateResultsBendWorstOCT.py` cross-check upper bounds and objectives reported by 
experiments to validate their correctness. For example that an optimal solution has not been reported which violates 
an upper bound from another experiment.

`BenchmarkFigure.py` and `BenchmarkTable.py` were used to create the figures and tables in the paper. `CreateTables.py` is
a more general script to create per-instance tables for many different instances simultaneously. It produces a raw tex file
which can be compiled to create a pdf of all tables. It can optionally produce a pdf, by default it searches for pdflatex.exe
 on your system (tested only on Windows). May require tweaking to get running. `CreateAggTable.py` is similar but collects
results into a single table and for each dataset aggregates the results.

There is naturally a large degree of overlap in their functionality, one day I may combine them
into more general scripts.