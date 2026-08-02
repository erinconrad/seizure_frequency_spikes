This contains the code for running all analyses associated with the
manuscript "Automated estimation of seizure and spike burden and their 
association in a large epilepsy cohort" by Conrad et al.

Requirements:
1. spike_counts.csv and clinical_data_deidentified.csv
     - both are available for download at: https://upenn.box.com/s/et0ws47f3aq543zl781aryafr3qhu2ev
     - spike_counts.csv contains a list of spike counts at varying SpikeNet
     probability threshold for each EEG
     - clinical_data_deidentified.csv contains clinical information for
     each patient. De-identified birth dates and dates of service are
     provided by date-shifting each date relative to the date of the
     patient's first clinic visit, defined to be Jan 1 2000. E.g., if a
     patient's date of birth is 1/1/1980 and their first visit is 1/1/2010,
     then their deidentified birth date is 1/1/1970. Each row is one EEG,
     and so the same patient may appear in multiple rows.
2. Matlab (R2024a) and the Statistics and Machine Learning Toolbox
3. This codebase, located at: https://github.com/erinconrad/seizure_severity
4. Create an output directory and a data directory in the paths noted in the script for run_spike_sz_pipeline_clean.m, and place both csv datasets in the data directory.

To run the analysis, navigate to the directory containing this script and
run
>> run_spike_sz_pipeline_clean

This function will reproduce all figures, the table, and the results text
from the manuscript and save them in the output directory.