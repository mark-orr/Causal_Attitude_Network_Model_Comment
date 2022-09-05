This is the public repo for the submitted article " " 

DATA:  Original data source used for Study 1 was the American National Election Survey, 1984, Part 1: Pre- and Post-Election Survey   can be downloaded from:  

https://www.icpsr.umich.edu/web/ICPSR/studies/8298/datadocumentation

See the following article for details on the data used for the simulations:  Dalege, J., Borsboom, D., van Harreveld, F., van den Berg, H., Conner, M., & van der Maas, H. L. J. (2016). Toward a formalized account of attitudes: The Causal Attitude Network (CAN) model. Psychological Review, 123(1), 2–22. https://doi.org/10.1037/a0039802. Or contact Mark Orr at *mo6xj@virginia.edu



These data will not be included as part of this repo.  All other data needed for running our simulations and analysis are either contained in this repo or the code needed to generate the data are contained in this repo. 

Data README for This Paper:
**Subportions of this file will be in the relative simulation folders for quick reference.

Mapping between article simulation number and simulation folders on GitHub:
Simulation Set 1 (Study 1) —> run_1d_fix
Simulation Set 2 (Study 1) —> run_1b
Simulation Set 3 (Study 2) —> run_2b
Simulation Set 4 (Study 2) —> run_2c

Data For  Study 1:
*Simulation folders run_1b, run_1d_fix (and all sub simulations).)

All data should be placed in the ./Graphs directory in reference to a run_X folder (e.g., run_1b) for each simulation.   We have put the detailed instructions for generating the data in each ./Graphs folder,  Here is a copy of this instruction set.

PROCEDURE IS THIS:
1. Register for ANES at: https://electionstudies.org
2. Once registered, download the stata (.dta) version of the full release (May 3, 1999 version) data file at:  https://electionstudies.org/data-center/1984-time-series-study/ 
3. Use the enclosed .R file “ANES 1984 Analyses.R” to generate the first R data file. (Note: “ANES 1984 Analyses.R” was downloaded off the website of Jonas Dalege, the first author of our target article for theoretical comment; thus is has extra code in it not used for our purposes).  The data framed generated on line 221 of this file is the data frame you want to export for ingestion by the next step; it will facilitated the next step if you call this exported data frame as “Reagan1984.Rdata”
4. Run the code IsingFit.R which generates the necessary files for the simulations, the input for this code is whatever you generated form the step above.  If you called it “Reagan1984.Rdata” then it is all set to go.

Please Note that the simulations for Study 1 (in the manuscript this is in relation to simulation Set 1)

of Study 1 generation of Study 1 files produces large volume of disk space (approx. 50 G per fixed node).

Data For Study 2:
These data are generated within each study.  

EOF

