Data For  Study 1:
(Simulation folders run_1b, run_1d_fix (and all sub simulations).)

./Graphs
All data should be placed in the ./Graphs directory (in reference to a run_X folder (e.g., run_1b) for each simulation.   We have put the detailed instructions for generating the data in each ./Graphs folder,  Here is a copy of this instruction set.

PROCEDURE IS THIS:
1. Register for ANES at: https://electionstudies.org
2. Once registered, download the stata (.dta) version of the full release (May 3, 1999 version) data file at:  https://electionstudies.org/data-center/1984-time-series-study/ 
3. Use the enclosed .R file “ANES 1984 Analyses.R” to generate the first R data file. (Note: “ANES 1984 Analyses.R” was downloaded off the website of Jonas Dalege, the first author of our target article for theoretical comment; thus is has extra code in it not used for our purposes).  The data framed generated on line 221 of this file is the data frame you want to export for ingestion by the next step; it will facilitated the next step if you call this exported data frame as “Reagan1984.Rdata”
4. Run the code IsingFit.R which generates the necessary files for the simulations, the input for this code is whatever you generated form the step above.  If you called it “Reagan1984.Rdata” then it is all set to go.;

EOF.