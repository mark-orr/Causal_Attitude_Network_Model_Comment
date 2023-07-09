This is the public repo for the submitted article "Theoretical Comment: The Relation Between Network Structure of the Causal Attitude Network and its Dynamics". 

Mapping between the simulation number in the submitted article and the simulation folders in this GitHub repo:
Simulation Set 1 (Study 1) —> run_1d_fix
Simulation Set 2 (Study 1) —> run_1b
Simulation Set 3 (Study 2) —> run_2b
Simulation Set 4 (Study 2) —> run_2c

These studies were run within a conda environement.  To recreate this environment, please see the spec file: ./spec_file_hopfield.txt.

Data for Study 1:
Original data source used for Study 1 was the American National Election Survey, 1984, Part 1: Pre- and Post-Election Survey.  Download instructions are provided below. 

See the following article for details on the data used for the simulations in Study 1:  Dalege, J., Borsboom, D., van Harreveld, F., van den Berg, H., Conner, M., & van der Maas, H. L. J. (2016). Toward a formalized account of attitudes: The Causal Attitude Network (CAN) model. Psychological Review, 123(1), 2–22. https://doi.org/10.1037/a0039802. Or contact Mark Orr at *mo6xj@virginia.edu

These data will not be included as part of this repo because the data holder requires that they are not shared.  However, these data are obtainable with little restriction.  It just requires registering at the data holders web portal.  

For a simulation, the data should be placed in the ./Graphs directory in reference to a run_X folder (e.g., run_1b).   We have put the detailed instructions for generating the data in each ./Graphs folder,  Here is a copy of this instruction set.

PROCEDURE IS THIS:
1. Register for ANES at: https://electionstudies.org
2. Once registered, download the stata (.dta) version of the full release (May 3, 1999 version) data file at:  https://electionstudies.org/data-center/1984-time-series-study/ 
3. Use the enclosed .R file “ANES 1984 Analyses.R” to generate the first R data file. (Note: “ANES 1984 Analyses.R” was downloaded off the website of Jonas Dalege, the first author of our target article for theoretical comment; thus is has extra code in it not used for our purposes).  The data framed generated on line 221 of this file is the data frame you want to export for ingestion by the next step; it will facilitated the next step if you call this exported data frame as “Reagan1984.Rdata”
4. Run the code IsingFit.R which generates the necessary files for the simulations, the input for this code is whatever you generated form the step above.  If you called it “Reagan1984.Rdata” then it is all set to go.

Please Note that the simulations for Study 1, Set 1 produce large volumes of disk space (approx. 50 G per fixed node).

Data For Study 2:
The initial data (to mean at the front of the simulation pipeline) for Study 2 are synthetic and can be generated from the code herein.  Here are the instructions (apply equally to run_2b and run_2c; in fact, we used the data generated for run_2b also for run_2c).

PROCEDURE IS THIS:
1. run the appropriate version of run_X_graph_gen.py.  This will created the necessary graph structures for a simulation set.  Notice that the saving directory will need mods to align with your local machine.
2. Using run_X.py will call the graphs made in step 1 for running the simulations.

Simulations and Analysis:
Each of the simulation folders has README.md files for describing each simulation.  Analysis is done in the same folders after simulation.  


EOF

