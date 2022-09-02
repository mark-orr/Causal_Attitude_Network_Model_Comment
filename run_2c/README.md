README for /Users/biocomplexity/Projects/SocioCognitiveModeling/NSF-NetCogSys/Products/CAN_Model/CAN_GitRepo/IsingFitStudy/Hopfield/Simulations/run_2b

run_2b_graph_gen.py:
    Makes final graphs for n 1000, k 10

run_2b.py:
    Final sims for graphs in above.  is sourceable.

binomial.py:
    Is the file that computes the probability that one bit/node
    in a hopfield net will flip based on negative edges.  It outputs
    catch_pXX.csvs.  It was copied from ~/orrthonfunctions.  It doesn work
    with conda env 'hopfield' because the math function .comb is blocked, so
    it was run in ~/orrthonfunctions

EOF