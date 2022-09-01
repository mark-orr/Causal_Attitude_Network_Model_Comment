#-------------------------------------#
##   Formatting the CAN Model        ##
##             for LENS              ##
#-------------------------------------#

rm(list = ls())
setwd("Graphs/")
library(qgraph)
library(IsingFit)
library(igraph)
library(reshape2)
library(ggplot2)


## Applying CAN Model -------------------------------------

#FOR CSV RUNS
#dataname <- 'X_nl0.5'
#my_data <- read.csv(paste(dataname, '.csv', sep = ''),header=FALSE)
load("Reagan1984.Rdata")
my_data <- na.omit(Reagan1984)

my_fit <- IsingFit(my_data)
origbias <- my_fit$thresholds

## Exporting Weight Lists for Set File Conversions --------
temp.igraph <- graph.adjacency(my_fit$weiadj, "undirected", diag = FALSE, weighted = TRUE)

# Undirected Graph
undir.graph <- get.adjacency(temp.igraph, attr="weight")
undir.graph <- as.data.frame(as.matrix(undir.graph))

#WRITE GRAPH AND THRESHOLDS AND DATA
write.table(undir.graph,file='graph_from_R.csv',row.names=FALSE, col.names=FALSE, sep=",")
write.table(origbias,file='thresholds_from_R.csv',row.names=FALSE, col.names=FALSE, sep=",")
write.table(my_data,file='data_from_R.csv',row.names=FALSE, col.names=FALSE, sep=",")


#EOF
