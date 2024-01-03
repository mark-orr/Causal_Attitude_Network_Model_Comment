#########################
#Install packages (uncomment if you have not installed them already)

#install.packages("foreign")
#install.packages("IsingFit")
#install.packages("qgraph")
#install.packages("igraph")
#install.packages("bootnet")
#install.packages("IsingSampler")
#install.packages("compute.es")
#install.packages("NetworkComparisonTest")
#install.packages("mgm")
#install.packages("Matrix")

#########################
#Load packages

library(foreign)
library(IsingFit)
library(qgraph)
library(igraph)
library(bootnet)
library(IsingSampler)
library(compute.es)
library(NetworkComparisonTest)
library(mgm)
library(Matrix)

#########################
#Create example network

set.seed(1)#this is used so that each time the same random numbers are used

exNet <- sample_pa(10, .5, m = 2, directed = FALSE)#this creates a network with ten nodes and 17 edges based on the growth algorithm preferential attachment

pdf('Figure1.pdf')#the pdf function is used to save the plot to a pdf file
qgraph (get.adjacency(exNet))#this plots the network
dev.off()

#########################
#Load data 

download.file('http://www.electionstudies.org/studypages/data/anes_timeseries_2012/anes_timeseries_2012_dta.zip', 'ANES2012.zip') #downloads the zipped data file to your working directory
unzip('ANES2012.zip')
ANES2012 <- read.dta('anes_timeseries_2012_Stata12.dta')#loads the data to the object ANES2012

#########################
#Recode variables 
#Items regarding Obama

ObamaCog <- data.frame(Mor = as.numeric(ANES2012$ctrait_dpcmoral),#this creates a data frame containing the items tapping beliefs
                        Led = as.numeric(ANES2012 $ ctrait_dpclead),
                        Car = as.numeric(ANES2012$ctrait_dpccare),
                        Kno = as.numeric(ANES2012$ctrait_dpcknow),
                        Int = as.numeric(ANES2012$ctrait_dpcint),
                        Hns = as.numeric(ANES2012$ctrait_dpchonst))
ObamaCog[ObamaCog < 3] <- NA#values below 3 represent missing values
ObamaCog <- binarize(ObamaCog, 5, removeNArows = FALSE)#this binarizes the data (this is done because the model we use for simulating networks assumes binary data); (not) endorsing the beliefs is encoded as 1 (0) 

ObamaAff <- data.frame(Ang = as.numeric(ANES2012$candaff_angdpc),#this creates a data frame containing the items tapping feelings
                        Hop = as.numeric(ANES2012$candaff_hpdpc), 
                        Afr = as.numeric(ANES2012$candaff_afrdpc), 
                        Prd = as.numeric(ANES2012$candaff_prddpc))
ObamaAff[ObamaAff < 3] <- NA#values below 3 represent missing values
ObamaAff <- binarize(ObamaAff, 4, removeNArows = FALSE)#(not) endorsing the feelings is encoded as 1 (0)

Obama <- data.frame(ObamaCog,ObamaAff)#this creates a data frame containing all items tapping evaluative reactions
Obama <- na.omit(Obama)#this deletes missing values casewise

#Items regarding Romney

RomneyCog <- data.frame(Mor = as.numeric(ANES2012$ctrait_rpcmoral),
                        Led = as.numeric(ANES2012 $ ctrait_rpclead),
                        Car = as.numeric(ANES2012$ctrait_rpccare),
                        Kno = as.numeric(ANES2012$ctrait_rpcknow),
                        Int = as.numeric(ANES2012$ctrait_rpcint),
                        Hns = as.numeric(ANES2012$ctrait_rpchonst))
RomneyCog[RomneyCog < 3] <- NA
RomneyCog <- binarize(RomneyCog, 5, removeNArows = FALSE)
RomneyAff <- data.frame(Ang = as.numeric(ANES2012$candaff_angrpc),
                        Hop = as.numeric(ANES2012$candaff_hprpc), 
                        Afr = as.numeric(ANES2012$candaff_afrrpc), 
                        Prd = as.numeric(ANES2012$candaff_prdrpc))
RomneyAff[RomneyAff < 3] <- NA
RomneyAff <- binarize(RomneyAff, 4, removeNArows = FALSE)

Romney <- data.frame(RomneyCog,RomneyAff)

#########################
#Network estimation

ObamaFit <- IsingFit(Obama)
ObamaGraph <- qgraph(ObamaFit $ weiadj, layout = 'spring', cut = .8)
ObamaiGraph <- graph_from_adjacency_matrix(abs(ObamaFit $ weiadj), 'undirected', weighted = TRUE, add.colnames = FALSE)

#########################
#Community detection and plotting

ObamaCom <- cluster_walktrap(ObamaiGraph)
communities(ObamaCom)

pdf('Figure2.pdf')
qgraph(ObamaFit $ weiadj, layout = 'spring', cut = .8, groups = communities(ObamaCom), legend = FALSE)
dev.off()

#plotting the network in a colorblind-friendly way
qgraph(ObamaFit $ weiadj, layout = 'spring', cut = .8, groups = communities(ObamaCom), legend = FALSE,
       theme = 'colorblind')#this argument can be used to plot the network in way that is more readable for colorbling people

#plotting the network in greyscale
qgraph(ObamaFit $ weiadj, layout = 'spring', cut = .8, groups = communities(ObamaCom), legend = FALSE,
       theme = 'colorblind')#this argument can be used to plot the network in greyscale



#########################
#Centrality

ObamaCen <- centralityTable(ObamaGraph, standardized = FALSE)

pdf('Figure3.pdf')
centralityPlot(ObamaGraph, scale = 'raw')
dev.off()

#########################
#Connectivity

ObamaSPL <- centrality(ObamaGraph)$ShortestPathLengths
ObamaSPL <- ObamaSPL[upper.tri(ObamaSPL)]
ObamaASPL <- mean(ObamaSPL)

#########################
#Comparison between Obama and Romney network

ObaRom <- data.frame(ObamaCog,ObamaAff,Romney)#this creates a data frame containing all items regarding Obama and Romney
ObaRom <- na.omit(ObaRom)#this is done so that only individuals, who have no missing values on items regarding both Obama and Romney, are included

ObamaComp <- ObaRom[,1:10]#this selects the items regarding Obama
RomneyComp <- ObaRom[,11:20]#this selects the items regarding Romney
names(RomneyComp) <- names(ObamaComp)#this makes sure that both data frames have the same names (which is neccessary for the NCT)

set.seed(1)

NCTObaRom <- NCT(ObamaComp, RomneyComp, it = 1000, binary.data = TRUE, paired = TRUE, test.edges = TRUE, edges = 'all')
NCTObaRom$glstrinv.real 
NCTObaRom$glstrinv.pval
NCTObaRom$nwinv.real 
NCTObaRom$nwinv.pval
NCTObaRom$einv.pvals

#input for Figure 4
ObamaCompFit <- IsingFit(ObamaComp)#this creates the Obama network that was tested in the NCT
RomneyCompFit <- IsingFit(RomneyComp)#this creates the Romney network that was tested in the NCT

inputNCTgraph <- ObamaCompFit$weiadj - RomneyCompFit$weiadj#this creates the input for Figure 4, which is based on the differences in edge weights between the Obama and Romney networks
inputNCTgraph[upper.tri(inputNCTgraph)][which(NCTObaRom$einv.pvals$`p-value` >= .05)] <- 0#this sets the not-significant edge weight differences in the upper triangle of the input matrix to 0
inputNCTgraph <- forceSymmetric(inputNCTgraph)#this sets the lower triangle of the matrix to be symmetric with the upper triangle

pdf('Figure4.pdf')
qgraph(inputNCTgraph, layout = ObamaGraph$layout, edge.labels = TRUE, esize = 1)#the layout argument specifies that we want the same layout as in the Obama graph, the edge.labels argument specifies that we want to plot the values of the edge weights, esize reduces the width of the edges
dev.off()

#########################
#Simulation input 

SimInput <- LinTransform(ObamaFit$weiadj, ObamaFit$thresholds)#The LinTransform function rescales the edge weights and thresholds, so that a threshold of 0 indicates that a node has no disposition to be in a given state and a positive (negative) threshold indicates that a node has the disposition to be "on" ("off")
SimInput$graph[c(7,9),-c(7,9)] <- SimInput$graph[c(7,9),-c(7,9)]*-1#this command and the next command rescale the nodes Ang and Afr
SimInput$graph[-c(7,9),c(7,9)] <- SimInput$graph[-c(7,9),c(7,9)]*-1

#########################
#Connectivity simulation 

set.seed(1)

sampleLowTemp <- IsingSampler(1000, SimInput $ graph, rep(0,10), 1.2, responses = c(-1L,1L)) 
sampleMidTemp <- IsingSampler(1000, SimInput $ graph, rep(0,10), .8, responses = c(-1L,1L)) 
sampleHighTemp <- IsingSampler(1000, SimInput $ graph, rep(0,10), .4, responses = c(-1L,1L)) 

#the simulated data contains -1;+1 responses. To estimate networks from the simulated data using IsingFit, 0s have to be assigned -1 responses

sampleLowTempResc <- sampleLowTemp
sampleLowTempResc[sampleLowTempResc == -1] <- 0

sampleMidTempResc <- sampleMidTemp
sampleMidTempResc[sampleMidTempResc == -1] <- 0

sampleHighTempResc <- sampleHighTemp
sampleHighTempResc[sampleHighTempResc == -1] <- 0

#these commands fit networks on the simulated data
sampleLowTempFit <- IsingFit(sampleLowTempResc)
sampleMidTempFit <- IsingFit(sampleMidTempResc)
sampleHighTempFit <- IsingFit(sampleHighTempResc)

pdf('Figure5.pdf', 7, 10.5)#this creates Figure 5
layout(matrix( c(1, 2, 
                    3, 4,
                    5, 6), 3, 2, byrow = TRUE))

qgraph(sampleHighTempFit $ weiadj, layout = ObamaGraph $ layout, maximum = max(abs(sampleLowTempFit $ weiadj)), 
       cut = quantile(abs(sampleMidTempFit $ weiadj), .75), label.font = 2)
hist(apply(sampleHighTemp, 1, sum), breaks = seq(-11,11,2), include.lowest = FALSE, axes = FALSE, 
     xlab = 'Sum Score', main = '')
axis(1, seq(-10,10,2), seq(-10,10,2), cex.axis = .9)
axis(2)
mtext('High Temperature', line = 2, at = -17, font = 2)

qgraph(sampleMidTempFit $ weiadj, layout = ObamaGraph $ layout, maximum = max(abs(sampleLowTempFit $ weiadj)), 
       cut = quantile(abs(sampleMidTempFit $ weiadj), .75), label.font = 2)
hist(apply(sampleMidTemp, 1, sum), breaks = seq(-11,11,2), include.lowest = FALSE, axes = FALSE, 
      xlab = 'Sum Score', main = '')
axis(1, seq(-10,10,2), seq(-10,10,2), cex.axis = .9)
axis(2)
mtext('Mid Temperature', line = 2, at = -17, font = 2)

qgraph(sampleLowTempFit $ weiadj, layout = ObamaGraph $ layout, maximum = max(abs(sampleLowTempFit $ weiadj)), #the layout argument specifies that all networks are plotted in the layout of the Obama network; the maximum argument makes the different networks comparable as the maximum value is set to the highest edge in the low temperature network
       cut = quantile(abs(sampleMidTempFit $ weiadj), .75), label.font = 2)#the cut argument is used to plot only edges higher than the 75% qunatile of all networks with higher width
hist(apply(sampleLowTemp, 1, sum), breaks = seq(-11,11,2), include.lowest = FALSE, axes = FALSE, 
     xlab = 'Sum Score', main = '')
axis(1, seq(-10,10,2), seq(-10,10,2), cex.axis = .9)
axis(2)
mtext('Low Temperature', line = 2, at = -17, font = 2)

dev.off()

#########################
#Centrality simulation 

set.seed (1)

SampleNeg <- IsingSampler(1000, SimInput$graph, 
                          rep(-.1,10),#this argument specifies the thresholds, which are all set to -.1 
                          responses = c(-1L,1L))
SampleHns <- IsingSampler(1000, SimInput$graph, 
                          c(rep(-.1,5),#this sets the thesholds of the first five nodes to -.1
                            1,#this sets the threshold of the sixth node, which is the node Hns, to 1
                            rep(-.1,4)),#this sets the thesholds of the last four nodes to -.1
                          responses = c(-1L,1L))
SampleAng <- IsingSampler (1000, SimInput$graph, c(rep(-.1,6),#this sets the thesholds of the first six nodes to -.1
                                                   1,#this sets the threshold of the seventh node, which is the node Ang, to 1
                                                   rep(-.1,3)),#this sets the thesholds of the last three nodes to -.1
                           responses = c(-1L,1L))

#calculate the sum scores of the different networks
sumSampleNeg <- apply(SampleNeg, 1, sum)
sumSampleHns <- apply(SampleHns, 1, sum)
sumSampleAng <- apply(SampleAng, 1, sum)

#calculate descriptive statistics of the different networks
mean(sumSampleNeg)
sd(sumSampleNeg)

mean(sumSampleHns)
sd(sumSampleHns)

mean(sumSampleAng)
sd(sumSampleAng)

#perform t-test on the sum scores of the different networks
t.test(sumSampleHns, sumSampleAng, var.equal = TRUE)

#calculate the effect size of the difference in the sum scores
mes(mean(sumSampleHns), mean(sumSampleAng), sd(sumSampleHns), sd(sumSampleAng), 1000, 1000)

#########################
#Supplementary information 

#########################
#Estimating networks based on continious (or categorical) data

ObamaWeiAdj <- EBICglasso(cor_auto(Obama), nrow(Obama))
ObamaqGraph2 <- qgraph(ObamaWeiAdj)
ObamaiGraph2 <- graph_from_adjacency_matrix(abs(ObamaWeiAdj), 'undirected', weighted = TRUE, add.colnames = FALSE)

#########################
#Covariates

gender <- as.numeric(ANES2012$gender_respondent_x)#this saves the gender of the participants to the numeric verctor gender
gender <- gender-1#this rescores the values into 0=male and 1=female
age <- ANES2012$dem_age_r_x#this saves the age of the participants to the numeric verctor age
age[age < 1] <- NA#values lower than 1 indicate missing values

ObamaCov <- data.frame(ObamaCog, ObamaAff, gender, age)
ObamaCov <- na.omit(ObamaCov)
ObamaFitCov <- mgmfit(ObamaCov, c(rep('c', 11),'g'), c(rep(2, 11),1), binary.sign = TRUE)
ObamaFitCov$signs[is.na(ObamaFitCov$signs)] <- 0
ObamaNetCovIn <- ObamaFitCov$wadj*ObamaFitCov$signs
ObamaNetCovOut <- ObamaNetCovIn[1:10,1:10]

#########################
#Edge stability

set.seed(1)

ObamaSta <- bootnet(Obama, 1000, 'IsingFit')

pdf('FigureS1.pdf')
plot(ObamaSta, plot = 'interval', order = 'sample')
dev.off()

#########################
#Centrality Stability

set.seed(1)

ObamaCenSta <- bootnet(Obama, 1000, 'IsingFit', 'person')

pdf('FigureS2.pdf')
plot(ObamaCenSta, subsetRange = c(100,50))
dev.off()

pdf('FigureS3.pdf')
plot(ObamaCenSta, c('betweenness'), perNode = TRUE, subsetRange = c(100,50)) 
dev.off()