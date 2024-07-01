###########################
# MANIPULATIONS BY AUTHOR
# 1. NEED TO DOWNLOAD OWN VERSION OF THE DATA AND CHANGE VARIABLE NAMES TO FIT
# 2. 
#
##########################


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
#library(compute.es)
#library(NetworkComparisonTest)
#library(mgm)
#library(Matrix)

#########################
#Load data 
ANES2012 <- read.dta('../ICPSR_35157/DS0001/35157-0001-Data.dta')#loads the data to the object ANES2012

#########################
#Recode variables 
#Items regarding Obama
ObamaCog <- data.frame(Mor = as.numeric(ANES2012$CTRAIT_DPCMORAL),#this creates a data frame containing the items tapping beliefs
                       Led = as.numeric(ANES2012$CTRAIT_DPCLEAD),
                       Car = as.numeric(ANES2012$CTRAIT_DPCCARE),
                       Kno = as.numeric(ANES2012$CTRAIT_DPCKNOW),
                       Int = as.numeric(ANES2012$CTRAIT_DPCINT),
                       Hns = as.numeric(ANES2012$CTRAIT_DPCHONST))

ObamaCog[ObamaCog < 3] <- NA#values below 3 represent missing values
ObamaCog <- binarize(ObamaCog, 5, removeNArows = FALSE)#this binarizes the data (this is done because the model we use for simulating networks assumes binary data); (not) endorsing the beliefs is encoded as 1 (0) 

ObamaAff <- data.frame(Ang = as.numeric(ANES2012$CANDAFF_ANGDPC),#this creates a data frame containing the items tapping feelings
                       Hop = as.numeric(ANES2012$CANDAFF_HPDPC), 
                       Afr = as.numeric(ANES2012$CANDAFF_AFRDPC), 
                       Prd = as.numeric(ANES2012$CANDAFF_PRDDPC))

ObamaAff[ObamaAff < 3] <- NA#values below 3 represent missing values
ObamaAff <- binarize(ObamaAff, 4, removeNArows = FALSE)#(not) endorsing the feelings is encoded as 1 (0)

Obama <- data.frame(ObamaCog,ObamaAff)#this creates a data frame containing all items tapping evaluative reactions
Obama <- na.omit(Obama)#this deletes missing values casewise


#########################
#Network estimation

ObamaFit <- IsingFit(Obama)
ObamaGraph <- qgraph(ObamaFit $ weiadj, layout = 'spring', cut = .8)
#ObamaiGraph <- graph_from_adjacency_matrix(abs(ObamaFit $ weiadj), 'undirected', weighted = TRUE, add.colnames = FALSE)

#########################
#Centrality

ObamaCen <- centralityTable(ObamaGraph, standardized = FALSE)

#########################
#Simulation input 

SimInput <- LinTransform(ObamaFit$weiadj, ObamaFit$thresholds)#The LinTransform function rescales the edge weights and thresholds, so that a threshold of 0 indicates that a node has no disposition to be in a given state and a positive (negative) threshold indicates that a node has the disposition to be "on" ("off")
#SimInput$graph[c(7,9),-c(7,9)] <- SimInput$graph[c(7,9),-c(7,9)]*-1#this command and the next command rescale the nodes Ang and Afr
#SimInput$graph[-c(7,9),c(7,9)] <- SimInput$graph[-c(7,9),c(7,9)]*-1

#########################
#Centrality simulation 

set.seed (1)

SampleNeg <- IsingSampler(1000, SimInput$graph, 
                          SimInput$thresholds,#this argument specifies the thresholds, which are all set to -.1 
                          responses = c(-1L,1L))
SampleHns <- IsingSampler(1000, SimInput$graph,
                          c(SimInput$thresholds[1:5],1,SimInput$thresholds[7:10]),
                          responses = c(-1L,1L))
SampleAng <- IsingSampler (1000, SimInput$graph, 
                           c(SimInput$thresholds[1:6],1,SimInput$thresholds[8:10]),
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

###################
#AUTHOR ADDITION
##################
#DO REST OF THE NODES
SampleMor <- IsingSampler(1000, SimInput$graph, 
                          c(1,SimInput$thresholds[2:10]),
                          responses = c(-1L,1L))

SampleLed <- IsingSampler(1000, SimInput$graph, 
                          c(SimInput$thresholds[1],1,SimInput$thresholds[3:10]),
                          responses = c(-1L,1L))

SampleCar <- IsingSampler(1000, SimInput$graph, 
                          c(SimInput$thresholds[1:2],1,SimInput$thresholds[4:10]),
                          responses = c(-1L,1L))

SampleKno <- IsingSampler(1000, SimInput$graph, 
                          c(SimInput$thresholds[1:3],1,SimInput$thresholds[5:10]),
                        responses = c(-1L,1L))

SampleInt <- IsingSampler(1000, SimInput$graph, 
                          c(SimInput$thresholds[1:4],1,SimInput$thresholds[6:10]),
                        responses = c(-1L,1L))

SampleHop <- IsingSampler(1000, SimInput$graph, 
                          c(SimInput$thresholds[1:7],1,SimInput$thresholds[9:10]),
                        responses = c(-1L,1L))

SampleAfr <- IsingSampler(1000, SimInput$graph, 
                          c(SimInput$thresholds[1:8],1,SimInput$thresholds[10]),
                        responses = c(-1L,1L))

SamplePrd <- IsingSampler(1000, SimInput$graph, 
                          c(SimInput$thresholds[1:9],1),
                        responses = c(-1L,1L))

#MORE SUMS
sumSampleMor <- apply(SampleMor, 1, sum)
sumSampleLed <- apply(SampleLed, 1, sum)
sumSampleCar <- apply(SampleCar, 1, sum)
sumSampleKno <- apply(SampleKno, 1, sum)
sumSampleInt <- apply(SampleInt, 1, sum)
sumSampleHop <- apply(SampleHop, 1, sum)
sumSampleAfr <- apply(SampleAfr, 1, sum)
sumSamplePrd <- apply(SamplePrd, 1, sum)

###ADD WASS DIST.
f_wass_dist <- function(dbase,dtest){
  x_prob <- dbase/sum(dbase)
  y_prob <- dtest/sum(dtest)
  return( sum(abs(cumsum(x_prob)-cumsum(y_prob))) )
}

f_wass_dist(c(10,rep(0,20)),c(rep(0,20),10))
par(mar = c(3, 4, 1, 2))

png("VertexPertDistributions2017a_DalegeStudy_NoRevCoding_EstThresh.png",units="in", width=10, height=10, res=300)
par(mfrow = c(4,4))
hist(sumSampleNeg,xlab="Sum Score",main='Baseline',ylim=c(0,700))
text(0,500,f_wass_dist(table(sumSampleNeg),table(sumSampleNeg)))
text(0,200,mean(sumSampleNeg))
hist(sumSampleMor,xlab="Sum Score",main='Is Moral',ylim=c(0,700))
text(0,500,f_wass_dist(table(sumSampleNeg),table(sumSampleMor)))
text(0,200,mean(sumSampleMor))
hist(sumSampleLed,xlab="Sum Score",main='Strong Leadership',ylim=c(0,700))
text(0,500,f_wass_dist(table(sumSampleNeg),table(sumSampleLed)))
text(0,200,mean(sumSampleLed))
hist(sumSampleCar,xlab="Sum Score",main='Is Caring',ylim=c(0,700))
text(0,500,f_wass_dist(table(sumSampleNeg),table(sumSampleCar)))
text(0,200,mean(sumSampleCar))
hist(sumSampleKno,xlab="Sum Score",main='Is Knowledgeable',ylim=c(0,700))
text(0,500,f_wass_dist(table(sumSampleNeg),table(sumSampleKno)))
text(0,200,mean(sumSampleKno))
hist(sumSampleInt,xlab="Sum Score",main='Is Intelligent',ylim=c(0,700))
text(0,500,f_wass_dist(table(sumSampleNeg),table(sumSampleInt)))
text(0,200,mean(sumSampleInt))
hist(sumSampleHns,xlab="Sum Score",main='Is Honest',ylim=c(0,700))
text(0,500,f_wass_dist(table(sumSampleNeg),table(sumSampleHns)))
text(0,200,mean(sumSampleHns))
hist(sumSampleAng,xlab="Sum Score",main='Angry',ylim=c(0,700))
text(0,500,f_wass_dist(table(sumSampleNeg),table(sumSampleAng)))
text(0,200,mean(sumSampleAng))
hist(sumSampleHop,xlab="Sum Score",main='Hopeful',ylim=c(0,700))
text(0,500,f_wass_dist(table(sumSampleNeg),table(sumSampleHop)))
text(0,200,mean(sumSampleHop))
hist(sumSampleAfr,xlab="Sum Score",main='Afraid',ylim=c(0,700))
text(0,500,f_wass_dist(table(sumSampleNeg),table(sumSampleAfr)))
text(0,200,mean(sumSampleAfr))
hist(sumSamplePrd,xlab="Sum Score",main='Proud',ylim=c(0,700))
text(0,500,f_wass_dist(table(sumSampleNeg),table(sumSamplePrd)))
text(0,200,mean(sumSamplePrd))
dev.off()

##ADD CENTRALITY STRENGTH
ObamaStrength.W.Mean <- ObamaCen[21:30,]
ObamaStrength.W.Mean$MeanSS <- NA
ObamaStrength.W.Mean$MeanSS[1] <- mean(sumSampleMor)
ObamaStrength.W.Mean$MeanSS[2] <- mean(sumSampleLed)
ObamaStrength.W.Mean$MeanSS[3] <- mean(sumSampleCar)
ObamaStrength.W.Mean$MeanSS[4] <- mean(sumSampleKno)
ObamaStrength.W.Mean$MeanSS[5] <- mean(sumSampleInt)
ObamaStrength.W.Mean$MeanSS[6] <- mean(sumSampleHns)
ObamaStrength.W.Mean$MeanSS[7] <- mean(sumSampleAng)
ObamaStrength.W.Mean$MeanSS[8] <- mean(sumSampleHop)
ObamaStrength.W.Mean$MeanSS[9] <- mean(sumSampleAfr)
ObamaStrength.W.Mean$MeanSS[10] <- mean(sumSamplePrd)
ObamaStrength.W.Mean$Wass <- NA
ObamaStrength.W.Mean$Wass[1] <- f_wass_dist(table(sumSampleNeg),table(sumSampleMor))
ObamaStrength.W.Mean$Wass[2] <- f_wass_dist(table(sumSampleNeg),table(sumSampleLed))
ObamaStrength.W.Mean$Wass[3] <- f_wass_dist(table(sumSampleNeg),table(sumSampleCar))
ObamaStrength.W.Mean$Wass[4] <- f_wass_dist(table(sumSampleNeg),table(sumSampleKno))
ObamaStrength.W.Mean$Wass[5] <- f_wass_dist(table(sumSampleNeg),table(sumSampleInt))
ObamaStrength.W.Mean$Wass[6] <- f_wass_dist(table(sumSampleNeg),table(sumSampleHns))
ObamaStrength.W.Mean$Wass[7] <- f_wass_dist(table(sumSampleNeg),table(sumSampleAng))
ObamaStrength.W.Mean$Wass[8] <- f_wass_dist(table(sumSampleNeg),table(sumSampleHop))
ObamaStrength.W.Mean$Wass[9] <- f_wass_dist(table(sumSampleNeg),table(sumSampleAfr))
ObamaStrength.W.Mean$Wass[10] <- f_wass_dist(table(sumSampleNeg),table(sumSamplePrd))


##ADD CENTRALITY BETWEEN
ObamaBetween.W.Mean <- ObamaCen[1:10,]
ObamaBetween.W.Mean$MeanSS <- NA
ObamaBetween.W.Mean$MeanSS[1] <- mean(sumSampleMor)
ObamaBetween.W.Mean$MeanSS[2] <- mean(sumSampleLed)
ObamaBetween.W.Mean$MeanSS[3] <- mean(sumSampleCar)
ObamaBetween.W.Mean$MeanSS[4] <- mean(sumSampleKno)
ObamaBetween.W.Mean$MeanSS[5] <- mean(sumSampleInt)
ObamaBetween.W.Mean$MeanSS[6] <- mean(sumSampleHns)
ObamaBetween.W.Mean$MeanSS[7] <- mean(sumSampleAng)
ObamaBetween.W.Mean$MeanSS[8] <- mean(sumSampleHop)
ObamaBetween.W.Mean$MeanSS[9] <- mean(sumSampleAfr)
ObamaBetween.W.Mean$MeanSS[10] <- mean(sumSamplePrd)
ObamaBetween.W.Mean$Wass <- NA
ObamaBetween.W.Mean$Wass[1] <- f_wass_dist(table(sumSampleNeg),table(sumSampleMor))
ObamaBetween.W.Mean$Wass[2] <- f_wass_dist(table(sumSampleNeg),table(sumSampleLed))
ObamaBetween.W.Mean$Wass[3] <- f_wass_dist(table(sumSampleNeg),table(sumSampleCar))
ObamaBetween.W.Mean$Wass[4] <- f_wass_dist(table(sumSampleNeg),table(sumSampleKno))
ObamaBetween.W.Mean$Wass[5] <- f_wass_dist(table(sumSampleNeg),table(sumSampleInt))
ObamaBetween.W.Mean$Wass[6] <- f_wass_dist(table(sumSampleNeg),table(sumSampleHns))
ObamaBetween.W.Mean$Wass[7] <- f_wass_dist(table(sumSampleNeg),table(sumSampleAng))
ObamaBetween.W.Mean$Wass[8] <- f_wass_dist(table(sumSampleNeg),table(sumSampleHop))
ObamaBetween.W.Mean$Wass[9] <- f_wass_dist(table(sumSampleNeg),table(sumSampleAfr))
ObamaBetween.W.Mean$Wass[10] <- f_wass_dist(table(sumSampleNeg),table(sumSamplePrd))

##ADD CENTRALITY CLOSENESS
ObamaClose.W.Mean <- ObamaCen[11:20,]
ObamaClose.W.Mean$MeanSS <- NA
ObamaClose.W.Mean$MeanSS[1] <- mean(sumSampleMor)
ObamaClose.W.Mean$MeanSS[2] <- mean(sumSampleLed)
ObamaClose.W.Mean$MeanSS[3] <- mean(sumSampleCar)
ObamaClose.W.Mean$MeanSS[4] <- mean(sumSampleKno)
ObamaClose.W.Mean$MeanSS[5] <- mean(sumSampleInt)
ObamaClose.W.Mean$MeanSS[6] <- mean(sumSampleHns)
ObamaClose.W.Mean$MeanSS[7] <- mean(sumSampleAng)
ObamaClose.W.Mean$MeanSS[8] <- mean(sumSampleHop)
ObamaClose.W.Mean$MeanSS[9] <- mean(sumSampleAfr)
ObamaClose.W.Mean$MeanSS[10] <- mean(sumSamplePrd)
ObamaClose.W.Mean$Wass <- NA
ObamaClose.W.Mean$Wass[1] <- f_wass_dist(table(sumSampleNeg),table(sumSampleMor))
ObamaClose.W.Mean$Wass[2] <- f_wass_dist(table(sumSampleNeg),table(sumSampleLed))
ObamaClose.W.Mean$Wass[3] <- f_wass_dist(table(sumSampleNeg),table(sumSampleCar))
ObamaClose.W.Mean$Wass[4] <- f_wass_dist(table(sumSampleNeg),table(sumSampleKno))
ObamaClose.W.Mean$Wass[5] <- f_wass_dist(table(sumSampleNeg),table(sumSampleInt))
ObamaClose.W.Mean$Wass[6] <- f_wass_dist(table(sumSampleNeg),table(sumSampleHns))
ObamaClose.W.Mean$Wass[7] <- f_wass_dist(table(sumSampleNeg),table(sumSampleAng))
ObamaClose.W.Mean$Wass[8] <- f_wass_dist(table(sumSampleNeg),table(sumSampleHop))
ObamaClose.W.Mean$Wass[9] <- f_wass_dist(table(sumSampleNeg),table(sumSampleAfr))
ObamaClose.W.Mean$Wass[10] <- f_wass_dist(table(sumSampleNeg),table(sumSamplePrd))

png("Wass&MeanSumScore2017a_DalegeStudy.png",units="in", width=9, height=5, res=300)
par(mfrow=c(2,3))
plot(ObamaStrength.W.Mean$value,ObamaStrength.W.Mean$MeanSS,ylim=c(-10,10),xlab="Vertex Strength",ylab="Mean Sum Score",pch=c(1,3,1,1,1,3,1,1,1,1))
plot(ObamaBetween.W.Mean$value,ObamaBetween.W.Mean$MeanSS,ylim=c(-10,10),xlab="Vertex Betweenness",ylab="Mean Sum Score",pch=c(1,3,1,1,1,3,1,1,1,1))
plot(ObamaClose.W.Mean$value,ObamaClose.W.Mean$MeanSS,ylim=c(-10,10),xlab="Vertex Closeness",ylab="Mean Sum Score",pch=c(1,3,1,1,1,3,1,1,1,1))
plot(ObamaStrength.W.Mean$value,ObamaStrength.W.Mean$Wass,ylim=c(0,20),xlab="Vertex Strength",ylab="Wass. Distance",pch=c(1,3,1,1,1,3,1,1,1,1))
plot(ObamaBetween.W.Mean$value,ObamaBetween.W.Mean$Wass,ylim=c(0,20),xlab="Vertex Betweenness",ylab="Wass. Distance",pch=c(1,3,1,1,1,3,1,1,1,1))
plot(ObamaClose.W.Mean$value,ObamaClose.W.Mean$Wass,ylim=c(0,20),xlab="Vertex Closeness",ylab="Wass. Distance",pch=c(1,3,1,1,1,3,1,1,1,1))
dev.off()








##################
#END AUTHOR ADDITION
#################


#EOF