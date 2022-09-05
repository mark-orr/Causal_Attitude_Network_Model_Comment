#########################
#Packages

library (foreign)
library (qgraph)
library (IsingFit)
library (igraph)


#########################
#Data
#The data set can be assessed through this link: http://goo.gl/JtukQv

ANES1984 <- read.dta ('NES1984.dta')

#########################
#Variables recoding

# V840319: Reagan hard-working
Reagan1984Hard <- as.vector(ANES1984$V840319)

# V840320: Reagan decent
Reagan1984Decent <- as.vector(ANES1984$V840320)

# V840321: Reagan compassionate
Reagan1984Comp <- as.vector(ANES1984$V840321)

# V840322: Reagan commands respect
Reagan1984Respect <- as.vector(ANES1984$V840322)

# V840323: Reagan intelligent
Reagan1984Int <- as.vector(ANES1984$V840323)

# V840324: Reagan moral
Reagan1984Moral <- as.vector(ANES1984$V840324)

# V840325: Reagan kind
Reagan1984Kind <- as.vector(ANES1984$V840325)

# V840326: Reagan inspiring
Reagan1984Insp <- as.vector(ANES1984$V840326)

# V840327: Reagan knowledgeable
Reagan1984Know <- as.vector(ANES1984$V840327)

# V840328: Reagan good-Example
Reagan1984Example <- as.vector(ANES1984$V840328)

# V840329: Reagan Cares
Reagan1984Cares <- as.vector(ANES1984$V840329)

# V840330: Reagan leadership
Reagan1984Lead <- as.vector(ANES1984$V840330)

# V840331: Reagan understands
Reagan1984Und <- as.vector(ANES1984$V840331)

# V840332: Reagan fair
Reagan1984Fair <- as.vector(ANES1984$V840332)

# V840333: Reagan in touch
Reagan1984Touch <- as.vector(ANES1984$V840333)

# V840334: Reagan religious
Reagan1984Rel <- as.vector(ANES1984$V840334)

Reagan1984Cog <- data.frame (Reagan1984Hard, Reagan1984Decent, Reagan1984Comp, Reagan1984Respect, Reagan1984Int, Reagan1984Moral, Reagan1984Kind, Reagan1984Insp, 
                              Reagan1984Know, Reagan1984Example, Reagan1984Cares, Reagan1984Lead, Reagan1984Und, Reagan1984Fair, Reagan1984Touch, Reagan1984Rel, stringsAsFactors = FALSE)
Reagan1984Cog [Reagan1984Cog == "1. EXTREMELY WELL" ] <- 1
Reagan1984Cog [Reagan1984Cog == "2. QUITE WELL" ] <- 1
Reagan1984Cog [Reagan1984Cog == "3. NOT TOO WELL" ] <- 0
Reagan1984Cog [Reagan1984Cog == "4. NOT WELL AT ALL"] <- 0
Reagan1984Cog [Reagan1984Cog == "8. DON-T KNOW"] <- NA
Reagan1984Cog [Reagan1984Cog == "9. NOT ASCERTAINED"] <- NA
Reagan1984Cog <- data.matrix (Reagan1984Cog)
Reagan1984Cog <- as.data.frame (Reagan1984Cog)

names(Reagan1984Cog) <- c('R00.Hard', 
                          'R01.Decent',
                          'R02.Comp',
                          'R03.Respect',
                          'R04.Int',
                          'R05.Moral',
                          'R06.Kind', 
                          'R07.Insp',
                          'R08.Know',
                          'R09.Example',
                          'R10.Cares',
                          'R11.Lead',
                          'R12.Und',
                          'R13.Fair',
                          'R14.Touch',
                          'R15.Rel')



# V840335: Mondale hard-working
Mondale1984Hard <- as.vector(ANES1984$V840335)

# V840336: Mondale decent
Mondale1984Decent <- as.vector(ANES1984$V840336)

# V840337: Mondale compassionate
Mondale1984Comp <- as.vector(ANES1984$V840337)

# V840338: Mondale commands respect
Mondale1984Respect <- as.vector(ANES1984$V840338)

# V840339: Mondale intelligent
Mondale1984Int <- as.vector(ANES1984$V840339)

# V840340: Mondale moral
Mondale1984Moral <- as.vector(ANES1984$V840340)

# V840341: Mondale kind
Mondale1984Kind <- as.vector(ANES1984$V840341)

# V840342: Mondale inspiring
Mondale1984Insp <- as.vector(ANES1984$V840342)

# V840343: Mondale knowledgeable
Mondale1984Know <- as.vector(ANES1984$V840343)

# V840344: Mondale good-Example
Mondale1984Example <- as.vector(ANES1984$V840344)

# V840345: Mondale Cares
Mondale1984Cares <- as.vector(ANES1984$V840345)

# V840346: Mondale leadership
Mondale1984Lead <- as.vector(ANES1984$V840346)

# V840347: Mondale understands
Mondale1984Und <- as.vector(ANES1984$V840347)

# V840348: Mondale fair
Mondale1984Fair <- as.vector(ANES1984$V840348)

# V840349: Mondale in touch
Mondale1984Touch <- as.vector(ANES1984$V840349)

# V840350: Mondale religious
Mondale1984Rel <- as.vector(ANES1984$V840350)

Mondale1984Cog <- data.frame (Mondale1984Hard, Mondale1984Decent, Mondale1984Comp, Mondale1984Respect, Mondale1984Int, Mondale1984Moral, Mondale1984Kind, Mondale1984Insp, 
                               Mondale1984Know, Mondale1984Example, Mondale1984Cares, Mondale1984Lead, Mondale1984Und, Mondale1984Fair, Mondale1984Touch, Mondale1984Rel, stringsAsFactors = FALSE)
Mondale1984Cog [Mondale1984Cog == "1. EXTREMELY WELL" ] <- 1
Mondale1984Cog [Mondale1984Cog == "2. QUITE WELL" ] <- 1
Mondale1984Cog [Mondale1984Cog == "3. NOT TOO WELL" ] <- 0
Mondale1984Cog [Mondale1984Cog == "4. NOT WELL AT ALL"] <- 0
Mondale1984Cog [Mondale1984Cog == "8. DON-T KNOW"] <- NA
Mondale1984Cog [Mondale1984Cog == "9. NOT ASCERTAINED"] <- NA
Mondale1984Cog <- data.matrix (Mondale1984Cog)
Mondale1984Cog <- as.data.frame (Mondale1984Cog)


# V840212: Reagan angry
Reagan1984Angry <- as.vector (ANES1984$V840212)

# V840213: Reagan hopeful
Reagan1984Hope <- as.vector (ANES1984$V840213)

# V840214: Reagan Afraid
Reagan1984Afr <- as.vector (ANES1984$V840214)

# V840215: Reagan proud
Reagan1984Prou <- as.vector (ANES1984$V840215)

# V840216: Reagan disgusted
Reagan1984Disg <- as.vector (ANES1984$V840216)

# V840217: Reagan sympathetic
Reagan1984Symp <- as.vector (ANES1984$V840217)

# V840218: Reagan uneasy
Reagan1984Uneasy <- as.vector (ANES1984$V840218)

Reagan1984Aff <- data.frame(Reagan1984Angry, Reagan1984Hope, Reagan1984Afr, Reagan1984Prou, Reagan1984Disg, Reagan1984Symp, Reagan1984Uneasy, stringsAsFactors = FALSE)
Reagan1984Aff [Reagan1984Aff == "1. YES; HAVE FELT"] <- 1
Reagan1984Aff [Reagan1984Aff == "5. NO; NEVER FELT" ] <- 0
Reagan1984Aff [Reagan1984Aff == "9. NA; DK"] <- NA
Reagan1984Aff <- data.matrix (Reagan1984Aff)
Reagan1984Aff <- as.data.frame (Reagan1984Aff)

names(Reagan1984Aff) <- c('R16.Angry',
                          'R17.Hope',
                          'R18.Afr',
                          'R19.Prou',
                          'R20.Disg', 
                          'R21.Symp',
                          'R22.Uneasy')

# V840219: Mondale angry
Mondale1984Angry <- as.vector (ANES1984$V840219)

# V840220: Mondale hopeful
Mondale1984Hope <- as.vector (ANES1984$V840220)

# V840221: Mondale Afraid
Mondale1984Afr <- as.vector (ANES1984$V840221)

# V840222: Mondale proud
Mondale1984Prou <- as.vector (ANES1984$V840222)

# V840223: Mondale disgusted
Mondale1984Disg <- as.vector (ANES1984$V840223)

# V840224: Mondale sympathetic
Mondale1984Symp <- as.vector (ANES1984$V840224)

# V840225: Mondale uneasy
Mondale1984Uneasy <- as.vector (ANES1984$V840225)

Mondale1984Aff <- data.frame(Mondale1984Angry, Mondale1984Hope, Mondale1984Afr, Mondale1984Prou, Mondale1984Disg, Mondale1984Symp, Mondale1984Uneasy, stringsAsFactors = FALSE)
Mondale1984Aff [Mondale1984Aff == "1. YES; HAVE FELT"] <- 1
Mondale1984Aff [Mondale1984Aff == "5. NO; NEVER FELT" ] <- 0
Mondale1984Aff [Mondale1984Aff == "9. NA; DK"] <- NA
Mondale1984Aff <- data.matrix (Mondale1984Aff)
Mondale1984Aff <- as.data.frame (Mondale1984Aff)

Reagan1984 <- data.frame (Reagan1984Cog [,1:15], Reagan1984Aff)
Mondale1984 <- data.frame (Mondale1984Cog [,1:15], Mondale1984Aff)

#########################
#Network estimation

Reagan1984Fit <- IsingFit (na.omit (Reagan1984))
Mondale1984Fit <- IsingFit (na.omit (Mondale1984))

#########################
#Small-world analyses

SW_Index <- function (Graph, ci = c (.1, .05, .01, .001))
{
  randomC <- vector (, 1000)
  randomL <- vector (, 1000)
  for (i in 1:1000)
  {
    Rgraph <- erdos.renyi.game (vcount (Graph), ecount (Graph), 'gnm')
    randomC [i] <- transitivity (Rgraph, 'average')
    randomL [i] <- average.path.length(Rgraph)
  }
  MrandomC <- mean (randomC)
  MrandomL <- mean (randomL)
  Clustering.Graph = transitivity (Graph, 'average')
  ASPL.Graph = average.path.length (Graph)
  Index <- (Clustering.Graph / MrandomC) / (ASPL.Graph / MrandomL)
  
  sm_sample <- vector (, 1000)
  for (i in 1:1000)
  {
    Rgraph <- erdos.renyi.game (vcount (Graph), ecount (Graph), 'gnm')
    sm_sample [i] <- (transitivity (Rgraph, 'average') / MrandomC) /(average.path.length(Rgraph) / MrandomL)
  }
  CI <- as.vector (((quantile (sm_sample, 1 - (ci / 2)) - quantile (sm_sample, ci / 2)) / 2) + 1)
  return (list (SW.Index = Index, Upper.CI = data.frame (CI = ci, Value.CI = CI), 
                Clustering.Graph = Clustering.Graph, Clustering.Random.Graph = MrandomC,
                ASPL.Graph = ASPL.Graph, ASPL.Random.Graph = MrandomL))
}

Reagan1984iGraph <- graph.adjacency (Reagan1984Fit $ weiadj, "undirected", diag = FALSE, weighted = TRUE)
Mondale1984iGraph <- graph.adjacency (Mondale1984Fit $ weiadj, "undirected", diag = FALSE, weighted = TRUE)



SW_Index (Reagan1984iGraph)
SW_Index (Mondale1984iGraph)

#########################
#Centrality Analyses

centrality.Reagan <- centrality (qgraph (Reagan1984Fit $ weiadj))
centrality.Mondale <- centrality (qgraph (Mondale1984Fit $ weiadj))

closeness.Reagan <- centrality.Reagan $ Closeness
degree.Reagan <- centrality.Reagan $ OutDegree
betweenness.Reagan <- centrality.Reagan $ Betweenness

nodes.can <- cbind(closeness.Reagan, degree.Reagan, betweenness.Reagan)
write.table(nodes.can, file="nodes_can.txt", 
            row.names=TRUE, col.names=TRUE)

closeness.Mondale <- centrality.Mondale $ Closeness
degree.Mondale <- centrality.Mondale $ OutDegree
betweenness.Mondale <- centrality.Mondale $ Betweenness

#########################
#Plotting

pdf ('Reagan&Mondale1984.pdf', 6, 3)
par (mfrow = c (1, 2))

qgraph (Reagan1984Fit $ weiadj, layout = 'spring', cut = .4, groups = list (1:15, seq (16, 22, 2), seq (17, 21, 2)), 
        color = c ('red', 'green', 'lightblue'), label.scale = FALSE, label.cex = .36,
        labels = c ('hard-Working', 'decent', 'compassionate', 'respect', 'intelligent', 'moral', 'kind', 'inspiring', 'knowledgeable',
                    'good example', 'cares', 'leadership', 'understands', 'fair', 'in touch', 'angry', 'hope', 'afraid', 'proud', 
                    'disgusted', 'sympathetic', 'uneasy'))
text (0, 1.2, 'Ronald Reagan', cex = .65, font = 2)
qgraph (Mondale1984Fit $ weiadj, layout = 'spring', cut = .4, groups = list (1:15, seq (16, 22, 2), seq (17, 21, 2)), 
        color = c ('red', 'green', 'lightblue'), label.scale = FALSE, label.cex = .36,
        labels = c ('hard-Working', 'decent', 'compassionate', 'respect', 'intelligent', 'moral', 'kind', 'inspiring', 'knowledgeable',
                    'good example', 'cares', 'leadership', 'understands', 'fair', 'in touch', 'angry', 'hope', 'afraid', 'proud', 
                    'disgusted', 'sympathetic', 'uneasy'))
text (0, 1.2, 'Walter Mondale', cex = .65, font = 2)

dev.off ()


