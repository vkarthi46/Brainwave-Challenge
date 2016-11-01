setwd("C:\\Users\\vkarth2\\Desktop\\Brainwave-Challenge-master")

# Variable Clustering
library(httr)
set_config( config( ssl_verifypeer = 0L ) )

if(!require(devtools)) install.packages("devtools")
devtools::install_github("kassambara/factoextra")

pacman::p_load(ClustOfVar,data.table,cluster,NbClust)

Train=read.csv("train.csv")[,-c(1,102)]
Test=read.csv("test.csv")[,-1]
data=data.frame(rbind(Train,Test))

tree <- hclustvar(data)
plot(tree)

stab <- stability(tree, B = 40)
plot(stab, main = "Stability of the partitions")
boxplot(stab$matCR, main = "Dispersion of the adjusted Rand index")

plot(stab,nmax=20)
dev.new()
boxplot(stab$matCR[,1:20])

P3 <- cutreevar(tree, 3, matsim = TRUE)
cluster <- P3$cluster
c=names(data)

submission=data.frame(cbind(c,cluster))
colnames(submission)=c("Asset","Cluster")

write.csv(submission,"Sub-Clust.csv")



