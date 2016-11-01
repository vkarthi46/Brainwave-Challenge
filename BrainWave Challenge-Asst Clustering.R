library(httr)
set_config( config( ssl_verifypeer = 0L ) )

if(!require(devtools)) install.packages("devtools")
devtools::install_github("kassambara/factoextra")

pacman::p_load(data.table,cluster,NbClust,factoextra)

Train=read.csv("train.csv")[,-c(1,102)]
Test=read.csv("test.csv")[,-1]
data=data.frame(rbind(Train,Test))


# K-means clustering
set.seed(123)
km.res <- kmeans(data, 5, nstart = 25)
# k-means group number of each observation
km.res$cluster

# Visualize k-means clusters
fviz_cluster(km.res, data = data, geom = "point",
             stand = FALSE, frame.type = "norm")

##PAM clustering
library("cluster")
pam.res <- pam(data, 5)
pam.res$cluster

# Visualize pam clusters
fviz_cluster(pam.res, stand = FALSE, geom = "point",
             frame.type = "norm")

#### Hierarchical clustering
# Compute pairewise distance matrices
dist.res <- dist(data, method = "euclidean")
# Hierarchical clustering results
hc <- hclust(dist.res, method = "complete")
# Visualization of hclust
plot(hc, labels = FALSE, hang = -1)
# Add rectangle around 3 groups
rect.hclust(hc, k = 3, border = 2:4) 

# Cut into 3 groups
hc.cut <- cutree(hc, k = 3)
head(hc.cut, 20)

## 5 Three popular methods for determining the optimal number of clusters
## Elbow method for k-means clustering
set.seed(123)
# Compute and plot wss for k = 2 to k = 15
k.max <- 15 # Maximal number of clusters
data <- data
wss <- sapply(1:k.max, 
              function(k){kmeans(data, k, nstart=10 )$tot.withinss})
plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")
abline(v = 3, lty =2)

fviz_nbclust(data, kmeans, method = "wss") +
  geom_vline(xintercept = 3, linetype = 2)

## Elbow method for PAM clustering
fviz_nbclust(data, pam, method = "wss") +
  geom_vline(xintercept = 3, linetype = 2)

## Elbow method for hierarchical clustering
fviz_nbclust(data, hcut, method = "wss") +
  geom_vline(xintercept = 3, linetype = 2)

### Average silhouette method for k-means clustering
library(cluster)
k.max <- 15
data <-data
sil <- rep(0, k.max)
# Compute the average silhouette width for 
# k = 2 to k = 15
for(i in 2:k.max){
  km.res <- kmeans(data, centers = i, nstart = 25)
  ss <- silhouette(km.res$cluster, dist(data))
  sil[i] <- mean(ss[, 3])
}
# Plot the  average silhouette width
plot(1:k.max, sil, type = "b", pch = 19, 
     frame = FALSE, xlab = "Number of clusters k")
abline(v = which.max(sil), lty = 2)

require(cluster)
fviz_nbclust(data, kmeans, method = "silhouette")

### Average silhouette method for PAM clustering
require(cluster)
fviz_nbclust(iris.scaled, pam, method = "silhouette")

### Average silhouette method for hierarchical clustering
require(cluster)
fviz_nbclust(data, hcut, method = "silhouette",
             hc_method = "complete")


### Gap statistic for k-means clustering
# Compute gap statistic
library(cluster)
set.seed(123)
gap_stat <- clusGap(data, FUN = kmeans, nstart = 25,
                    K.max = 10, B = 50)
# Print the result
print(gap_stat, method = "firstmax")

# Base plot of gap statistic
plot(gap_stat, frame = FALSE, xlab = "Number of clusters k")
abline(v = 3, lty = 2)

# Use factoextra
fviz_gap_stat(gap_stat)

# Print
print(gap_stat, method = "Tibs2001SEmax")
# Plot
fviz_gap_stat(gap_stat, 
              maxSE = list(method = "Tibs2001SEmax"))
# Relaxed the gap test to be within two standard deviations
fviz_gap_stat(gap_stat, 
              maxSE = list(method = "Tibs2001SEmax", SE.factor = 2))

## NbClust Package providing 30 indices for determining the best number of clusters

library("NbClust")
set.seed(123)
res.nb <- NbClust(data, distance = "euclidean",
                  min.nc = 2, max.nc = 10, 
                  method = "complete", index ="gap") 
res.nb # print the results

# All gap statistic values
res.nb$All.index
# Best number of clusters
res.nb$Best.nc
# Best partition
res.nb$Best.partition

nb <- NbClust(data, distance = "euclidean", min.nc = 2,
              max.nc = 10, method = "complete", index ="all")
# Print the result
nb

fviz_nbclust(nb) + theme_minimal()

