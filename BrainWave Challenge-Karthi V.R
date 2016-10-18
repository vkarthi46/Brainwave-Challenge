# Working directory
options(gsubfn.engine = "R")
setwd("C:\\Users\\karthiv\\Desktop\\Other files\\Competition\\BrainWave\\Predict the growth")
# Loading Required Packages
pacman::p_load(ROCR,DiagrammeR,data.table,dplyr,ggplot2,sqldf,gridExtra,RSQLite,caTools,
               randomForest,xgboost,caret,car,stringr,corrplot,Rtsne,knitr)
knitr::opts_chunk$set(cache=TRUE)

# Reading train and test data
Train=read.csv("train.csv")
Test=read.csv("test.csv")

# Building Predictive Models
# Train Pre-Processing
# Check for features's variance
# check for zero variance
zero.var = nearZeroVar(Train[,-c(1,102)], saveMetrics=TRUE)
zero.var

# Plot of relationship between features and outcome
# target outcome (label)
outcome.org = as.factor(Train[, "Y"])
outcome = outcome.org 
levels(outcome)
# convert character levels to numeric
num.class = length(levels(outcome))
levels(outcome) = 1:num.class
head(outcome)

x1=Train[,-c(1,102)]
featurePlot(x1, outcome.org , "strip")

#Plot of correlation matrix
corrplot.mixed(cor(x1), lower="circle", upper="color", 
               tl.pos="lt", diag="n", order="hclust", hclust.method="complete")

embedding = as.data.frame(tsne$Y)
embedding$Class = outcome.org
g = ggplot(embedding, aes(x=V1, y=V2, color=Class)) +
  geom_point(size=1.25) +
  guides(colour=guide_legend(override.aes=list(size=6))) +
  xlab("") + ylab("") +
  ggtitle("t-SNE 2D Embedding of 'Classe' Outcome") +
  theme_light(base_size=20) +
  theme(axis.text.x=element_blank(),
        axis.text.y=element_blank())
print(g)

# Missing value Imputation

Missing_summary=data.frame(sapply(Train,function(x) {sum(is.na(x))}),sapply(Train, class))
colnames(Missing_summary)=c("Count","data Type")
Missing_summary

## Variable Type
split(names(Train),sapply(Train, function(x) paste(class(x), collapse=" ")))

## Model Building
# Convert into Numeric
Train$Y=as.numeric(ifelse(Train$Y == 1, 1,0))

# Principal component Analysis
x_train<-sapply(Train[,-102],function(x){as.numeric(x)});

#principal component analysis
prin_comp <- prcomp(x_train, scale. = T)

# rotation
prin_comp$rotation
prin_comp$rotation[1:5,1:4]

#compute standard deviation of each principal component
std_dev <- prin_comp$sdev

#compute variance
pr_var <- std_dev^2

#check variance of first 10 components
sum(pr_var[1:10])

#proportion of variance explained
prop_varex <- pr_var/sum(pr_var)
sum(prop_varex[1:10])

#scree plot
plot(prop_varex, xlab = "Principal Component",
       ylab = "Proportion of Variance Explained",
       type = "b")

#cumulative scree plot
plot(cumsum(prop_varex), xlab = "Principal Component",
       ylab = "Cumulative Proportion of Variance Explained",
       type = "b")

#add a training set with principal components
train.data <- data.frame(Y = Train$Y, prin_comp$x)

#we are interested in first 10 PCAs
train.data <- train.data[,c(1:11)]

#transform test into PCA
test.data <- predict(prin_comp, newdata = Test)
test.data <- as.data.frame(test.data)

#select the first 10 components
test.data <- data.frame(test.data[,1:10])

#Preparing data for Xgboost
X_train=train.data[,-1]
X_test=test.data

# extracting target
X_tar=train.data[,1];
X_target <- as.numeric(X_tar)

seed <- 235
set.seed(seed)

## xgboost - 10 fold cross validation
model_xgb_cv <- xgb.cv(data=as.matrix(X_train),label=as.matrix(X_target),num_class = num.class,
                       nfold=10, objective="binary:logistic", nrounds=300, eta=0.2, 
                       max_depth=15, subsample=0.7, colsample_bytree=0.8, min_child_weight=1, 
                       eval_metric="auc")
bestRound = which.max(as.matrix(model_xgb_cv)[,3]-as.matrix(model_xgb_cv)[,4])
bestRound


# index of Maximum auc
tail(model_xgb_cv$dt) 
min.auc.idx = which.max(model_xgb_cv$dt[, test.auc.mean]) 
min.auc.idx 

# Maximum auc
model_xgb_cv$dt[min.auc.idx ,]

# get CV's prediction decoding
pred.cv = matrix(model_xgb_cv$pred, nrow=3000, ncol=1)


# confusion matrix
confusionMatrix(factor(X_target), factor(pred.cv))


model_xgb <- xgboost(data=as.matrix(X_train), label=as.matrix(X_target), 
                     objective="binary:logistic", nrounds=250, eta=0.005, max_depth=6, 
                     subsample=0.75, colsample_bytree=0.8, min_child_weight=1, 
                     eval_metric="auc")


## prediction
pred <- predict(model_xgb, as.matrix(X_test),ntreelimit = bestRound))

model <- xgb.dump(model_xgb, with.stats = T)
model[1:10] #This statement prints top 10 nodes of the model

# Get the feature real names
names <- dimnames(data.matrix(X_train[,-1]))[[2]]

# Compute feature importance matrix
names = dimnames(X_train)[[2]]
importance_matrix <- xgb.importance(names, model = model_xgb)

# Nice graph
xgb.plot.importance(importance_matrix)

# Prediction decoding

perf <- performance(pred,measure = &quot;tpr&quot;, x.measure = &quot;fpr&quot;)
plot(perf, colorize=T)

pred &lt;- prediction(pred,ROCR.simple$labels)
class(pred)

summary(pred)


rorder=rank(pred,ties.method = "first")
threshold=0.50
ntop=length(rorder)-as.integer(threshold*rorder)
plabel=ifelse(rorder>ntop,1,-1)
outdata=data.frame(Time=Test[,1],Y=plabel)
write.csv(outdata,"Submission_Xgb.csv")
