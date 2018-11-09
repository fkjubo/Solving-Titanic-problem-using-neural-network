# importing data

data.Train <- read.csv("train.csv", T, ",")
data.Test <- read.csv("test.csv", T, ",")

# analysing the data

str(data.Train)
str(data.Test)

summary(data.Train)
summary(data.Test)

# deleting useless variable

data.Train <- data.Train[,-c(1,4,9,11,12)]
data.Test <- data.Test[,-c(3,8,10,11)]

# imputing knn for predicting the missing value

library(VIM)

Imputation_MV <- kNN(data.Train, variable = "Age", k= 7)

data.Train <- subset(Imputation_MV, select = Survived:Fare)

Imputation_MV <- kNN(data.Test, variable = c("Age", "Fare"), k= 7)

data.Test <- subset(Imputation_MV, select = PassengerId:Fare)

data.Test$Survived <- NA

# changing structure of the data for the model
data.Train$Sex <- as.integer(data.Train$Sex)
data.Test$Sex <- as.integer(data.Test$Sex)
data.Test$Survived <- as.integer(data.Test$Survived)

# in order to run the neural network all the value should be in between 0 and 1

# training data

data.Train$Pclass <- (data.Train$Pclass - min(data.Train$Pclass)) / (max(data.Train$Pclass) - min(data.Train$Pclass))
data.Train$Sex <- (data.Train$Sex - min(data.Train$Sex)) / (max(data.Train$Sex) - min(data.Train$Sex))
data.Train$Age <- (data.Train$Age - min(data.Train$Age)) / (max(data.Train$Age) - min(data.Train$Age))
data.Train$SibSp <- (data.Train$SibSp - min(data.Train$SibSp)) / (max(data.Train$SibSp) - min(data.Train$SibSp))
data.Train$Parch <- (data.Train$Parch - min(data.Train$Parch)) / (max(data.Train$Parch) - min(data.Train$Parch))
data.Train$Fare <- (data.Train$Fare - min(data.Train$Fare)) / (max(data.Train$Fare) - min(data.Train$Fare))

# testing data

data.Test$Pclass <- (data.Test$Pclass - min(data.Test$Pclass)) / (max(data.Test$Pclass) - min(data.Test$Pclass))
data.Test$Sex <- (data.Test$Sex - min(data.Test$Sex)) / (max(data.Test$Sex) - min(data.Test$Sex))
data.Test$Age <- (data.Test$Age - min(data.Test$Age)) / (max(data.Test$Age) - min(data.Test$Age))
data.Test$SibSp <- (data.Test$SibSp - min(data.Test$SibSp)) / (max(data.Test$SibSp) - min(data.Test$SibSp))
data.Test$Parch <- (data.Test$Parch - min(data.Test$Parch)) / (max(data.Test$Parch) - min(data.Test$Parch))
data.Test$Fare <- (data.Test$Fare - min(data.Test$Fare)) / (max(data.Test$Fare) - min(data.Test$Fare))


# running neural model on the data with hidden nodes 10

library(neuralnet)

model <- neuralnet(Survived ~ Pclass+Sex+Age+SibSp+Parch+Fare,
                   data = data.Train,
                   hidden = 10,
                   err.fct = "ce",
                   linear.output = F)
# plotting the data

plot(model)

# predicting the data

output <- compute(model, data.Test[,-c(1,8)])
p1 <- output$net.result

# using ifelse function to predict the data
pred1 <- ifelse(p1>.5,1,0)

# only applicable if predicting the traing data as our test data has no data about Surviving

tab <- table(pred1, data.Train$Survived)
1-sum(diag(tab))/sum(tab)

# creating a file for kaggle submission

PassengerId <- data.Test$PassengerId
output.df <- as.data.frame(PassengerId)
output.df$Survived <- pred1
head(output.df)

# creating a csv file

write.csv(output.df, file = "submission.csv", row.names = F)
