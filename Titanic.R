###Introduction 

#Loading libraries
library(data.table)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(randomForest)
library(tidyr)
library(caret)

#Read in data
titanic <- fread("/Users/antbae/OneDrive/R/Side_projects/Titanicv2/training.txt")

#Convert to data table
titanic <- as.data.table(titanic)

#Data exploration
head(titanic)
str(titanic)

#Convert to factors
titanic$Survived <- as.factor(titanic$Survived)
titanic$Pclass <- as.factor(titanic$Pclass)
titanic$Sex <- as.factor(titanic$Sex)
titanic$Embarked <- as.factor(titanic$Embarked)

###Exploration

#Create a function to be able to visualize tables and bar charts
vis1 <- function(feat){
  temp.table <- table(titanic[[feat]], titanic$Survived)
  print(temp.table)
  
  ggplot(titanic, aes_string(x = feat)) +
    geom_bar(aes(fill = Survived)) +
    theme_few() +
    guides(fill = guide_legend(reverse=TRUE)) +
    ylab("") +
    xlab("") +
    ggtitle(paste(feat))
}

#Create a function to be able to visualize proportions and stacked bar charts
vis2 <- function(feat){
  prop.temp  <- as.data.frame(prop.table(table(titanic[[feat]], titanic$Survived),1))
  print(prop.temp)
  
  ggplot(prop.temp, aes(x = Var1, y = Freq, fill = Var2)) +
    geom_bar(stat = "identity") +
    theme_few() +
    guides(fill = guide_legend(reverse=TRUE)) +
    ylab("") +
    xlab("") +
    ggtitle(paste(feat)) +
    scale_fill_discrete(name="Survived")
}

#Sex and survival
vis1("Sex")
vis2("Sex")

#Pclass and survival
vis1("Pclass")
vis2("Pclass")

#Embarked and survival
vis1("Embarked")
vis2("Embarked")

#Go deeper
ggplot(titanic, aes(x = Pclass)) +
  geom_bar(aes(fill = Survived)) +
  facet_grid(~Sex) +
  theme_few() +
  guides(fill = guide_legend(reverse=TRUE))

###Missing values

#missing embarked data 
table(titanic$Embarked)
titanic[Embarked == ""]

#From same port. Ticket = 113572, Fare is 80, Pclass 1, Sex is female

#Let's see if there is a relationship between ticket number and where they boarded
titanic[Ticket > 11300 & Ticket < 11500,]

#Medians for embarked prices 
ggplot(titanic, aes(x = Pclass, y = Fare, fill = Pclass)) +
  geom_boxplot() +
  facet_grid(~Embarked, scales = "free") +
  geom_hline(aes(yintercept=80)) +
  theme_few() +
  scale_fill_few() +
  xlab("") +
  ylab("") +
  ggtitle("Medians of fares for Embarks")

titanic[, .(median = median(Fare)), by = .(Embarked, Pclass)][order(Embarked)]

#We can assume that a female that pays a fare of 80 at first class comes from Embarked 'C'
titanic$Embarked[titanic$Embarked == ""] <- "C"

#Missing ages
sum(is.na(titanic))
sum(is.na(titanic$Age))

titanic[is.na(Age)]

#use feature engineering to find missing values of age. Let's extract titles from passenger's names
titanic$Title <- gsub('(.*, )|(\\..*)', '', titanic$Name)

#using wikipedia they define the title master as a boy younger than 16. Trying to fill in NAs for age 
#will prove difficult unless we use a predictive model since NA ages include females that do not have a 
#corresponding term for younger than 16.

#verification
titanic[is.na(Age), .(Title, Name, Sex, Age, Fare)]

### Outliers
temp <- titanic[,.(Fare)]
summary(temp)

#Let's see how much people are paying by title
temp <- titanic[!is.na(Age), .(median = median(Fare), Sex, Age, Fare, Embarked), by = .(Title, Pclass)][order(Title)]

temp %>% 
  distinct(Title)

#Create a boxplot to check
ggplot(titanic, aes(x = Title, y = Fare, fill = Pclass)) +
  geom_boxplot() +
  theme_few() +
  scale_fill_few() +
  facet_wrap(~Pclass, scales = "free") + 
  xlab("") +
  ylab("") +
  ggtitle("Outliers") +
  theme(
    axis.text.x = element_text(angle = 90)
  )

#Verify the right skew 
ggplot(titanic, aes(x = Fare)) + 
  geom_histogram(binwidth = 15) +
  theme_few() +
  xlab("") +
  ylab("") +
  ggtitle("Right skew verification")

#Fare transformation
titanic$sqrtfares <- sqrt(titanic$Fare)

ggplot(titanic, aes(x = Title, y = sqrtfares, fill = Pclass)) +
  geom_boxplot() +
  theme_few() +
  scale_fill_few() +
  facet_wrap(~Pclass, scales = "free") + 
  xlab("") +
  ylab("") +
  ggtitle("Medians of fares for titles x Pclass") +
  theme(
    axis.text.x = element_text(angle = 90)
  )

ggplot(titanic, aes(x = sqrtfares)) + 
  geom_histogram(binwidth = 1) +
  theme_few() +
  xlab("") +
  ylab("") +
  ggtitle("Sqrt of fares")

#Comparing previous fares with sqrtfares 
temp3 <- gather(titanic, type, fare, c(Fare, sqrtfares))                                

ggplot(temp3, aes(x = fare, fill = type, color = type)) +
  stat_density(alpha = .1) +
  facet_wrap(~type, scale = "free") + 
  theme_few() +
  xlab("") +
  ylab("") +
  ggtitle("Fares vs Sqrt of Fares")

### Feature engineering

#Creating a family by adding siblings + parents + and self
str(titanic)
titanic$Fam <- (titanic$SibSp + titanic$Parch + 1)

#Looks like there is a rise in death for family sizes over 4. It may be worth it to create a dummy variable for this. 
ggplot(titanic, aes(x = Fam, fill = Survived)) + 
  geom_bar(position = "fill") +
  theme_few() +
  xlab("") +
  ylab("") +
  ggtitle("Family size and surival") +
  guides(fill = guide_legend(reverse=TRUE))

titanic[,Fam.Five := ifelse(titanic$Fam >= 5, 1, 0)]

#Let's verify 
titanic[Fam.Five == 1, .(Fam, Fam.Five)] 

#Cabin feature creation. $First letter
titanic$Cabin

titanic[, Cab.Letter := substr(Cabin, 1,1)]

levels(titanic$Cab.Letter)
titanic$Cab.Letter[titanic$Cab.Letter == ""] <- "Unknown"

#1 digit number
titanic[, Cabin.1digit := ifelse(nchar(Cabin) == 2, 1, 0)]

#2 digit number
titanic[, Cabin.2digit := ifelse(nchar(Cabin) == 3, 1, 0)]

#3 digit number
titanic[, Cabin.3digit := ifelse(nchar(Cabin) == 4, 1, 0)]

#multiple rooms 
titanic[, Cabin.Multiple := ifelse(nchar(Cabin) > 4, 1, 0)]

###RandomForest

#Set up factors for randomForest
titanic$Title <- as.factor(titanic$Title)
titanic$Cab.Letter <- as.factor(titanic$Cab.Letter)
titanic$Cabin.1digit <- as.factor(titanic$Cabin.1digit)
titanic$Cabin.2digit <- as.factor(titanic$Cabin.2digit)
titanic$Cabin.3digit <- as.factor(titanic$Cabin.3digit)
titanic$Cabin.Multiple <- as.factor(titanic$Cabin.Multiple)

#Set up training and test data set
set.seed(1)
split <- createDataPartition(titanic$Survived, p = .85, list = FALSE)

training <- titanic[split,]
testing <- titanic[-split,]

#Create model
m1 <- randomForest(Survived ~ Pclass + Sex + sqrtfares + Fam.Five + Title + 
                    Cabin.2digit, data = training, importance = TRUE)

m1
m1$importance
varImpPlot(m1)

#Test model 
p1 <- predict(m1, testing)

#Check results
table(testing$Survived, p1)


mean(testing$Survived == p1)

.218