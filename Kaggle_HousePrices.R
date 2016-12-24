###### Guide https://www.kaggle.com/skirmer/house-prices-advanced-regression-techniques/fun-with-real-estate-data/notebook
setwd("C:/Github/Kaggle_HousePrice/")

###### Loading pacakges
library(data.table)
library(FeatureHashing)
library(Matrix)
library(xgboost)
require(randomForest)
require(caret)
require(dplyr)
require(ggplot2)
library(pROC)
library(stringr)
library(dummies)
library(Metrics)
library(kernlab)
library(mlbench)

##### Load the data
train <- read.csv("train.csv", stringsAsFactors=FALSE)
test <- read.csv("test.csv", stringsAsFactors=FALSE)

names(train)

##### Formatting data
## Street
train$paved[train$Street == "Pave"] <- 1
train$paved[train$Street != "Pave"] <- 0

## Lot Shape
train$regshape[train$LotShape == "Reg"] <- 1
train$regshape[train$LotShape != "Reg"] <- 0

train$flat[train$LandContour == "Lvl"] <- 1
train$flat[train$LandContour != "Lvl"] <- 0

train$pubutil[train$Utilities == "AllPub"] <- 1
train$pubutil[train$Utilities != "AllPub"] <- 0

train$gentle_slope[train$LandSlope == "Gtl"] <- 1
train$gentle_slope[train$LandSlope != "Gtl"] <- 0

train$culdesac_fr3[train$LandSlope %in% c("CulDSac", "FR3")] <- 1
train$culdesac_fr3[!train$LandSlope %in% c("CulDSac", "FR3")] <- 0

nbhdprice <- summarize(group_by(train, Neighborhood),mean(SalePrice, na.rm=T))
nbhdprice_lo <- filter(nbhdprice, nbhdprice$`mean(SalePrice, na.rm = T)` < 140000)
nbhdprice_med <- filter(nbhdprice, nbhdprice$`mean(SalePrice, na.rm = T)` < 200000 &
                          nbhdprice$`mean(SalePrice, na.rm = T)` >= 140000 )
nbhdprice_hi <- filter(nbhdprice, nbhdprice$`mean(SalePrice, na.rm = T)` >= 200000)
train$nbhd_price_level[train$Neighborhood %in% nbhdprice_lo$Neighborhood] <- 1
train$nbhd_price_level[train$Neighborhood %in% nbhdprice_med$Neighborhood] <- 2
train$nbhd_price_level[train$Neighborhood %in% nbhdprice_hi$Neighborhood] <- 3

train$pos_features_1[train$Condition1 %in% c("PosA", "PosN")] <- 1
train$pos_features_1[!train$Condition1 %in% c("PosA", "PosN")] <- 0

train$pos_features_2[train$Condition1 %in% c("PosA", "PosN")] <- 1
train$pos_features_2[!train$Condition1 %in% c("PosA", "PosN")] <- 0

train$twnhs_end_or_1fam[train$BldgType %in% c("1Fam", "TwnhsE")] <- 1
train$twnhs_end_or_1fam[!train$BldgType %in% c("1Fam", "TwnhsE")] <- 0

housestyle_price <- summarize(group_by(train, HouseStyle),mean(SalePrice, na.rm=T))

housestyle_lo <- filter(housestyle_price, housestyle_price$`mean(SalePrice, na.rm = T)` < 140000)
housestyle_med <- filter(housestyle_price, housestyle_price$`mean(SalePrice, na.rm = T)` < 200000 &
housestyle_price$`mean(SalePrice, na.rm = T)` >= 140000 )
housestyle_hi <- filter(housestyle_price, housestyle_price$`mean(SalePrice, na.rm = T)` >= 200000)

train$house_style_level[train$HouseStyle %in% housestyle_lo$HouseStyle] <- 1
train$house_style_level[train$HouseStyle %in% housestyle_med$HouseStyle] <- 2
train$house_style_level[train$HouseStyle %in% housestyle_hi$HouseStyle] <- 3

roofstyle_price <- summarize(group_by(train, RoofStyle),mean(SalePrice, na.rm=T))

train$roof_hip_shed[train$RoofStyle %in% c("Hip", "Shed")] <- 1
train$roof_hip_shed[!train$RoofStyle %in% c("Hip", "Shed")] <- 0

roofmatl_price <- summarize(group_by(train, RoofMatl),mean(SalePrice, na.rm=T))

train$roof_matl_hi[train$RoofMatl %in% c("Membran", "WdShake", "WdShngl")] <- 1
train$roof_matl_hi[!train$RoofMatl %in% c("Membran", "WdShake", "WdShngl")] <- 0

price <- summarize(group_by(train, Exterior1st),mean(SalePrice, na.rm=T))

matl_lo_1 <- filter(price, price$`mean(SalePrice, na.rm = T)` < 140000)
matl_med_1<- filter(price, price$`mean(SalePrice, na.rm = T)` < 200000 & price$`mean(SalePrice, na.rm = T)` >= 140000 )
matl_hi_1 <- filter(price, price$`mean(SalePrice, na.rm = T)` >= 200000)

train$exterior_1[train$Exterior1st %in% matl_lo_1$Exterior1st] <- 1
train$exterior_1[train$Exterior1st %in% matl_med_1$Exterior1st] <- 2
train$exterior_1[train$Exterior1st %in% matl_hi_1$Exterior1st] <- 3

price <- summarize(group_by(train, Exterior2nd),mean(SalePrice, na.rm=T))

matl_lo <- filter(price, price$`mean(SalePrice, na.rm = T)` < 140000)
matl_med <- filter(price, price$`mean(SalePrice, na.rm = T)` < 200000 & price$`mean(SalePrice, na.rm = T)` >= 140000 )
matl_hi <- filter(price, price$`mean(SalePrice, na.rm = T)` >= 200000)

train$exterior_2[train$Exterior2nd %in% matl_lo$Exterior2nd] <- 1
train$exterior_2[train$Exterior2nd %in% matl_med$Exterior2nd] <- 2
train$exterior_2[train$Exterior2nd %in% matl_hi$Exterior2nd] <- 3

price <- summarize(group_by(train, MasVnrType),mean(SalePrice, na.rm=T))

train$exterior_mason_1[train$MasVnrType %in% c("Stone", "BrkFace") | is.na(train$MasVnrType)] <- 1
train$exterior_mason_1[!train$MasVnrType %in% c("Stone", "BrkFace") & !is.na(train$MasVnrType)] <- 0

price <- summarize(group_by(train, ExterQual),mean(SalePrice, na.rm=T))

train$exterior_cond[train$ExterQual == "Ex"] <- 4
train$exterior_cond[train$ExterQual == "Gd"] <- 3
train$exterior_cond[train$ExterQual == "TA"] <- 2
train$exterior_cond[train$ExterQual == "Fa"] <- 1

price <- summarize(group_by(train, ExterCond),mean(SalePrice, na.rm=T))

train$exterior_cond2[train$ExterCond == "Ex"] <- 5
train$exterior_cond2[train$ExterCond == "Gd"] <- 4
train$exterior_cond2[train$ExterCond == "TA"] <- 3
train$exterior_cond2[train$ExterCond == "Fa"] <- 2
train$exterior_cond2[train$ExterCond == "Po"] <- 1

price <- summarize(group_by(train, Foundation),mean(SalePrice, na.rm=T))

train$found_concrete[train$Foundation == "PConc"] <- 1
train$found_concrete[train$Foundation != "PConc"] <- 0

price <- summarize(group_by(train, BsmtQual),mean(SalePrice, na.rm=T))

train$bsmt_cond1[train$BsmtQual == "Ex"] <- 5
train$bsmt_cond1[train$BsmtQual == "Gd"] <- 4
train$bsmt_cond1[train$BsmtQual == "TA"] <- 3
train$bsmt_cond1[train$BsmtQual == "Fa"] <- 2
train$bsmt_cond1[is.na(train$BsmtQual)] <- 1

price <- summarize(group_by(train, BsmtCond),mean(SalePrice, na.rm=T))

train$bsmt_cond2[train$BsmtCond == "Gd"] <- 5
train$bsmt_cond2[train$BsmtCond == "TA"] <- 4
train$bsmt_cond2[train$BsmtCond == "Fa"] <- 3
train$bsmt_cond2[is.na(train$BsmtCond)] <- 2
train$bsmt_cond2[train$BsmtCond == "Po"] <- 1

price <- summarize(group_by(train, BsmtExposure),mean(SalePrice, na.rm=T))

train$bsmt_exp[train$BsmtExposure == "Gd"] <- 5
train$bsmt_exp[train$BsmtExposure == "Av"] <- 4
train$bsmt_exp[train$BsmtExposure == "Mn"] <- 3
train$bsmt_exp[train$BsmtExposure == "No"] <- 2
train$bsmt_exp[is.na(train$BsmtExposure)] <- 1

price <- summarize(group_by(train, BsmtFinType1),mean(SalePrice, na.rm=T))

train$bsmt_fin1[train$BsmtFinType1 == "GLQ"] <- 5
train$bsmt_fin1[train$BsmtFinType1 == "Unf"] <- 4
train$bsmt_fin1[train$BsmtFinType1 == "ALQ"] <- 3
train$bsmt_fin1[train$BsmtFinType1 %in% c("BLQ", "Rec", "LwQ")] <- 2
train$bsmt_fin1[is.na(train$BsmtFinType1)] <- 1

price <- summarize(group_by(train, BsmtFinType2),mean(SalePrice, na.rm=T))

train$bsmt_fin2[train$BsmtFinType2 == "ALQ"] <- 6
train$bsmt_fin2[train$BsmtFinType2 == "Unf"] <- 5
train$bsmt_fin2[train$BsmtFinType2 == "GLQ"] <- 4
train$bsmt_fin2[train$BsmtFinType2 %in% c("Rec", "LwQ")] <- 3
train$bsmt_fin2[train$BsmtFinType2 == "BLQ"] <- 2
train$bsmt_fin2[is.na(train$BsmtFinType2)] <- 1

price <- summarize(group_by(train, Heating),mean(SalePrice, na.rm=T))

train$gasheat[train$Heating %in% c("GasA", "GasW")] <- 1
train$gasheat[!train$Heating %in% c("GasA", "GasW")] <- 0

price <- summarize(group_by(train, HeatingQC),mean(SalePrice, na.rm=T))

train$heatqual[train$HeatingQC == "Ex"] <- 5
train$heatqual[train$HeatingQC == "Gd"] <- 4
train$heatqual[train$HeatingQC == "TA"] <- 3
train$heatqual[train$HeatingQC == "Fa"] <- 2
train$heatqual[train$HeatingQC == "Po"] <- 1

price <- summarize(group_by(train, CentralAir),mean(SalePrice, na.rm=T))

train$air[train$CentralAir == "Y"] <- 1
train$air[train$CentralAir == "N"] <- 0

price <- summarize(group_by(train, Electrical),mean(SalePrice, na.rm=T))

train$standard_electric[train$Electrical == "SBrkr" | is.na(train$Electrical)] <- 1
train$standard_electric[!train$Electrical == "SBrkr" & !is.na(train$Electrical)] <- 0

price <- summarize(group_by(train, KitchenQual),mean(SalePrice, na.rm=T))

train$kitchen[train$KitchenQual == "Ex"] <- 4
train$kitchen[train$KitchenQual == "Gd"] <- 3
train$kitchen[train$KitchenQual == "TA"] <- 2
train$kitchen[train$KitchenQual == "Fa"] <- 1

price <- summarize(group_by(train, FireplaceQu),mean(SalePrice, na.rm=T))

train$fire[train$FireplaceQu == "Ex"] <- 5
train$fire[train$FireplaceQu == "Gd"] <- 4
train$fire[train$FireplaceQu == "TA"] <- 3
train$fire[train$FireplaceQu == "Fa"] <- 2
train$fire[train$FireplaceQu == "Po" | is.na(train$FireplaceQu)] <- 1

price <- summarize(group_by(train, GarageType),mean(SalePrice, na.rm=T))

train$gar_attach[train$GarageType %in% c("Attchd", "BuiltIn")] <- 1
train$gar_attach[!train$GarageType %in% c("Attchd", "BuiltIn")] <- 0

price <- summarize(group_by(train, GarageFinish),mean(SalePrice, na.rm=T))

train$gar_finish[train$GarageFinish %in% c("Fin", "RFn")] <- 1
train$gar_finish[!train$GarageFinish %in% c("Fin", "RFn")] <- 0

price <- summarize(group_by(train, GarageQual),mean(SalePrice, na.rm=T))

train$garqual[train$GarageQual == "Ex"] <- 5
train$garqual[train$GarageQual == "Gd"] <- 4
train$garqual[train$GarageQual == "TA"] <- 3
train$garqual[train$GarageQual == "Fa"] <- 2
train$garqual[train$GarageQual == "Po" | is.na(train$GarageQual)] <- 1

price <- summarize(group_by(train, GarageCond),mean(SalePrice, na.rm=T))

train$garqual2[train$GarageCond == "Ex"] <- 5
train$garqual2[train$GarageCond == "Gd"] <- 4
train$garqual2[train$GarageCond == "TA"] <- 3
train$garqual2[train$GarageCond == "Fa"] <- 2
train$garqual2[train$GarageCond == "Po" | is.na(train$GarageCond)] <- 1

price <- summarize(group_by(train, PavedDrive),mean(SalePrice, na.rm=T))

train$paved_drive[train$PavedDrive == "Y"] <- 1
train$paved_drive[!train$PavedDrive != "Y"] <- 0
train$paved_drive[is.na(train$paved_drive)] <- 0

price <- summarize(group_by(train, Functional),mean(SalePrice, na.rm=T))

train$housefunction[train$Functional %in% c("Typ", "Mod")] <- 1
train$housefunction[!train$Functional %in% c("Typ", "Mod")] <- 0

price <- summarize(group_by(train, PoolQC),mean(SalePrice, na.rm=T))

train$pool_good[train$PoolQC %in% c("Ex")] <- 1
train$pool_good[!train$PoolQC %in% c("Ex")] <- 0

price <- summarize(group_by(train, Fence),mean(SalePrice, na.rm=T))

train$priv_fence[train$Fence %in% c("GdPrv")] <- 1
train$priv_fence[!train$Fence %in% c("GdPrv")] <- 0

price <- summarize(group_by(train, MiscFeature),mean(SalePrice, na.rm=T))
price <- summarize(group_by(train, SaleType),mean(SalePrice, na.rm=T))

train$sale_cat[train$SaleType %in% c("New", "Con")] <- 5
train$sale_cat[train$SaleType %in% c("CWD", "ConLI")] <- 4
train$sale_cat[train$SaleType %in% c("WD")] <- 3
train$sale_cat[train$SaleType %in% c("COD", "ConLw", "ConLD")] <- 2
train$sale_cat[train$SaleType %in% c("Oth")] <- 1

price <- summarize(group_by(train, SaleCondition),mean(SalePrice, na.rm=T))

train$sale_cond[train$SaleCondition %in% c("Partial")] <- 4
train$sale_cond[train$SaleCondition %in% c("Normal", "Alloca")] <- 3
train$sale_cond[train$SaleCondition %in% c("Family","Abnorml")] <- 2
train$sale_cond[train$SaleCondition %in% c("AdjLand")] <- 1

price <- summarize(group_by(train, MSZoning),mean(SalePrice, na.rm=T))

train$zone[train$MSZoning %in% c("FV")] <- 4
train$zone[train$MSZoning %in% c("RL")] <- 3
train$zone[train$MSZoning %in% c("RH","RM")] <- 2
train$zone[train$MSZoning %in% c("C (all)")] <- 1

price <- summarize(group_by(train, Alley),mean(SalePrice, na.rm=T))

train$alleypave[train$Alley %in% c("Pave")] <- 1
train$alleypave[!train$Alley %in% c("Pave")] <- 0

train$Street <- NULL
train$LotShape <- NULL
train$LandContour <- NULL
train$Utilities <- NULL
train$LotConfig <- NULL
train$LandSlope <- NULL
train$Neighborhood <- NULL
train$Condition1 <- NULL
train$Condition2 <- NULL
train$BldgType <- NULL
train$HouseStyle <- NULL
train$RoofStyle <- NULL
train$RoofMatl <- NULL

train$Exterior1st <- NULL
train$Exterior2nd <- NULL
train$MasVnrType <- NULL
train$ExterQual <- NULL
train$ExterCond <- NULL

train$Foundation <- NULL
train$BsmtQual <- NULL
train$BsmtCond <- NULL
train$BsmtExposure <- NULL
train$BsmtFinType1 <- NULL
train$BsmtFinType2 <- NULL

train$Heating <- NULL
train$HeatingQC <- NULL
train$CentralAir <- NULL
train$Electrical <- NULL
train$KitchenQual <- NULL
train$FireplaceQu <- NULL

train$GarageType <- NULL
train$GarageFinish <- NULL
train$GarageQual <- NULL
train$GarageCond <- NULL
train$PavedDrive <- NULL

train$Functional <- NULL
train$PoolQC <- NULL
train$Fence <- NULL
train$MiscFeature <- NULL
train$SaleType <- NULL
train$SaleCondition <- NULL
train$MSZoning <- NULL
train$Alley <- NULL

###### Correlations
library(corrplot)

correlations <- cor(train[,c(5,6,7,8, 16:25)], use="everything")
corrplot(correlations, method="circle", type="lower",  sig.level = 0.01, insig = "blank")

correlations <- cor(train[,c(5,6,7,8, 26:35)], use="everything")
corrplot(correlations, method="circle", type="lower",  sig.level = 0.01, insig = "blank")

correlations <- cor(train[,c(5,6,7,8, 66:75)], use="everything")
corrplot(correlations, method="circle", type="lower",  sig.level = 0.01, insig = "blank")


pairs(~YearBuilt+OverallQual+TotalBsmtSF+GrLivArea,data=train,main="Simple Scatterplot Matrix")

library(car)

scatterplot(SalePrice ~ YearBuilt, data=train,  xlab="Year Built", ylab="Sale Price", grid=FALSE)
scatterplot(SalePrice ~ YrSold, data=train,  xlab="Year Sold", ylab="Sale Price", grid=FALSE)
scatterplot(SalePrice ~ X1stFlrSF, data=train,  xlab="Square Footage Floor 1", ylab="Sale Price", grid=FALSE)

#Fix some NAs
train$GarageYrBlt[is.na(train$GarageYrBlt)] <- 0
train$MasVnrArea[is.na(train$MasVnrArea)] <- 0
train$LotFrontage[is.na(train$LotFrontage)] <- 0

#Interactions based on correlation
train$year_qual <- train$YearBuilt*train$OverallQual #overall condition
train$year_r_qual <- train$YearRemodAdd*train$OverallQual #quality x remodel
train$qual_bsmt <- train$OverallQual*train$TotalBsmtSF #quality x basement size

train$livarea_qual <- train$OverallQual*train$GrLivArea #quality x living area
train$qual_bath <- train$OverallQual*train$FullBath #quality x baths

train$qual_ext <- train$OverallQual*train$exterior_cond #quality x exterior

###### Doing all the same to the test data
test$paved[test$Street == "Pave"] <- 1
test$paved[test$Street != "Pave"] <- 0

test$regshape[test$LotShape == "Reg"] <- 1
test$regshape[test$LotShape != "Reg"] <- 0

test$flat[test$LandContour == "Lvl"] <- 1
test$flat[test$LandContour != "Lvl"] <- 0

test$pubutil[test$Utilities == "AllPub"] <- 1
test$pubutil[test$Utilities != "AllPub"] <- 0

test$gentle_slope[test$LandSlope == "Gtl"] <- 1
test$gentle_slope[test$LandSlope != "Gtl"] <- 0

test$culdesac_fr3[test$LandSlope %in% c("CulDSac", "FR3")] <- 1
test$culdesac_fr3[!test$LandSlope %in% c("CulDSac", "FR3")] <- 0

test$nbhd_price_level[test$Neighborhood %in% nbhdprice_lo$Neighborhood] <- 1
test$nbhd_price_level[test$Neighborhood %in% nbhdprice_med$Neighborhood] <- 2
test$nbhd_price_level[test$Neighborhood %in% nbhdprice_hi$Neighborhood] <- 3

test$pos_features_1[test$Condition1 %in% c("PosA", "PosN")] <- 1
test$pos_features_1[!test$Condition1 %in% c("PosA", "PosN")] <- 0

test$pos_features_2[test$Condition1 %in% c("PosA", "PosN")] <- 1
test$pos_features_2[!test$Condition1 %in% c("PosA", "PosN")] <- 0

test$twnhs_end_or_1fam[test$BldgType %in% c("1Fam", "TwnhsE")] <- 1
test$twnhs_end_or_1fam[!test$BldgType %in% c("1Fam", "TwnhsE")] <- 0

test$house_style_level[test$HouseStyle %in% housestyle_lo$HouseStyle] <- 1
test$house_style_level[test$HouseStyle %in% housestyle_med$HouseStyle] <- 2
test$house_style_level[test$HouseStyle %in% housestyle_hi$HouseStyle] <- 3

test$roof_hip_shed[test$RoofStyle %in% c("Hip", "Shed")] <- 1
test$roof_hip_shed[!test$RoofStyle %in% c("Hip", "Shed")] <- 0

test$roof_matl_hi[test$RoofMatl %in% c("Membran", "WdShake", "WdShngl")] <- 1
test$roof_matl_hi[!test$RoofMatl %in% c("Membran", "WdShake", "WdShngl")] <- 0

test$exterior_1[test$Exterior1st %in% matl_lo_1$Exterior1st] <- 1
test$exterior_1[test$Exterior1st %in% matl_med_1$Exterior1st] <- 2
test$exterior_1[test$Exterior1st %in% matl_hi_1$Exterior1st] <- 3

test$exterior_2[test$Exterior2nd %in% matl_lo$Exterior2nd] <- 1
test$exterior_2[test$Exterior2nd %in% matl_med$Exterior2nd] <- 2
test$exterior_2[test$Exterior2nd %in% matl_hi$Exterior2nd] <- 3

test$exterior_mason_1[test$MasVnrType %in% c("Stone", "BrkFace") | is.na(test$MasVnrType)] <- 1
test$exterior_mason_1[!test$MasVnrType %in% c("Stone", "BrkFace") & !is.na(test$MasVnrType)] <- 0

test$exterior_cond[test$ExterQual == "Ex"] <- 4
test$exterior_cond[test$ExterQual == "Gd"] <- 3
test$exterior_cond[test$ExterQual == "TA"] <- 2
test$exterior_cond[test$ExterQual == "Fa"] <- 1

test$exterior_cond2[test$ExterCond == "Ex"] <- 5
test$exterior_cond2[test$ExterCond == "Gd"] <- 4
test$exterior_cond2[test$ExterCond == "TA"] <- 3
test$exterior_cond2[test$ExterCond == "Fa"] <- 2
test$exterior_cond2[test$ExterCond == "Po"] <- 1

test$found_concrete[test$Foundation == "PConc"] <- 1
test$found_concrete[test$Foundation != "PConc"] <- 0

test$bsmt_cond1[test$BsmtQual == "Ex"] <- 5
test$bsmt_cond1[test$BsmtQual == "Gd"] <- 4
test$bsmt_cond1[test$BsmtQual == "TA"] <- 3
test$bsmt_cond1[test$BsmtQual == "Fa"] <- 2
test$bsmt_cond1[is.na(test$BsmtQual)] <- 1

test$bsmt_cond2[test$BsmtCond == "Gd"] <- 5
test$bsmt_cond2[test$BsmtCond == "TA"] <- 4
test$bsmt_cond2[test$BsmtCond == "Fa"] <- 3
test$bsmt_cond2[is.na(test$BsmtCond)] <- 2
test$bsmt_cond2[test$BsmtCond == "Po"] <- 1

test$bsmt_exp[test$BsmtExposure == "Gd"] <- 5
test$bsmt_exp[test$BsmtExposure == "Av"] <- 4
test$bsmt_exp[test$BsmtExposure == "Mn"] <- 3
test$bsmt_exp[test$BsmtExposure == "No"] <- 2
test$bsmt_exp[is.na(test$BsmtExposure)] <- 1

test$bsmt_fin1[test$BsmtFinType1 == "GLQ"] <- 5
test$bsmt_fin1[test$BsmtFinType1 == "Unf"] <- 4
test$bsmt_fin1[test$BsmtFinType1 == "ALQ"] <- 3
test$bsmt_fin1[test$BsmtFinType1 %in% c("BLQ", "Rec", "LwQ")] <- 2
test$bsmt_fin1[is.na(test$BsmtFinType1)] <- 1

test$bsmt_fin2[test$BsmtFinType2 == "ALQ"] <- 6
test$bsmt_fin2[test$BsmtFinType2 == "Unf"] <- 5
test$bsmt_fin2[test$BsmtFinType2 == "GLQ"] <- 4
test$bsmt_fin2[test$BsmtFinType2 %in% c("Rec", "LwQ")] <- 3
test$bsmt_fin2[test$BsmtFinType2 == "BLQ"] <- 2
test$bsmt_fin2[is.na(test$BsmtFinType2)] <- 1

test$gasheat[test$Heating %in% c("GasA", "GasW")] <- 1
test$gasheat[!test$Heating %in% c("GasA", "GasW")] <- 0

test$heatqual[test$HeatingQC == "Ex"] <- 5
test$heatqual[test$HeatingQC == "Gd"] <- 4
test$heatqual[test$HeatingQC == "TA"] <- 3
test$heatqual[test$HeatingQC == "Fa"] <- 2
test$heatqual[test$HeatingQC == "Po"] <- 1

test$air[test$CentralAir == "Y"] <- 1
test$air[test$CentralAir == "N"] <- 0

test$standard_electric[test$Electrical == "SBrkr" | is.na(test$Electrical)] <- 1
test$standard_electric[!test$Electrical == "SBrkr" & !is.na(test$Electrical)] <- 0

test$kitchen[test$KitchenQual == "Ex"] <- 4
test$kitchen[test$KitchenQual == "Gd"] <- 3
test$kitchen[test$KitchenQual == "TA"] <- 2
test$kitchen[test$KitchenQual == "Fa"] <- 1

test$fire[test$FireplaceQu == "Ex"] <- 5
test$fire[test$FireplaceQu == "Gd"] <- 4
test$fire[test$FireplaceQu == "TA"] <- 3
test$fire[test$FireplaceQu == "Fa"] <- 2
test$fire[test$FireplaceQu == "Po" | is.na(test$FireplaceQu)] <- 1

test$gar_attach[test$GarageType %in% c("Attchd", "BuiltIn")] <- 1
test$gar_attach[!test$GarageType %in% c("Attchd", "BuiltIn")] <- 0

test$gar_finish[test$GarageFinish %in% c("Fin", "RFn")] <- 1
test$gar_finish[!test$GarageFinish %in% c("Fin", "RFn")] <- 0

test$garqual[test$GarageQual == "Ex"] <- 5
test$garqual[test$GarageQual == "Gd"] <- 4
test$garqual[test$GarageQual == "TA"] <- 3
test$garqual[test$GarageQual == "Fa"] <- 2
test$garqual[test$GarageQual == "Po" | is.na(test$GarageQual)] <- 1

test$garqual2[test$GarageCond == "Ex"] <- 5
test$garqual2[test$GarageCond == "Gd"] <- 4
test$garqual2[test$GarageCond == "TA"] <- 3
test$garqual2[test$GarageCond == "Fa"] <- 2
test$garqual2[test$GarageCond == "Po" | is.na(test$GarageCond)] <- 1

test$paved_drive[test$PavedDrive == "Y"] <- 1
test$paved_drive[!test$PavedDrive != "Y"] <- 0
test$paved_drive[is.na(test$paved_drive)] <- 0

test$housefunction[test$Functional %in% c("Typ", "Mod")] <- 1
test$housefunction[!test$Functional %in% c("Typ", "Mod")] <- 0

test$pool_good[test$PoolQC %in% c("Ex")] <- 1
test$pool_good[!test$PoolQC %in% c("Ex")] <- 0

test$priv_fence[test$Fence %in% c("GdPrv")] <- 1
test$priv_fence[!test$Fence %in% c("GdPrv")] <- 0

test$sale_cat[test$SaleType %in% c("New", "Con")] <- 5
test$sale_cat[test$SaleType %in% c("CWD", "ConLI")] <- 4
test$sale_cat[test$SaleType %in% c("WD")] <- 3
test$sale_cat[test$SaleType %in% c("COD", "ConLw", "ConLD")] <- 2
test$sale_cat[test$SaleType %in% c("Oth")] <- 1

test$sale_cond[test$SaleCondition %in% c("Partial")] <- 4
test$sale_cond[test$SaleCondition %in% c("Normal", "Alloca")] <- 3
test$sale_cond[test$SaleCondition %in% c("Family","Abnorml")] <- 2
test$sale_cond[test$SaleCondition %in% c("AdjLand")] <- 1

test$zone[test$MSZoning %in% c("FV")] <- 4
test$zone[test$MSZoning %in% c("RL")] <- 3
test$zone[test$MSZoning %in% c("RH","RM")] <- 2
test$zone[test$MSZoning %in% c("C (all)")] <- 1

test$alleypave[test$Alley %in% c("Pave")] <- 1
test$alleypave[!test$Alley %in% c("Pave")] <- 0

test$Street <- NULL
test$LotShape <- NULL
test$LandContour <- NULL
test$Utilities <- NULL
test$LotConfig <- NULL
test$LandSlope <- NULL
test$Neighborhood <- NULL
test$Condition1 <- NULL
test$Condition2 <- NULL
test$BldgType <- NULL
test$HouseStyle <- NULL
test$RoofStyle <- NULL
test$RoofMatl <- NULL

test$Exterior1st <- NULL
test$Exterior2nd <- NULL
test$MasVnrType <- NULL
test$ExterQual <- NULL
test$ExterCond <- NULL

test$Foundation <- NULL
test$BsmtQual <- NULL
test$BsmtCond <- NULL
test$BsmtExposure <- NULL
test$BsmtFinType1 <- NULL
test$BsmtFinType2 <- NULL

test$Heating <- NULL
test$HeatingQC <- NULL
test$CentralAir <- NULL
test$Electrical <- NULL
test$KitchenQual <- NULL
test$FireplaceQu <- NULL

test$GarageType <- NULL
test$GarageFinish <- NULL
test$GarageQual <- NULL
test$GarageCond <- NULL
test$PavedDrive <- NULL

test$Functional <- NULL
test$PoolQC <- NULL
test$Fence <- NULL
test$MiscFeature <- NULL
test$SaleType <- NULL
test$SaleCondition <- NULL
test$MSZoning <- NULL
test$Alley <- NULL

test$GarageYrBlt[is.na(test$GarageYrBlt)] <- 0
test$MasVnrArea[is.na(test$MasVnrArea)] <- 0
test$LotFrontage[is.na(test$LotFrontage)] <- 0
test$BsmtFinSF1[is.na(test$BsmtFinSF1)] <- 0
test$BsmtFinSF2[is.na(test$BsmtFinSF2)] <- 0
test$BsmtUnfSF[is.na(test$BsmtUnfSF)] <- 0
test$TotalBsmtSF[is.na(test$TotalBsmtSF)] <- 0

test$BsmtFullBath[is.na(test$BsmtFullBath)] <- 0
test$BsmtHalfBath[is.na(test$BsmtHalfBath)] <- 0
test$GarageCars[is.na(test$GarageCars)] <- 0
test$GarageArea[is.na(test$GarageArea)] <- 0
test$pubutil[is.na(test$pubutil)] <- 0

test$year_qual <- test$YearBuilt*test$OverallQual #overall condition
test$year_r_qual <- test$YearRemodAdd*test$OverallQual #quality x remodel
test$qual_bsmt <- test$OverallQual*test$TotalBsmtSF #quality x basement size

test$livarea_qual <- test$OverallQual*test$GrLivArea #quality x living area
test$qual_bath <- test$OverallQual*test$FullBath #quality x baths

test$qual_ext <- test$OverallQual*test$exterior_cond #quality x exterior

###### Model Prepping
outcome <- train$SalePrice

partition <- createDataPartition(y=outcome,p=.5,list=F)
training <- train[partition,]
testing <- train[-partition,]

###### Linear Model
lm_model_15 <- lm(SalePrice ~ ., data=training)
summary(lm_model_15)

lm_model_15 <- lm(SalePrice ~ MSSubClass+LotArea+BsmtUnfSF+
                    X1stFlrSF+X2ndFlrSF+GarageCars+
                    WoodDeckSF+nbhd_price_level+
                    exterior_cond+pos_features_1+
                    bsmt_exp+kitchen+housefunction+pool_good+sale_cond+
                    qual_ext+qual_bsmt, data=training)
summary(lm_model_15)

prediction <- predict(lm_model_15, testing, type="response")
model_output <- cbind(testing, prediction)

model_output$log_prediction <- log(model_output$prediction)
model_output$log_SalePrice <- log(model_output$SalePrice)

#Test with RMSE
rmse(model_output$log_SalePrice,model_output$log_prediction)

## Model over full test set
lm_model_15a <- lm(SalePrice ~ MSSubClass+LotArea+BsmtUnfSF+
                    X1stFlrSF+X2ndFlrSF+GarageCars+
                    WoodDeckSF+nbhd_price_level+
                    exterior_cond+pos_features_1+
                    bsmt_exp+kitchen+housefunction+pool_good+sale_cond+
                    qual_ext+qual_bsmt,data=train)
summary(lm_model_15a)

###### Generating full prediction
predict_lm<-predict(lm_model_15a,test,type="response")
predict_lm[96]<-150000
submit_lm<-cbind(test$Id,predict_lm)
colnames(submit_lm)<-c("Id","SalePrice")
write.csv(submit_lm,"Submit_LinearModel_20161224.csv",row.names=F)




















```

###A Random Forest

Not too bad, given that this is just an LM. Let's try training the model with an RF. Let's use all the variables and see how things look, since randomforest does its own feature selection.

```{r caret1}

model_1 <- randomForest(SalePrice ~ ., data=training)


# Predict using the test set
prediction <- predict(model_1, testing)
model_output <- cbind(testing, prediction)


model_output$log_prediction <- log(model_output$prediction)
model_output$log_SalePrice <- log(model_output$SalePrice)

#Test with RMSE

rmse(model_output$log_SalePrice,model_output$log_prediction)


```

###An xgboost
Nice! Try it with xgboost?

```{r matrices}

#Assemble and format the data

training$log_SalePrice <- log(training$SalePrice)
testing$log_SalePrice <- log(testing$SalePrice)

#Create matrices from the data frames
trainData<- as.matrix(training, rownames.force=NA)
testData<- as.matrix(testing, rownames.force=NA)

#Turn the matrices into sparse matrices
train2 <- as(trainData, "sparseMatrix")
test2 <- as(testData, "sparseMatrix")

#####
#colnames(train2)
#Cross Validate the model

vars <- c(2:37, 39:86) #choose the columns we want to use in the prediction matrix

trainD <- xgb.DMatrix(data = train2[,vars], label = train2[,"SalePrice"]) #Convert to xgb.DMatrix format

#Cross validate the model
cv.sparse <- xgb.cv(data = trainD,
nrounds = 600,
min_child_weight = 0,
max_depth = 10,
eta = 0.02,
subsample = .7,
colsample_bytree = .7,
booster = "gbtree",
eval_metric = "rmse",
verbose = TRUE,
print_every_n = 50,
nfold = 4,
nthread = 2,
objective="reg:linear")

#Train the model

#Choose the parameters for the model
param <- list(colsample_bytree = .7,
subsample = .7,
booster = "gbtree",
max_depth = 10,
eta = 0.02,
eval_metric = "rmse",
objective="reg:linear")


#Train the model using those parameters
bstSparse <-
xgb.train(params = param,
data = trainD,
nrounds = 600,
watchlist = list(train = trainD),
verbose = TRUE,
print_every_n = 50,
nthread = 2)
```


Predict and test the RMSE.
```{r evaluate1}
testD <- xgb.DMatrix(data = test2[,vars])
#Column names must match the inputs EXACTLY
prediction <- predict(bstSparse, testD) #Make the prediction based on the half of the training data set aside

#Put testing prediction and test dataset all together
test3 <- as.data.frame(as.matrix(test2))
prediction <- as.data.frame(as.matrix(prediction))
colnames(prediction) <- "prediction"
model_output <- cbind(test3, prediction)

model_output$log_prediction <- log(model_output$prediction)
model_output$log_SalePrice <- log(model_output$SalePrice)

#Test with RMSE

rmse(model_output$log_SalePrice,model_output$log_prediction)

```

Nice, that's pretty good stuff. I'll take the xgboost I think, let's call that good and make up the submission. Honestly, this is where the interesting stuff basically ends, unless you want to see the retraining and submission formatting.

***
  
  
  ##Retrain on the full sample
  
  ```{r retrain}
rm(bstSparse)

#Create matrices from the data frames
retrainData<- as.matrix(train, rownames.force=NA)

#Turn the matrices into sparse matrices
retrain <- as(retrainData, "sparseMatrix")

param <- list(colsample_bytree = .7,
              subsample = .7,
              booster = "gbtree",
              max_depth = 10,
              eta = 0.02,
              eval_metric = "rmse",
              objective="reg:linear")

retrainD <- xgb.DMatrix(data = retrain[,vars], label = retrain[,"SalePrice"])

#retrain the model using those parameters
bstSparse <-
  xgb.train(params = param,
            data = retrainD,
            nrounds = 600,
            watchlist = list(train = trainD),
            verbose = TRUE,
            print_every_n = 50,
            nthread = 2)

```


##Prepare the prediction data

Here I just repeat the same work I did on the training set, check the code tab to see all the details.

```{r formatting_predictiondata, echo=FALSE}




```


Then, format it for xgboost, I'm just using my boilerplate code for that.
```{r finalpredict2}
# Get the supplied test data ready #

predict <- as.data.frame(test) #Get the dataset formatted as a frame for later combining

#Create matrices from the data frames
predData<- as.matrix(predict, rownames.force=NA)

#Turn the matrices into sparse matrices
predicting <- as(predData, "sparseMatrix")

```


Make sure your training sample and prediction sample have the same variables. I have been including this in code lately because I was making silly mistakes on variable choice.

```{r finalpredict3}
colnames(train[,c(2:37, 39:86)])

vars <- c("MSSubClass","LotFrontage","LotArea","OverallQual","OverallCond","YearBuilt",
"YearRemodAdd","MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF"   ,   
"X1stFlrSF","X2ndFlrSF","LowQualFinSF","GrLivArea","BsmtFullBath","BsmtHalfBath"  ,   
"FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","TotRmsAbvGrd","Fireplaces"     ,  
"GarageYrBlt","GarageCars","GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch"    ,
"X3SsnPorch","ScreenPorch","PoolArea","MiscVal","MoSold","YrSold",
"paved","regshape","flat","pubutil","gentle_slope","culdesac_fr3"     ,
"nbhd_price_level" , "pos_features_1","pos_features_2","twnhs_end_or_1fam","house_style_level", "roof_hip_shed"    ,
"roof_matl_hi","exterior_1","exterior_2","exterior_mason_1","exterior_cond","exterior_cond2"   ,
"found_concrete","bsmt_cond1","bsmt_cond2","bsmt_exp","bsmt_fin1","bsmt_fin2"    ,   
"gasheat","heatqual","air","standard_electric", "kitchen","fire",
"gar_attach","gar_finish","garqual","garqual2","paved_drive","housefunction",
"pool_good","priv_fence","sale_cat","sale_cond","zone","alleypave",
"year_qual","year_r_qual","qual_bsmt","livarea_qual","qual_bath", "qual_ext")

#colnames(predicting)
colnames(predicting[,vars])
```

Actually do the predicting.

```{r finalpredict4}
#Column names must match the inputs EXACTLY
prediction <- predict(bstSparse, predicting[,vars])

prediction <- as.data.frame(as.matrix(prediction))  #Get the dataset formatted as a frame for later combining
colnames(prediction) <- "prediction"
model_output <- cbind(predict, prediction) #Combine the prediction output with the rest of the set

sub2 <- data.frame(Id = model_output$Id, SalePrice = model_output$prediction)
length(model_output$prediction)
write.csv(sub2, file = "sub3.csv", row.names = F)
head(sub2$SalePrice)

```
Comments
3

Mike Hall
4 months ago
A good starter kit for this competition. Especially thanks for all the heavy lifting on the data, as of now I am using that. The classifiers seem fitting for the competition and competitions in general. Although, for now I am still looking at alternative ones. My current score is based on a Weka LibSVM with grid searched parameters, using your data. My prior was a Weka AdditiveRegression meta using RandomForest as the base classifier. It's a long competition, I will probably look at R Random Forest and XGBoost as well sometime. Anyhow, nice, thanks.
3

Andrew Chiu
4 months ago
This is a really good start.

If you read the strings as factors, they should be read as factors anyway when run in your models, so you don't necessarily need to replace the values with numbers.

Leaving the observations as (string) factors would also allow you to explore relationships between variables to create new ones.

If you do want to convert to integers, however, you can try the "as.integer" function on variables. For example:

train$heatqual[train$HeatingQC == "Ex"] <- 5
train$heatqual[train$HeatingQC == "Gd"] <- 4
train$heatqual[train$HeatingQC == "TA"] <- 3
train$heatqual[train$HeatingQC == "Fa"] <- 2
train$heatqual[train$HeatingQC == "Po"] <- 1
could be shortened to:

train$heatqual <- as.integer(train$heatqual)
This way, your code could be reduced quite significantly.
2

Stephanie Kirmer
4 months ago
Hey, thanks guys! I am glad this was helpful- it's nice to be able to give back a little after how much different users and kernels on this site have helped me learn. (And, y'know, share the wealth- if you take big chunks of my code, I'd love if you credit me in your code.)

I actually have continued working on the features on my own a tad, although it hasn't so far gotten me too much better in the results. Been busy with other things too.

Mike: I have to admit, I don't know what a Weka LibSVM is, but I'll go have a google for it and probably learn a lot:)

Andrew: Totally good point, however, in this case, even though some of these variables have sort of ordinal categorical values, if you look at the summaries of the mean SalePrice by value for some, it turns out that the order is not always linear with sale price. Does that make sense? Like, for some of these, there is actually no difference in mean sale price for Fa and Po, so I'd group them for the numeric version, or sometimes the mean sale price for TA is actually lower than Fa, so I'd reorder them when building the numeric. If you look closely at the code I think you'll find a few cases where I did that. It would be much faster to write just converting to integer, but it actually wouldn't be taking advantage of this nice continuous numeric outcome we have to work with. I come from a social science background, not CS, so simplifying code is not so much in my nature, anyway :)
0

Mike Hall
4 months ago
Weka is a java machine learning toolkit. LibSVM is a fairly popular 3rd party SVM implementation that they support as an extension.

Not to nit, when you are considerably outscoring me, but if you haven't yet you might consider null'ing the Id attribute. It seemed to make a difference on some of the modeling I did.
0

Lalit Khandelwal
4 months ago
Has anyone tried GBM here?
0

Mike Hall
4 months ago
Yes, this was the first time I tried it. I think it was messing with shrinkage where I noticed that Id was included in training and removing it got different results, for max.iter and it appeared for attributes eliminated by the shrinkage.

I was trying to determine if R provided anything like the Weka 'meta' wrappers. Where you can do a generalized additive model on top of an arbitrary regression supporting classifier. Maybe even Random Forest or XGBoost. Although as a boosting type it is supposed to be used to improve on weaker models I think. There seemed to be no such thing in R. Not being integrally developed as a whole there doesn't seem to be the plug and play type usage between packages.

Even with Weka though AdditiveRegression appeared to help with RandomForest but didn't with LibSVM. LibSVM was better without it standalone.

gbm itself appeared to get results similar to what Stephanie showed for linear regression. I assume that LR is what it uses as the only classifier it boosts?
0

MathisPanzani
4 months ago
Thanks this was very useful :)
0

Tang
3 months ago
Thanks for sharing! I was wondering:

why you would log-transform the predicted and observed dependant variable ('SalePrice') before measuring the RMSE?
besides, you also log-transformed the dependant variable for the XGboost model, but not for the other models (linear and RF). I thought the main reason to log-transform the dependant variable was to interpret the coefficient of an explanotory variable as the effect in percentage on the dependant variable. But I believe this only holds for linear models (am I wrong?). If the log-transformation was used to shrink the variance of high values of' 'SalePrice', why not using it for the other models?
0

Stephanie Kirmer
3 months ago
Thank you for the questions! I actually log transformed the variables before testing RMSE in all of the models, not just xgboost. If you search for "log" on the code tab you'll see this. The question of why is to do with RMSE itself, actually. I am a little new to this evaluation metric, but I found that the sizable range of values for the price variables, and especially the high end outliers, were causing problems in the RMSE function. I did some research and found that using the log of each was a way to reduce the inordinate effect of these extremely large high end outliers. It proved effective and the leaderboard agreed. I hope that's helpful; sorry I can't give more insight into the math.
0

Tang
3 months ago
Hey, thanks for the quick answer :)

OK I got the idea for the RMSE. It's like you are giving less weight to predictions/observations with high values (and thus potentially high errors).

For my second question, correct me if I'm wrong, but it seems you are modeling the absolute value of 'SalePrice' with both LM and RF, but not for XGBoost for which you modeled the log value of 'SalePrice'. Any particular reason for that?
0

Stephanie Kirmer
3 months ago
Oh, ha, I see what you mean! I didn't actually use log of sale price to train the code, I just generated it ahead of the model training (for some idiot reason I can't recall). This line is identifying what will train the code:
                                                                                                                                         
                                                                                                                                         trainD <- xgb.DMatrix(data = train2[,vars], label = train2[,"SalePrice"]) #Convert to xgb.DMatrix format
                                                                                                                                         
                                                                                                                                         "label" is telling the matrix what the outcome is, which is sale price, not log. I don't know why I generated the saleprice_log before that point and again after the training, it doesn't get included in the training dataset at all if you look closely:
                                                                                                                                         
                                                                                                                                         vars <- c(2:37, 39:86) #choose the columns we want to use in the prediction matrix
                                                                                                                                         
                                                                                                                                         saleprice_log is 87, if memory serves. Lines 665 and 737 are more or less duplicating work unnecessarily. So, as I said, I don't know why past me generated the sale price log earlier.
                                                                                                                                         0
                                                                                                                                         
                                                                                                                                         Tang
                                                                                                                                         3 months ago
                                                                                                                                         Oh my bad then, I didn't read your code carefully enough ;-)
                                                                                                                                         
                                                                                                                                         Thanks for the answer!
                                                                                                                                         0
                                                                                                                                         
                                                                                                                                         Thai Le
                                                                                                                                         3 months ago
                                                                                                                                         @Stephanie Line 99&100 should be accessing variable LotConfig isn't it?
                                                                                                                                         0
                                                                                                                                         
                                                                                                                                         Thai Le · last edited 3 months ago by Thai Le
                                                                                                                                         3 months ago
                                                                                                                                         Thanks for great post @Stephanie
                                                                                                                                         0
                                                                                                                                         
                                                                                                                                         Danny Malter · last edited 3 months ago by Danny Malter
                                                                                                                                         3 months ago
                                                                                                                                         @Stephanie, great post. Instead of adding/deleting features from the training data way at the top of your code and then the test set way at the bottom, you can run it with less lines of code using the mutate function from the dplyr package.
                                                                                                                                         
                                                                                                                                         clean <- function(df){
                                                                                                                                           mutate(df, paved = ifelse(Street == 'Pave', 1, 0),
                                                                                                                                                  df, regshape = ifelse(LotShape == 'Reg', 1, 0))
                                                                                                                                         }
                                                                                                                                         
                                                                                                                                         training <- clean(training)
                                                                                                                                         testing <- clean(testing)
                                                                                                                                         
                                                                                                                                         library(data.table)
                                                                                                                                         training <- data.table(training)
                                                                                                                                         training[ ,c("Street","LotShape") := NULL]
                                                                                                                                         Additionally,
                                                                                                                                         
                                                                                                                                         train$culdesac_fr3[train$LandSlope %in% c("CulDSac", "FR3")] <- 1
                                                                                                                                         train$culdesac_fr3[!train$LandSlope %in% c("CulDSac", "FR3")] <- 0
                                                                                                                                         should be train$LotConfig instead of LandSlope
                                                                                                                                         0
                                                                                                                                         
                                                                                                                                         Stephanie Kirmer
                                                                                                                                         3 months ago
                                                                                                                                         Hi Danny, yes, that's a good idea, I would probably do that in the future. One reason I didn't here is because testing functions in the editing interface here on kaggle is a bit of a pain/time consuming. Thanks for pointing it out, though!
                                                                                                                                           0
                                                                                                                                         
                                                                                                                                         Richat
                                                                                                                                         2 months ago
                                                                                                                                         Thank you for sharing this! It gave me few more ideas!
                                                                                                                                           0
                                                                                                                                         
                                                                                                                                         David Samson
                                                                                                                                         2 months ago
                                                                                                                                         Thanks for this. Very helpful to organize my work.
                                                                                                                                         0
                                                                                                                                         
                                                                                                                                         Scottish Chemjong
                                                                                                                                         20 days ago
                                                                                                                                         Hey mate,
                                                                                                                                         
                                                                                                                                         I didn't understand the logic behind converting factors into numeric such as:
                                                                                                                                         
                                                                                                                                         train$paved[train$Street == "Pave"] <- 1 train$paved[train$Street != "Pave"] <- 0
                                                                                                                                         
                                                                                                                                         You then ran correlation between variables. But the correlation coefficient depends how you assign values to factors. For eg. if you had assigned
                                                                                                                                         
                                                                                                                                         train$paved[train$Street == "Pave"] <- 0 train$paved[train$Street != "Pave"] <- 1
                                                                                                                                         
                                                                                                                                         you would get different correlation - cor(train$SalePrice, train$paved).
                                                                                                                                         
                                                                                                                                         And my next question is why cant you run categorical variables straight into lm function? whats your reason behind assigning numeric to these categorical values?
                                                                                                                                         
                                                                                                                                         Thanks
                                                                                                                                         Styling with Markdown supported
                                                                                                                                         Enter your comments.
                                                                                                                                         Post Reply
                                                                                                                                         © 2016 Kaggle Inc
                                                                                                                                         Our Team Careers Terms Privacy Contact/Support
                                                                                                                                         