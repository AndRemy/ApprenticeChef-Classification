library(readxl)
library(rpart)
library(rpart.plot)

setwd("C:/Users/Andre/Documents/AndRemy/Hult/2.MSBA/2.Spring Term/3.Machine Learning/3.Assignments/A2")

original_df <- read_excel("Apprentice_Chef_Dataset.xlsx")
tree_tit <- rpart(
  formula = `CROSS_SELL_SUCCESS` ~ 
    `AVG_PREP_VID_TIME`         +
    `AVG_TIME_PER_SITE_VISIT`   +
    `LARGEST_ORDER_SIZE`        +
    `LATE_DELIVERIES`           +
    `MASTER_CLASSES_ATTENDED`   +
    `PACKAGE_LOCKER`            +
    `PC_LOGINS`                 +
    `PRODUCT_CATEGORIES_VIEWED` +
    `TASTES_AND_PREFERENCES`    +
    `TOTAL_MEALS_ORDERED`       +
    `UNIQUE_MEALS_PURCH`,
  data    = original_df,
  method  = "class"
)

plotcp(tree_tit)

rpart.plot(
  x     = tree_tit,
  type  = 1,
  extra = 1,
  cex   = 0.5
)
