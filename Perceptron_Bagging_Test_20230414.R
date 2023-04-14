# This is free and unencumbered software released into the public domain.
# 
# Anyone is free to copy, modify, publish, use, compile, sell, or
# distribute this software, either in source code form or as a compiled
# binary, for any purpose, commercial or non-commercial, and by any
# means.
# 
# In jurisdictions that recognize copyright laws, the author or authors
# of this software dedicate any and all copyright interest in the
# software to the public domain. We make this dedication for the benefit
# of the public at large and to the detriment of our heirs and
# successors. We intend this dedication to be an overt act of
# relinquishment in perpetuity of all present and future rights to this
# software under copyright law.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
# 
# For more information, please refer to <http://unlicense.org/>
  
#' References :
#' - Generalization of the code available on the website https://www.databricks.com/ to use more quantitative predictors and encapsulation within a function, 
#' Source material including a data simulation is from this page :
#' https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2961012104553482/2761297084239405/1806228006848429/latest.html
#'
#' - Cheng, B., & Titterington, D. M. (1994). Neural Networks: A Review from a Statistical Perspective. Statistical Science, 9(1), 2–30. http://www.jstor.org/stable/2246275
#' - Rosenblatt, F. (1958). The perceptron: a probabilistic model for information storage and organization in the brain. Psychological review, 65(6), 386.
#'
library(tidyverse)
# source("Perceptron_function_20230414.R")
set.seed(4567)
H <- function(x, threshold = 0){if(x >= threshold){result <- 1} else {result <- -1}; return(result)}

N = 50 # total number of data points each group
x_offset = 0.5 # group separation on x axis
y_offset = 0.5 # group separation on y axis

g1_x = runif(N, min = 0, max = 1)
g1_y = runif(N, min = 0, max = 1)

g2_x = runif(N, min = 0+x_offset, max = 1+x_offset)
g2_y = runif(N, min = 0+y_offset, max = 1+y_offset)

g_x = c(g1_x, g2_x)
g_y = c(g1_y, g2_y)
y = c(rep(-1,N), rep(1,N))

print(g_x)
print(g_y)
print(y)

X <- cbind(x = g_x, y = g_y)

# Bagging
X <- cbind(intercept = rep(1,nrow(X)), X)
pred            <- NULL
predictions_OOB <- data.frame(n_obs = 1:nrow(X))
B <- 5000
for(i in 1:B){
  cat("Boostrap iteration = ")
  cat(i)
  cat("\n")
  cat("------------------------")
  cat("\n")
  
  index <- sample(1:length(y), size = length(y), replace = TRUE)
  y_ <- y[index]
  X_ <- X[index, -1]
  
  results <- perceptron(y_, X_, target_accurracy = 0.80)
  w <- results$weights

  y_pred <- unlist(lapply(X %*% w, H))
  pred <- cbind(pred, y_pred)
  
  y_oob <- y[-unique(index)]
  X_oob <- X[-unique(index),]
  y_pred_oob <- unlist(lapply(X_oob %*% w, H))
  predictions_oob <- eval(parse(text = paste("data.frame(n_obs = (1:nrow(X))[-unique(index)], fit_oob_",i, "= y_pred_oob)", sep = "")))
  predictions_OOB <- left_join(predictions_OOB, predictions_oob, by = join_by(n_obs))                                       
}

predictions_OOB <- column_to_rownames(predictions_OOB, var = "n_obs")
sum.oob <- apply(predictions_OOB, 1, sum, na.rm = TRUE)
any(is.na(sum.oob))
any(sum.oob == 0)
fitted.oob <- sign(sum.oob)

# Confusion matrix
conf_matrix_oob <- table(fitted = fitted.oob, initial = y)
conf_matrix_oob

# Sensibility
sens_oob <- conf_matrix_oob[1,1] / sum(conf_matrix_oob[,1])
sens_oob

# Specificity
spec_oob <- conf_matrix_oob[2,2] / sum(conf_matrix_oob[,2])
spec_oob

# Results
sum.vec <- apply(pred,1,sum)
length(sum.vec)
any(sum.vec == 0)
fitted.vec <- sign(sum.vec)

# Confusion matrix
conf_matrix <- table(fitted = fitted.vec, initial = y)
conf_matrix

# Plot with the observed classes
par(mfrow=c(1,2))
plot(g_x, g_y, type='n', xlab='X', ylab='Y', main = "Scatterplot of predictors\nwith the initial classes")
points(g1_x, g1_y, col='red')
points(g2_x, g2_y, col='blue')

# second part of the plot
colors.vec <- fitted.vec
colors.vec[colors.vec == -1] <- 0
colors.vec <- colors.vec + 1
plot(g_x, g_y, type='n', xlab='X', ylab='Y', main = "Scatterplot of predictors\nwith the predicted classes")
points(c(g1_x,g2_x), c(g1_y, g2_y), col=c("burlywood","green")[colors.vec])

# Sensibility
sens <- conf_matrix[1,1] / sum(conf_matrix[,1])
sens
# Specificity
spec <- conf_matrix[2,2] / sum(conf_matrix[,2])
spec
