#' Copyright (c) 2023 Stéphane LASSALVY
#' 
#' Disclaimer of Warranty.
#' 
#' THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY
#' APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT
#' HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY
#' OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO,
#' THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#' PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM
#' IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF
#' ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

#' References :
#' - Generalization of the code available on the website https://www.databricks.com/
#'  to use more quantitative predictors and encapsulation within a function. 
#' Source material is from this webpage :
#' https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/2961012104553482/2761297084239405/1806228006848429/latest.html
#'
#' - Cheng, B., & Titterington, D. M. (1994). Neural Networks: A Review from a Statistical Perspective. Statistical Science, 9(1), 2–30. http://www.jstor.org/stable/2246275
#' - Rosenblatt, F. (1958). The perceptron: a probabilistic model for information storage and organization in the brain. Psychological review, 65(6), 386.
#'

#' Function H
#' 
#' @author Stéphane Lassalvy 2023
#' 
#' @description Heaviside function
#' @param x        : a scalar
#' @param treshold : a scalar used as threshold for the Heaviside function
#' 
#' @return         : a scalar -1 or +1
#' 
H <- function(x, threshold = 0){if(x >= threshold){result <- 1} else {result <- -1}; return(result)}

#' Function perceptron
#' 
#' @author Stéphane Lassalvy 2023
#' 
#' @description Binary classifier based on the Rosenblatt Perceptron (1957)
#' 
#' @param y                : binary variable response -1 / 1
#' @param X :              : quantitative predictors
#' @param f :              : transfer function
#' @param M :              : number of epochs to run
#' @param batch_size :     : sub-sample size to sub-sample data to apply the stochastic gradient descent
#' @param eta :            : learning rate
#' @param target_accurracy : target accuracy (True positive + True negative) / (total sample size)
#' @param verbose :        : verbose mode TRUE / FALSE
#'
perceptron <- function(y, X, f = H, M = 15, batch_size = trunc(nrow(X)/3), eta = 0.005, target_accurracy = 0.8, verbose = FALSE){
  if(batch_size > nrow(X)){stop("batch_size must be lower than the complete sample size")}
  # add an intercept column to X
  X <- cbind(rep(1, nrow(X)), X)
  
  # initialize weights vector
  w <- rep(0, ncol(X))
  
  # initialize fitted response vector
  y_init <- rep(-1, length(y))
  
  # compute initial accuracy
  accuracy <- sum(y_init == y) / length(y)
  
  # while accuracy is under the targeted accuracy
  i <- 0
  while(accuracy <  target_accurracy && i <= M){
    i <- i + 1
    cat(paste('Epoch starts: ', i, "\n"))
    cat("-------------------------------------------\n")
    
    # We reshuffle the order of the observations for each epoch (batch_size = nrow(X)),
    # or make a sub-sample without replacement (batch_size < nrow(X))
    index = 1:nrow(X)
    index = sample(index, size = batch_size, replace = FALSE)
    
    for (j in index){
      z = X[j,] %*% w
      
      y_pred <- f(z)
      
      w <- w + eta * (y[j] - y_pred) * X[j,]       
      
      if (verbose == TRUE){
        cat(paste("-> updating data point ', j, ' :     \n"))
        names(w) <- paste("w", seq(0,length(w)-1,by = 1),sep="_")
        print(w)
      }
    }
    
    z_all  <- X %*% w
    y_pred_all <- unlist(lapply(X = z_all, FUN = f))
    
    accuracy <- sum(y_pred_all == y) / length(y)
    cat(paste('Epoch ends  : ', i, "with accuracy: ", round(accuracy, 5), "\n"))
    cat("-------------------------------------------\n")
  }     
  return(list(weights = w, z = z_all, f = f , fitted = y_pred_all))
}