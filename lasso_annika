# Load required libraries
library(tidyverse)
library(glmnet)
mutate(across(c(mode, key), as.numeric))
# Load the dataset
spotify_data <- read_csv("datasets/Spotify Most Streamed Songs.csv")

# Clean and transform the data
spotify_data_clean <- spotify_data %>%
  drop_na() %>%
  mutate(
    streams = as.numeric(gsub("[^0-9]", "", streams)),
    danceability = scale(`danceability_%` / 100),
    valence = scale(`valence_%` / 100),
    energy = scale(`energy_%` / 100),
    acousticness = scale(`acousticness_%` / 100),
    instrumentalness = scale(`instrumentalness_%` / 100),
    liveness = scale(`liveness_%` / 100),
    speechiness = scale(`speechiness_%` / 100),
    bpm_scaled = scale(bpm),
    mode = as.factor(mode),
    key = as.factor(key)
  ) %>%
  select(
    streams, danceability, valence, energy, acousticness,
    instrumentalness, liveness, speechiness, mode, key, bpm_scaled
  )
spotify_data_clean <- spotify_data_clean %>%
  mutate(streams = log1p(streams))  

spotify_data_encoded <- spotify_data_clean %>%
  mutate(across(c(mode, key), as.numeric))  

X <- spotify_data_encoded %>% select(-streams) %>% as.matrix()
y <- spotify_data_encoded$streams

set.seed(42)
train_indices <- sample(seq_len(nrow(X)), size = 0.8 * nrow(X))
X_train <- X[train_indices, ]
X_test <- X[-train_indices, ]
y_train <- y[train_indices]
y_test <- y[-train_indices]

lasso_model <- glmnet(X_train, y_train, alpha = 1)

cv_lasso <- cv.glmnet(X_train, y_train, alpha = 1, lambda = 10^seq(-5, 5, length = 100))

best_lambda <- cv_lasso$lambda.min

final_model <- glmnet(X_train, y_train, alpha = 1, lambda = best_lambda)

y_pred <- predict(final_model, newx = X_test)

mse <- mean((y_test - y_pred)^2)
r_squared <- 1 - sum((y_test - y_pred)^2) / sum((y_test - mean(y_test))^2)

plot(cv_lasso)
title("Cross-Validation for Lasso Regression", line = 2)


plot(lasso_model, xvar = "lambda", label = TRUE)
title("Lasso Coefficient Paths", line = 2)

list(mse = mse, r_squared = r_squared, best_lambda = best_lambda)
