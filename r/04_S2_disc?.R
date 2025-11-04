suppressPackageStartupMessages({
  library(ggplot2)
  library(tidyr)
  library(dplyr)
  library(emmeans)
  library(lme4)
  library(lmerTest)
  library(DHARMa)
})

r_data <- read.csv("data/spss/S2_exp.csv", header=TRUE, na.strings = "NA", sep=";", dec=",")

data <- r_data %>%
  select(Video, GRUPO, SPACE_IDX) %>%
  mutate(
    Index = as.numeric(SPACE_IDX),
    Group = as.factor(GRUPO),
    Rat = as.factor(Video),
  )

# data$Duration <- data$Duration + 0.00001
levels(data$Group) <- c("Saline", "Muscimol")

cat("Data dimensions:", dim(data), "\n")
cat("Data summary:\n")
print(str(data))

p1 <- ggplot(data, aes(x = Group, y = Index, fill = Group)) +
  geom_boxplot(position = position_dodge(0.8)) +
  scale_fill_manual(values = c("Saline" = "#145faa", "Muscimol" = "#828287")) +
  labs(title = "Exploration Index by Group", x = "Group", y = "Index") +
  theme_classic()

formula <- Index ~ Group

# Modelo linear simples
lmm <- lm(formula, data = data)

compare_models <- function(models, outcome) {
  n_models <- length(models)
  model_names <- names(models)
  if(is.null(model_names) || length(model_names) == 0 || any(model_names == "")) {
    model_names <- paste0("Model_", 1:n_models)
  }
  results <- data.frame(
    Model = model_names,
    AIC = rep(NA, n_models),
    BIC = rep(NA, n_models),
    logLik = rep(NA, n_models),
    stringsAsFactors = FALSE
  )
  
  for(i in 1:n_models) {
    model <- models[[i]]
    if(!is.null(model)) {
      aic_val <- tryCatch(AIC(model), error = function(e) NA)
      bic_val <- tryCatch(BIC(model), error = function(e) NA)
      loglik_val <- tryCatch(logLik(model), error = function(e) NA)

      results$AIC[i] <- if(length(aic_val) > 0 && !any(is.na(aic_val))) aic_val[1] else NA
      results$BIC[i] <- if(length(bic_val) > 0 && !any(is.na(bic_val))) bic_val[1] else NA
      results$logLik[i] <- if(length(loglik_val) > 0 && !any(is.na(loglik_val))) loglik_val[1] else NA
    }
  }
  
  results$Outcome <- outcome
  return(results)
}


residuals <- residuals(lmm)
simres <- simulateResiduals(fittedModel = lmm)
  
p2 <- ggplot(mapping = aes(sample = residuals)) +
  stat_qq_line(size = 1, col = "steelblue") +
  stat_qq(size = 2) +
  labs(title = paste("Q-Q Plot: LMM Index"), 
       x = "Theoretical Quantiles", y = "Sample Quantiles") +
  theme_classic()

print(summary(lmm))
print(anova(lmm))

ggsave("figures/glmm/S2_disc_index_boxplot.png", p1, width = 8, height = 6, dpi = 300)

ggsave("figures/glmm/S2_disc_index_qq.png", p2, width = 6, height = 6, dpi = 300)
