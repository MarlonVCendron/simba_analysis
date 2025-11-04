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
  select(Video, GRUPO, STAT_1, STAT_2, DISL_1, DISL_2) %>%
  pivot_longer(cols = c(STAT_1, STAT_2, DISL_1, DISL_2), names_to = "Object", values_to = "Duration") %>%
  mutate(
    Object = as.factor(Object),
    Group = as.factor(GRUPO),
    Rat = as.factor(Video),
  )

# data$Duration <- data$Duration + 0.00001
levels(data$Group) <- c("Saline", "Muscimol")

cat("Data dimensions:", dim(data), "\n")
cat("Data summary:\n")
print(str(data))

p1 <- ggplot(data, aes(x = Object, y = Duration, fill = Group)) +
  geom_boxplot(position = position_dodge(0.8)) +
  scale_fill_manual(values = c("Saline" = "#145faa", "Muscimol" = "#828287")) +
  labs(title = "Exploration Duration by Object and Group", x = "Object", y = "Duration (s)") +
  theme_classic()

formula <- Duration ~ Group + Object + Group:Object + (1|Rat)

lmm <- lmer(formula, data = data)
glmm_gaussian_log <- glmer(formula, data = data, family = gaussian(link = "log"))
# Tirando pq tem 0 nos dados e lmm é melhor, mesmo quando faz +1
# glmm_gamma_log <- glmer(formula, data = data, family = Gamma(link = "log"))

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

# models <- list("LMM" = lmm, "GLMM_Gamma_Log" = glmm_gamma_log)
models <- list("LMM" = lmm)
# models <- list("LMM" = lmm, "GLMM_Gaussian_Log" = glmm_gaussian_log)
comparison <- compare_models(models, "Duration")

best_model_idx <- which.min(comparison$AIC)
best_model <- if(length(best_model_idx) > 0 && !is.na(best_model_idx)) comparison$Model[best_model_idx] else "LMM"
model <- models[[best_model]]

print(best_model)

residuals <- residuals(model)
simres <- simulateResiduals(fittedModel = model)
  
p2 <- ggplot(mapping = aes(sample = residuals)) +
  stat_qq_line(size = 1, col = "steelblue") +
  stat_qq(size = 2) +
  labs(title = paste("Q-Q Plot:", best_model, "Duration"), 
       x = "Theoretical Quantiles", y = "Sample Quantiles") +
  theme_classic()

plot(simres, main = paste("DHARMa Diagnostics:", best_model, "Duration"))

print(summary(model))
print(anova(model))

ggsave("figures/glmm/S2_exp_duration_boxplot.png", p1, width = 8, height = 6, dpi = 300)

ggsave("figures/glmm/S2_exp_duration_qq.png", p2, width = 6, height = 6, dpi = 300)
