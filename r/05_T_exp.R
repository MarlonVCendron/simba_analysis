suppressPackageStartupMessages({
  library(ggplot2)
  library(tidyr)
  library(dplyr)
  library(emmeans)
  library(lme4)
  library(lmerTest)
  library(DHARMa)
})

r_data <- read.csv("data/spss/T_exp.csv", header=TRUE, na.strings = "NA", sep=";", dec=",")

data <- r_data %>%
  select(ANIMAL, GRUPO, ETHO_A1, ETHO_A2, ETHO_B1, ETHO_B2) %>%
  pivot_longer(cols = c(ETHO_A1, ETHO_A2, ETHO_B1, ETHO_B2), names_to = "Object", values_to = "Duration") %>%
  # filter(Duration > 0) %>%  # Remove rows where A1, A2, B1, or B2 is 0
  mutate(
    Object = case_when(
      Object == "ETHO_A1" ~ "A1",
      Object == "ETHO_A2" ~ "A2",
      Object == "ETHO_B1" ~ "B1",
      Object == "ETHO_B2" ~ "B2"
    ),
    Object = as.factor(Object),
    Group = as.factor(GRUPO),
    Rat = as.factor(ANIMAL),
  )

# data$Duration <- data$Duration + 1

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
# glmm_gaussian_log <- glmer(formula, data = data, family = gaussian(link = "log"))
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
# models <- list("LMM" = lmm, "GLMM_Gaussian_Log" = glmm_gaussian_log, "GLMM_Gamma_Log" = glmm_gamma_log)
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

ggsave("figures/glmm/T_exp_duration_boxplot.png", p1, width = 8, height = 6, dpi = 300)

ggsave("figures/glmm/T_exp_duration_qq.png", p2, width = 6, height = 6, dpi = 300)


which_outliers <- which(abs(residuals) > 3)
print(paste("Outliers found:", which_outliers))
data[which_outliers, ]                   # inspect the rows
