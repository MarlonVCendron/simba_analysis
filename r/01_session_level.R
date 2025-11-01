# Session-Level Rearing Analysis: Group × Session
# GEE vs GLMM comparison for rearing duration and frequency

# Setup
suppressPackageStartupMessages({
  library(geepack)
  library(ggplot2)
  library(tidyr)
  library(dplyr)
  library(emmeans)
  library(lme4)
  library(lmerTest)
  library(DHARMa)
})

# Data
r_data <- read.csv("data/spss/rearing.csv", header=TRUE, na.strings = "NA", sep=";", dec=",")

# Prepare data
data_long <- r_data %>%
  select(Video, GRUPO, DUR_S1, DUR_S2, DUR_T, FREQ_S1, FREQ_S2, FREQ_T) %>%
  pivot_longer(cols = c(DUR_S1, DUR_S2, DUR_T), names_to = "Session", values_to = "Duration") %>%
  pivot_longer(cols = c(FREQ_S1, FREQ_S2, FREQ_T), names_to = "Session_freq", values_to = "Frequency") %>%
  filter(
    (Session == "DUR_S1" & Session_freq == "FREQ_S1") |
    (Session == "DUR_S2" & Session_freq == "FREQ_S2") |
    (Session == "DUR_T" & Session_freq == "FREQ_T")
  ) %>%
  select(-Session_freq) %>%
  mutate(
    Session = case_when(
      Session == "DUR_S1" ~ "S1",
      Session == "DUR_S2" ~ "S2", 
      Session == "DUR_T" ~ "T"
    ),
    Session = as.factor(Session),
    Group = as.factor(GRUPO),
    Rat = as.factor(Video),
    Rat_id = as.numeric(as.factor(Video))
  )

levels(data_long$Group) <- c("Saline", "Muscimol")
levels(data_long$Session) <- c("S1", "S2", "T")

cat("Data dimensions:", dim(data_long), "\n")
cat("Data summary:\n")
print(str(data_long))

p1 <- ggplot(data_long, aes(x = Session, y = Duration, fill = Group)) +
  geom_boxplot(position = position_dodge(0.8)) +
  scale_fill_manual(values = c("Saline" = "#145faa", "Muscimol" = "#828287")) +
  labs(title = "Rearing Duration by Group and Session", x = "Session", y = "Duration (s)") +
  theme_classic()

p2 <- ggplot(data_long, aes(x = Session, y = Frequency, fill = Group)) +
  geom_boxplot(position = position_dodge(0.8)) +
  scale_fill_manual(values = c("Saline" = "#145faa", "Muscimol" = "#828287")) +
  labs(title = "Rearing Frequency by Group and Session", x = "Session", y = "Frequency") +
  theme_classic()

formula_dur_null <- Duration ~ 1 + (1|Rat)
formula_dur <- Duration ~ Group + Session + Group:Session + (1|Rat)
formula_dur_gee <- Duration ~ Group + Session + Group:Session

cat("Fitting LMM for duration...\n")
lmm_dur <- tryCatch(lmer(formula_dur, data = data_long), error = function(e) NULL)

cat("Fitting GLMM Gamma for duration...\n")
glmm_dur <- tryCatch(glmer(formula_dur, data = data_long, family = Gamma(link = "log")), error = function(e) NULL)
glmm_dur_null <- tryCatch(glmer(formula_dur_null, data = data_long, family = Gamma(link = "log")), error = function(e) NULL)

cat("Fitting GEE Normal for duration...\n")
gee_dur_norm <- tryCatch(geeglm(formula_dur_gee, data = data_long, id = Rat_id, 
                                family = gaussian("identity"), corstr = "ar1"), error = function(e) NULL)

cat("Fitting GEE Gamma for duration...\n")
gee_dur_gamma <- tryCatch(geeglm(formula_dur_gee, data = data_long, id = Rat_id, 
                                 family = Gamma("log"), corstr = "ar1"), error = function(e) NULL)

formula_freq <- Frequency ~ Group + Session + Group:Session + (1|Rat)
formula_freq_gee <- Frequency ~ Group + Session + Group:Session

cat("Fitting LMM for frequency...\n")
lmm_freq <- tryCatch(lmer(formula_freq, data = data_long), error = function(e) NULL)

cat("Fitting GLMM Gamma for frequency...\n")
glmm_freq <- tryCatch(glmer(formula_freq, data = data_long, family = Gamma(link = "log")), error = function(e) NULL)

cat("Fitting GEE Normal for frequency...\n")
gee_freq_norm <- tryCatch(geeglm(formula_freq_gee, data = data_long, id = Rat_id, 
                                 family = gaussian("identity"), corstr = "ar1"), error = function(e) NULL)

cat("Fitting GEE Gamma for frequency...\n")
gee_freq_gamma <- tryCatch(geeglm(formula_freq_gee, data = data_long, id = Rat_id, 
                                  family = Gamma("log"), corstr = "ar1"), error = function(e) NULL)

dur_models <- list(
  "LMM" = lmm_dur,
  "GLMM_Gamma" = glmm_dur,
  "GEE_Normal" = gee_dur_norm,
  "GEE_Gamma" = gee_dur_gamma
)

freq_models <- list(
  "LMM" = lmm_freq,
  "GLMM_Gamma" = glmm_freq,
  "GEE_Normal" = gee_freq_norm,
  "GEE_Gamma" = gee_freq_gamma
)

compare_models <- function(models, outcome) {
  valid_models <- models[!sapply(models, is.null)]
  
  if(length(valid_models) == 0) {
    return(data.frame(Model = "None", AIC = NA, BIC = NA, QIC = NA, Outcome = outcome))
  }
  
  n_models <- length(valid_models)
  results <- data.frame(
    Model = names(valid_models),
    AIC = rep(NA, n_models),
    BIC = rep(NA, n_models),
    QIC = rep(NA, n_models),
    stringsAsFactors = FALSE
  )
  
  for(i in 1:n_models) {
    model <- valid_models[[i]]
    if(!is.null(model)) {
      aic_val <- tryCatch(AIC(model), error = function(e) NA)
      bic_val <- tryCatch(BIC(model), error = function(e) NA)
      qic_val <- tryCatch(QIC(model), error = function(e) NA)
      
      results$AIC[i] <- if(length(aic_val) > 0 && !any(is.na(aic_val))) aic_val[1] else NA
      results$BIC[i] <- if(length(bic_val) > 0 && !any(is.na(bic_val))) bic_val[1] else NA
      results$QIC[i] <- if(length(qic_val) > 0 && !any(is.na(qic_val))) qic_val[1] else NA
    }
  }
  
  results$Outcome <- outcome
  return(results)
}

dur_comparison <- compare_models(dur_models, "Duration")
freq_comparison <- compare_models(freq_models, "Frequency")

print("Model Comparison - Duration:")
print(dur_comparison)
print("\nModel Comparison - Frequency:")
print(freq_comparison)

cat("\nModel fitting status:\n")
for(i in 1:length(dur_models)) {
  cat(names(dur_models)[i], ":", class(dur_models[[i]]), "\n")
}

best_dur_idx <- which.min(dur_comparison$AIC)
best_freq_idx <- which.min(freq_comparison$AIC)

best_dur <- if(length(best_dur_idx) > 0 && !is.na(best_dur_idx)) names(dur_models)[best_dur_idx] else "LMM"
best_freq <- if(length(best_freq_idx) > 0 && !is.na(best_freq_idx)) names(freq_models)[best_freq_idx] else "LMM"

cat("\nBest model for Duration:", best_dur)
cat("\nBest model for Frequency:", best_freq, "\n")

# Diagnostics for best models
if(best_dur %in% c("LMM", "GLMM_Gamma")) {
  dur_residuals <- residuals(dur_models[[best_dur]])
  dur_simres <- simulateResiduals(fittedModel = dur_models[[best_dur]])
  
  p3 <- ggplot(mapping = aes(sample = dur_residuals)) +
    stat_qq_line(size = 1, col = "steelblue") +
    stat_qq(size = 2) +
    labs(title = paste("Q-Q Plot:", best_dur, "Duration"), 
         x = "Theoretical Quantiles", y = "Sample Quantiles") +
    theme_classic()
  
  plot(dur_simres, main = paste("DHARMa Diagnostics:", best_dur, "Duration"))
}

if(best_freq %in% c("LMM", "GLMM_Gamma")) {
  freq_residuals <- residuals(freq_models[[best_freq]])
  freq_simres <- simulateResiduals(fittedModel = freq_models[[best_freq]])
  
  p4 <- ggplot(mapping = aes(sample = freq_residuals)) +
    stat_qq_line(size = 1, col = "steelblue") +
    stat_qq(size = 2) +
    labs(title = paste("Q-Q Plot:", best_freq, "Frequency"), 
         x = "Theoretical Quantiles", y = "Sample Quantiles") +
    theme_classic()
  
  plot(freq_simres, main = paste("DHARMa Diagnostics:", best_freq, "Frequency"))
}

# Results from best models
if(best_dur %in% c("LMM", "GLMM_Gamma")) {
  cat("\n=== Duration Model Results ===\n")
  print(summary(dur_models[[best_dur]]))
  print(anova(dur_models[[best_dur]]))
  
  dur_emmeans <- emmeans(dur_models[[best_dur]], ~ Group|Session, type = "response")
  print(dur_emmeans)
}

if(best_freq %in% c("LMM", "GLMM_Gamma")) {
  cat("\n=== Frequency Model Results ===\n")
  print(summary(freq_models[[best_freq]]))
  print(anova(freq_models[[best_freq]]))
  
  freq_emmeans <- emmeans(freq_models[[best_freq]], ~ Group|Session, type = "response")
  print(freq_emmeans)
}

# Save plots
ggsave("figures/glmm/session_duration_boxplot.png", p1, width = 8, height = 6, dpi = 300)
ggsave("figures/glmm/session_frequency_boxplot.png", p2, width = 8, height = 6, dpi = 300)

if(exists("p3")) ggsave("figures/glmm/session_duration_qq.png", p3, width = 6, height = 6, dpi = 300)
if(exists("p4")) ggsave("figures/glmm/session_frequency_qq.png", p4, width = 6, height = 6, dpi = 300)
