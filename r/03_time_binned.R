# Time-Binned Rearing Analysis: Group × Time
# Temporal dynamics of rearing behavior within sessions

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
time_data <- read.csv("data/processed/time_bins.csv", header=TRUE, na.strings = "NA")

# Prepare data
time_data <- time_data %>%
  mutate(
    Group = as.factor(ifelse(group == "salina", "Saline", "Muscimol")),
    Session = as.factor(toupper(session)),
    Rat = as.factor(video),
    Rat_id = as.numeric(as.factor(video)),
    Time = second,
    Rearing = rearing + 0.001  # Add small constant for Gamma models
  ) %>%
  select(Rat, Group, Session, Time, Rearing, Rat_id)

# Filter to main sessions
time_data <- time_data %>%
  filter(Session %in% c("S1", "S2", "T"))

# Exploratory plots
p1 <- ggplot(time_data, aes(x = Time, y = Rearing, color = Group)) +
  geom_smooth(method = "loess", se = TRUE) +
  facet_wrap(~Session, scales = "free") +
  scale_color_manual(values = c("Saline" = "#145faa", "Muscimol" = "#828287")) +
  labs(title = "Rearing Over Time by Group and Session", 
       x = "Time (seconds)", y = "Rearing") +
  theme_classic()

p2 <- ggplot(time_data, aes(x = Time, y = Rearing, fill = Group)) +
  geom_boxplot(aes(group = interaction(Group, cut(Time, breaks = 10))), 
               position = position_dodge(0.8)) +
  facet_wrap(~Session, scales = "free") +
  scale_fill_manual(values = c("Saline" = "#145faa", "Muscimol" = "#828287")) +
  labs(title = "Rearing Distribution Over Time", 
       x = "Time (seconds)", y = "Rearing") +
  theme_classic()

# Models - separate for each session
sessions <- c("S1", "S2", "T")
time_models <- list()
time_comparisons <- list()

for(sess in sessions) {
  sess_data <- time_data %>% filter(Session == sess)
  
  formula_time <- Rearing ~ Group + Time + Group:Time + (1|Rat)
  formula_time_gee <- Rearing ~ Group + Time + Group:Time
  
  # LMM
  lmm_time <- lmer(formula_time, data = sess_data)
  
  # GLMM Gamma
  glmm_time <- glmer(formula_time, data = sess_data, family = Gamma(link = "log"))
  
  # GEE Normal
  gee_time_norm <- geeglm(formula_time_gee, data = sess_data, id = Rat_id, 
                          family = gaussian("identity"), corstr = "ar1")
  
  # GEE Gamma
  gee_time_gamma <- geeglm(formula_time_gee, data = sess_data, id = Rat_id, 
                           family = Gamma("log"), corstr = "ar1")
  
  # Store models
  time_models[[sess]] <- list(
    "LMM" = lmm_time,
    "GLMM_Gamma" = glmm_time,
    "GEE_Normal" = gee_time_norm,
    "GEE_Gamma" = gee_time_gamma
  )
  
  # Model comparison
  valid_models <- time_models[[sess]][!sapply(time_models[[sess]], is.null)]
  
  if(length(valid_models) == 0) {
    comparison <- data.frame(Model = "None", AIC = NA, BIC = NA, QIC = NA, Session = sess)
  } else {
    n_models <- length(valid_models)
    comparison <- data.frame(
      Model = names(valid_models),
      AIC = rep(NA, n_models),
      BIC = rep(NA, n_models),
      QIC = rep(NA, n_models),
      Session = sess,
      stringsAsFactors = FALSE
    )
    
    # Fill in values
    for(i in 1:n_models) {
      model <- valid_models[[i]]
      if(!is.null(model)) {
        aic_val <- tryCatch(AIC(model), error = function(e) NA)
        bic_val <- tryCatch(BIC(model), error = function(e) NA)
        qic_val <- tryCatch(QIC(model), error = function(e) NA)
        
        comparison$AIC[i] <- if(length(aic_val) > 0 && !any(is.na(aic_val))) aic_val[1] else NA
        comparison$BIC[i] <- if(length(bic_val) > 0 && !any(is.na(bic_val))) bic_val[1] else NA
        comparison$QIC[i] <- if(length(qic_val) > 0 && !any(is.na(qic_val))) qic_val[1] else NA
      }
    }
  }
  time_comparisons[[sess]] <- comparison
  
  cat("\n=== Session", sess, "Model Comparison ===\n")
  print(comparison)
}

# Best models for each session
best_models <- sapply(sessions, function(sess) {
  comparison <- time_comparisons[[sess]]
  valid_aic <- comparison$AIC[!is.na(comparison$AIC)]
  if(length(valid_aic) > 0) {
    best_idx <- which.min(comparison$AIC)
    if(length(best_idx) > 0 && !is.na(best_idx)) {
      return(comparison$Model[best_idx])
    }
  }
  return("None")
})

cat("\nBest models by session:\n")
for(i in 1:length(sessions)) {
  cat(sessions[i], ":", best_models[i], "\n")
}

# Diagnostics for best models
for(sess in sessions) {
  best_model_name <- best_models[sess]
  best_model <- time_models[[sess]][[best_model_name]]
  
  if(best_model_name %in% c("LMM", "GLMM_Gamma")) {
    residuals <- residuals(best_model)
    simres <- simulateResiduals(fittedModel = best_model)
    
    p_diag <- ggplot(mapping = aes(sample = residuals)) +
      stat_qq_line(size = 1, col = "steelblue") +
      stat_qq(size = 2) +
      labs(title = paste("Q-Q Plot:", best_model_name, sess), 
           x = "Theoretical Quantiles", y = "Sample Quantiles") +
      theme_classic()
    
    ggsave(paste0("figures/glmm/time_", tolower(sess), "_qq.png"), 
           p_diag, width = 6, height = 6, dpi = 300)
    
    png(paste0("figures/glmm/time_", tolower(sess), "_dharMa.png"), 
        width = 800, height = 600, res = 100)
    plot(simres, main = paste("DHARMa Diagnostics:", best_model_name, sess))
    dev.off()
  }
}

# Results from best models
for(sess in sessions) {
  best_model_name <- best_models[sess]
  best_model <- time_models[[sess]][[best_model_name]]
  
  cat("\n=== Session", sess, "Results (", best_model_name, ") ===\n")
  
  if(best_model_name %in% c("LMM", "GLMM_Gamma")) {
    print(summary(best_model))
    print(anova(best_model))
    
    # Group differences at specific times (avoids nonEst)
    time_grid <- seq(0, max(time_data$Time[time_data$Session == sess]), by = 60)
    group_time_emmeans <- tryCatch(
      emmeans(best_model, ~ Group | Time, type = "response", at = list(Time = time_grid)),
      error = function(e) NULL
    )
    if(!is.null(group_time_emmeans)) print(group_time_emmeans)
    
    # Compare time trends (slopes) between groups for numeric Time
    trend_contrasts <- tryCatch(
      emtrends(best_model, pairwise ~ Group, var = "Time"),
      error = function(e) NULL
    )
    if(!is.null(trend_contrasts)) print(trend_contrasts)
  }
}

# Combined analysis across all sessions
formula_combined <- Rearing ~ Group + Session + Time + Group:Session + Group:Time + Session:Time + Group:Session:Time + (1|Rat)
formula_combined_gee <- Rearing ~ Group + Session + Time + Group:Session + Group:Time + Session:Time + Group:Session:Time

# LMM
lmm_combined <- lmer(formula_combined, data = time_data)

# GLMM Gamma
glmm_combined <- glmer(formula_combined, data = time_data, family = Gamma(link = "log"))

# GEE Normal
gee_combined_norm <- geeglm(formula_combined_gee, data = time_data, id = Rat_id, 
                            family = gaussian("identity"), corstr = "ar1")

# GEE Gamma
gee_combined_gamma <- geeglm(formula_combined_gee, data = time_data, id = Rat_id, 
                             family = Gamma("log"), corstr = "ar1")

# Combined model comparison
combined_models <- list(
  "LMM" = lmm_combined,
  "GLMM_Gamma" = glmm_combined,
  "GEE_Normal" = gee_combined_norm,
  "GEE_Gamma" = gee_combined_gamma
)

# Combined model comparison
valid_combined_models <- combined_models[!sapply(combined_models, is.null)]

if(length(valid_combined_models) == 0) {
  combined_comparison <- data.frame(Model = "None", AIC = NA, BIC = NA, QIC = NA)
} else {
  n_models <- length(valid_combined_models)
  combined_comparison <- data.frame(
    Model = names(valid_combined_models),
    AIC = rep(NA, n_models),
    BIC = rep(NA, n_models),
    QIC = rep(NA, n_models),
    stringsAsFactors = FALSE
  )
  
  # Fill in values
  for(i in 1:n_models) {
    model <- valid_combined_models[[i]]
    if(!is.null(model)) {
      aic_val <- tryCatch(AIC(model), error = function(e) NA)
      bic_val <- tryCatch(BIC(model), error = function(e) NA)
      qic_val <- tryCatch(QIC(model), error = function(e) NA)
      
      combined_comparison$AIC[i] <- if(length(aic_val) > 0 && !any(is.na(aic_val))) aic_val[1] else NA
      combined_comparison$BIC[i] <- if(length(bic_val) > 0 && !any(is.na(bic_val))) bic_val[1] else NA
      combined_comparison$QIC[i] <- if(length(qic_val) > 0 && !any(is.na(qic_val))) qic_val[1] else NA
    }
  }
}

cat("\n=== Combined Model Comparison ===\n")
print(combined_comparison)

valid_aic <- combined_comparison$AIC[!is.na(combined_comparison$AIC)]
if(length(valid_aic) > 0) {
  best_combined_idx <- which.min(combined_comparison$AIC)
  if(length(best_combined_idx) > 0 && !is.na(best_combined_idx)) {
    best_combined <- combined_comparison$Model[best_combined_idx]
  } else {
    best_combined <- "None"
  }
} else {
  best_combined <- "None"
}
cat("\nBest combined model:", best_combined, "\n")

# Save plots
ggsave("figures/glmm/time_trends.png", p1, width = 12, height = 8, dpi = 300)
ggsave("figures/glmm/time_distribution.png", p2, width = 12, height = 8, dpi = 300)
