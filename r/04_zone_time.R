# Zone-Time Rearing Analysis: Group × Zone × Time
# Full spatiotemporal dynamics of rearing behavior

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

# Prepare data with zone information
zone_time_data <- time_data %>%
  mutate(
    Group = as.factor(ifelse(group == "salina", "Saline", "Muscimol")),
    Session = as.factor(toupper(session)),
    Rat = as.factor(video),
    Rat_id = as.numeric(as.factor(video)),
    Time = second,
    Rearing = rearing + 0.001
  ) %>%
  filter(Session %in% c("S1", "S2", "T"))

# Create zone variables based on session
zone_time_data <- zone_time_data %>%
  mutate(
    Zone = case_when(
      Session == "S1" & (OBJ_1 > 0 | OBJ_2 > 0 | OBJ_3 > 0 | OBJ_4 > 0) ~ "OBJ",
      Session == "S1" & (NO_OBJ_1 > 0 | NO_OBJ_2 > 0 | NO_OBJ_3 > 0 | NO_OBJ_4 > 0) ~ "NO_OBJ",
      Session == "S2" & (FORMER_1 > 0 | FORMER_2 > 0) ~ "FORMER",
      Session == "S2" & (NOVEL_1 > 0 | NOVEL_2 > 0) ~ "NOVEL",
      Session == "S2" & (SAME_1 > 0 | SAME_2 > 0) ~ "SAME",
      Session == "S2" & (NEVER_1 > 0 | NEVER_2 > 0) ~ "NEVER",
      Session == "T" & (A1 > 0 | A2 > 0) ~ "A",
      Session == "T" & (B1 > 0 | B2 > 0) ~ "B",
      Session == "T" & FORMER > 0 ~ "FORMER",
      Session == "T" & (NEVER_1 > 0 | NEVER_2 > 0 | NEVER_3 > 0) ~ "NEVER",
      TRUE ~ "OTHER"
    )
  ) %>%
  filter(Zone != "OTHER") %>%
  select(Rat, Group, Session, Time, Zone, Rearing, Rat_id) %>%
  mutate(Zone = as.factor(Zone))

# Exploratory plots
p1 <- ggplot(zone_time_data, aes(x = Time, y = Rearing, color = Group)) +
  geom_smooth(method = "loess", se = TRUE) +
  facet_grid(Zone ~ Session, scales = "free") +
  scale_color_manual(values = c("Saline" = "#145faa", "Muscimol" = "#828287")) +
  labs(title = "Rearing Over Time by Group, Session, and Zone", 
       x = "Time (seconds)", y = "Rearing") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p2 <- ggplot(zone_time_data, aes(x = Zone, y = Rearing, fill = Group)) +
  geom_boxplot(position = position_dodge(0.8)) +
  facet_wrap(~Session, scales = "free_x") +
  scale_fill_manual(values = c("Saline" = "#145faa", "Muscimol" = "#828287")) +
  labs(title = "Rearing by Zone, Group, and Session", 
       x = "Zone", y = "Rearing") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Models - separate for each session due to different zone structures
sessions <- c("S1", "S2", "T")
zone_time_models <- list()
zone_time_comparisons <- list()

for(sess in sessions) {
  sess_data <- zone_time_data %>% filter(Session == sess)
  sess_data <- droplevels(sess_data)
  sess_data <- as.data.frame(sess_data)
  
  formula_zone_time <- Rearing ~ Group + Zone + Time + Group:Zone + Group:Time + Zone:Time + Group:Zone:Time + (1|Rat)
  formula_zone_time_gee <- Rearing ~ Group + Zone + Time + Group:Zone + Group:Time + Zone:Time
  
  # LMM
  lmm_zone_time <- tryCatch(lmer(formula_zone_time, data = sess_data), error = function(e) NULL)
  
  # GLMM Gamma
  glmm_zone_time <- tryCatch(glmer(formula_zone_time, data = sess_data, family = Gamma(link = "log")), error = function(e) NULL)
  
  # GEE Normal
  gee_zone_time_norm <- tryCatch(
    geeglm(formula_zone_time_gee, data = sess_data, id = Rat_id, family = gaussian("identity"), corstr = "ar1"),
    error = function(e) NULL
  )
  
  # GEE Gamma
  gee_zone_time_gamma <- tryCatch(
    geeglm(formula_zone_time_gee, data = sess_data, id = Rat_id, family = Gamma("log"), corstr = "ar1"),
    error = function(e) NULL
  )
  
  # Store models
  zone_time_models[[sess]] <- list(
    "LMM" = lmm_zone_time,
    "GLMM_Gamma" = glmm_zone_time,
    "GEE_Normal" = gee_zone_time_norm,
    "GEE_Gamma" = gee_zone_time_gamma
  )
  
  # Model comparison (robust)
  valid_models <- zone_time_models[[sess]][!sapply(zone_time_models[[sess]], is.null)]
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
    for(i in 1:n_models) {
      model <- valid_models[[i]]
      aic_val <- tryCatch(AIC(model), error = function(e) NA)
      bic_val <- tryCatch(BIC(model), error = function(e) NA)
      qic_val <- tryCatch(QIC(model), error = function(e) NA)
      comparison$AIC[i] <- if(length(aic_val) > 0 && !any(is.na(aic_val))) aic_val[1] else NA
      comparison$BIC[i] <- if(length(bic_val) > 0 && !any(is.na(bic_val))) bic_val[1] else NA
      comparison$QIC[i] <- if(length(qic_val) > 0 && !any(is.na(qic_val))) qic_val[1] else NA
    }
  }
  zone_time_comparisons[[sess]] <- comparison
  
  cat("\n=== Session", sess, "Zone-Time Model Comparison ===\n")
  print(comparison)
}

# Best models for each session
best_zone_time_models <- sapply(sessions, function(sess) {
  comparison <- zone_time_comparisons[[sess]]
  valid_aic <- comparison$AIC[!is.na(comparison$AIC)]
  if(length(valid_aic) > 0) {
    best_idx <- which.min(comparison$AIC)
    if(length(best_idx) > 0 && !is.na(best_idx)) return(comparison$Model[best_idx])
  }
  return("None")
})

cat("\nBest zone-time models by session:\n")
for(i in 1:length(sessions)) {
  cat(sessions[i], ":", best_zone_time_models[i], "\n")
}

# Diagnostics for best models
for(sess in sessions) {
  best_model_name <- best_zone_time_models[sess]
  best_model <- zone_time_models[[sess]][[best_model_name]]
  
  if(best_model_name %in% c("LMM", "GLMM_Gamma") && !is.null(best_model)) {
    residuals <- residuals(best_model)
    simres <- simulateResiduals(fittedModel = best_model)
    
    p_diag <- ggplot(mapping = aes(sample = residuals)) +
      stat_qq_line(size = 1, col = "steelblue") +
      stat_qq(size = 2) +
      labs(title = paste("Q-Q Plot:", best_model_name, sess, "Zone-Time"), 
           x = "Theoretical Quantiles", y = "Sample Quantiles") +
      theme_classic()
    
    ggsave(paste0("figures/glmm/zone_time_", tolower(sess), "_qq.png"), 
           p_diag, width = 6, height = 6, dpi = 300)
    
    png(paste0("figures/glmm/zone_time_", tolower(sess), "_dharMa.png"), 
        width = 800, height = 600, res = 100)
    plot(simres, main = paste("DHARMa Diagnostics:", best_model_name, sess, "Zone-Time"))
    dev.off()
  }
}

# Results from best models
for(sess in sessions) {
  best_model_name <- best_zone_time_models[sess]
  best_model <- zone_time_models[[sess]][[best_model_name]]
  
  cat("\n=== Session", sess, "Zone-Time Results (", best_model_name, ") ===\n")
  
  if(best_model_name %in% c("LMM", "GLMM_Gamma") && !is.null(best_model)) {
    print(summary(best_model))
    print(anova(best_model))
    
    time_grid <- seq(0, max(zone_time_data$Time[zone_time_data$Session == sess]), by = 60)
    # Group differences within Zone over time (grid)
    group_zone_time_emmeans <- tryCatch(
      emmeans(best_model, ~ Group | Zone, type = "response", at = list(Time = time_grid)),
      error = function(e) NULL
    )
    if(!is.null(group_zone_time_emmeans)) print(group_zone_time_emmeans)
    
    # Time trends (slopes) by Group within Zone
    slope_contrasts <- tryCatch(
      emtrends(best_model, pairwise ~ Group | Zone, var = "Time"),
      error = function(e) NULL
    )
    if(!is.null(slope_contrasts)) print(slope_contrasts)
  }
}

# Combined analysis (simplified due to different zone structures)
# Focus on common zones across sessions
common_zones <- zone_time_data %>%
  group_by(Session, Zone) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(Zone) %>%
  summarise(sessions = n(), .groups = "drop") %>%
  filter(sessions == 3) %>%
  pull(Zone)

if(length(common_zones) > 0) {
  combined_data <- zone_time_data %>%
    filter(Zone %in% common_zones)
  
  formula_combined <- Rearing ~ Group + Session + Zone + Time + Group:Session + Group:Zone + Group:Time + Session:Zone + Session:Time + Zone:Time + Group:Session:Zone + Group:Session:Time + Group:Zone:Time + Session:Zone:Time + Group:Session:Zone:Time + (1|Rat)
  formula_combined_gee <- Rearing ~ Group + Session + Zone + Time + Group:Session + Group:Zone + Group:Time + Session:Zone + Session:Time + Zone:Time + Group:Session:Zone + Group:Session:Time + Group:Zone:Time + Session:Zone:Time + Group:Session:Zone:Time
  
  # LMM
  lmm_combined <- lmer(formula_combined, data = combined_data)
  
  # GLMM Gamma
  glmm_combined <- glmer(formula_combined, data = combined_data, family = Gamma(link = "log"))
  
  # GEE Normal
  gee_combined_norm <- geeglm(formula_combined_gee, data = combined_data, id = Rat_id, 
                              family = gaussian("identity"), corstr = "ar1")
  
  # GEE Gamma
  gee_combined_gamma <- geeglm(formula_combined_gee, data = combined_data, id = Rat_id, 
                               family = Gamma("log"), corstr = "ar1")
  
  # Combined model comparison
  combined_models <- list(
    "LMM" = lmm_combined,
    "GLMM_Gamma" = glmm_combined,
    "GEE_Normal" = gee_combined_norm,
    "GEE_Gamma" = gee_combined_gamma
  )
  
  combined_comparison <- data.frame(
    Model = names(combined_models),
    AIC = sapply(combined_models, function(x) {
      tryCatch(AIC(x), error = function(e) NA)
    }),
    BIC = sapply(combined_models, function(x) {
      tryCatch(BIC(x), error = function(e) NA)
    }),
    QIC = sapply(combined_models, function(x) {
      tryCatch(QIC(x), error = function(e) NA)
    })
  )
  
  cat("\n=== Combined Zone-Time Model Comparison ===\n")
  print(combined_comparison)
  
  best_combined_idx <- which.min(combined_comparison$AIC)
  best_combined <- if(length(best_combined_idx) > 0 && !is.na(best_combined_idx)) names(combined_models)[best_combined_idx] else "LMM"
  cat("\nBest combined zone-time model:", best_combined, "\n")
}

# Save plots
ggsave("figures/glmm/zone_time_trends.png", p1, width = 16, height = 12, dpi = 300)
ggsave("figures/glmm/zone_time_distribution.png", p2, width = 12, height = 8, dpi = 300)
