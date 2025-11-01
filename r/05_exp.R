# Exploration Preference Analysis: Object Preferences Across Sessions
# Group × Session × Object analysis

suppressPackageStartupMessages({
  library(ggplot2)
  library(tidyr)
  library(dplyr)
  library(emmeans)
  library(lme4)
  library(lmerTest)
  library(DHARMa)
})

# Load data
s1 <- read.csv("data/spss/S1_exp.csv", header=TRUE, na.strings = "NA", sep=";", dec=",")
s2 <- read.csv("data/spss/S2_exp.csv", header=TRUE, na.strings = "NA", sep=";", dec=",")
t_data <- read.csv("data/spss/T_exp.csv", header=TRUE, na.strings = "NA", sep=";", dec=",")

# Prepare S1: 4 objects (A, B, C, D)
s1_long <- s1 %>%
  select(Video, GRUPO, A, B, C, D) %>%
  mutate(Rat_code = gsub("S1$", "", Video)) %>%
  pivot_longer(cols = c(A, B, C, D), names_to = "Object", values_to = "Time") %>%
  mutate(
    Session = "S1",
    Rat = as.factor(Rat_code)
  )

# Prepare S2: 2 objects (1, 2) - using STAT_AVG (static time) as preference measure
s2_long <- s2 %>%
  select(Video, GRUPO, STAT_1, STAT_2) %>%
  mutate(Rat_code = gsub("S2$", "", Video)) %>%
  pivot_longer(cols = c(STAT_1, STAT_2), names_to = "Object_raw", values_to = "Time") %>%
  mutate(
    Object = case_when(
      Object_raw == "STAT_1" ~ "1",
      Object_raw == "STAT_2" ~ "2"
    ),
    Session = "S2",
    Rat = as.factor(Rat_code)
  ) %>%
  select(-Object_raw)

# Prepare T: 4 objects (A1, A2, B1, B2) - keeping them separate
t_long <- t_data %>%
  select(ANIMAL, GRUPO, AUTO_A1, AUTO_A2, AUTO_B1, AUTO_B2) %>%
  mutate(Rat_code = ANIMAL) %>%
  pivot_longer(cols = c(AUTO_A1, AUTO_A2, AUTO_B1, AUTO_B2), names_to = "Object_raw", values_to = "Time") %>%
  mutate(
    Object = case_when(
      Object_raw == "AUTO_A1" ~ "A1",
      Object_raw == "AUTO_A2" ~ "A2",
      Object_raw == "AUTO_B1" ~ "B1",
      Object_raw == "AUTO_B2" ~ "B2"
    ),
    Session = "T",
    Rat = as.factor(Rat_code)
  ) %>%
  select(-Object_raw)

# Combine all sessions
data_long <- bind_rows(s1_long, s2_long, t_long) %>%
  mutate(
    Session = as.factor(Session),
    Object = as.factor(Object),
    Group = as.factor(GRUPO),
    Time = as.numeric(Time)
  )

levels(data_long$Group) <- c("Saline", "Muscimol")
levels(data_long$Session) <- c("S1", "S2", "T")

# Plot 1: Time by Object and Session
p1 <- ggplot(data_long, aes(x = Object, y = Time, fill = Group)) +
  geom_boxplot(position = position_dodge(0.8)) +
  facet_wrap(~Session, scales = "free_x") +
  scale_fill_manual(values = c("Saline" = "#145faa", "Muscimol" = "#828287")) +
  labs(title = "Exploration Time by Object, Session, and Group", 
       x = "Object", y = "Time (s)") +
  theme_classic()

# Plot 2: Mean preference patterns
p2 <- data_long %>%
  group_by(Session, Object, Group) %>%
  summarize(Mean_Time = mean(Time, na.rm = TRUE), .groups = "drop") %>%
  ggplot(aes(x = Object, y = Mean_Time, color = Group, group = Group)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3) +
  facet_wrap(~Session) +
  scale_color_manual(values = c("Saline" = "#145faa", "Muscimol" = "#828287")) +
  labs(title = "Mean Exploration Time Patterns", 
       x = "Object", y = "Mean Time (s)") +
  theme_classic()

# Statistical models - simplified to avoid rank deficiency (objects differ by session)
formula_null <- Time ~ 1 + (1|Rat)
formula_full <- Time ~ Group + Session + Object + Group:Session + Group:Object + (1|Rat)

# LMM
lmm_full <- tryCatch(lmer(formula_full, data = data_long), error = function(e) NULL)
lmm_null <- tryCatch(lmer(formula_null, data = data_long), error = function(e) NULL)

# GLMM Gamma
glmm_full <- tryCatch(glmer(formula_full, data = data_long, family = Gamma(link = "log")), error = function(e) NULL)
glmm_null <- tryCatch(glmer(formula_null, data = data_long, family = Gamma(link = "log")), error = function(e) NULL)

# Model comparison
mixed_models <- list("LMM" = lmm_full, "GLMM_Gamma" = glmm_full)
valid_mixed <- mixed_models[!sapply(mixed_models, is.null)]

aic_mixed <- sapply(valid_mixed, function(m) {
  if(!is.null(m)) tryCatch(AIC(m), error = function(e) NA) else NA
})

best_model <- if(length(valid_mixed) > 0 && !all(is.na(aic_mixed))) {
  names(valid_mixed)[which.min(aic_mixed)]
} else {
  "LMM"
}

# Diagnostics for best mixed model
if(best_model %in% c("LMM", "GLMM_Gamma")) {
  best_m <- mixed_models[[best_model]]
  residuals_val <- residuals(best_m)
  simres <- simulateResiduals(fittedModel = best_m)
  
  p3 <- ggplot(mapping = aes(sample = residuals_val)) +
    stat_qq_line(linewidth = 1, col = "steelblue") +
    stat_qq(size = 2) +
    labs(title = paste("Q-Q Plot:", best_model), 
         x = "Theoretical Quantiles", y = "Sample Quantiles") +
    theme_classic()
  
  plot(simres, main = paste("DHARMa Diagnostics:", best_model))
}

# Results from best model
if(best_model %in% c("LMM", "GLMM_Gamma")) {
  best_m <- mixed_models[[best_model]]
  print(summary(best_m))
  print(anova(best_m))
  
  # Within-session object preferences (only for objects that exist in that session)
  for(sess in c("S1", "S2", "T")) {
    sess_data <- data_long %>% filter(Session == sess)
    if(nrow(sess_data) > 0 && length(unique(sess_data$Object)) > 1) {
      formula_sess <- Time ~ Group + Object + Group:Object + (1|Rat)
      model_sess <- tryCatch({
        if(best_model == "LMM") {
          lmer(formula_sess, data = sess_data)
        } else {
          glmer(formula_sess, data = sess_data, family = Gamma(link = "log"))
        }
      }, error = function(e) NULL)
      
      if(!is.null(model_sess)) {
        cat("\n=== Object Preferences in", sess, "===\n")
        obj_emmeans <- emmeans(model_sess, ~ Object|Group, type = "response")
        print(obj_emmeans)
        obj_pairs <- pairs(obj_emmeans, adjust = "tukey")
        print(obj_pairs)
      }
    }
  }
}

# Save plots
ggsave("figures/glmm/exploration_by_object.png", p1, width = 12, height = 6, dpi = 300)
ggsave("figures/glmm/exploration_patterns.png", p2, width = 12, height = 6, dpi = 300)
if(exists("p3")) ggsave("figures/glmm/exploration_qq.png", p3, width = 6, height = 6, dpi = 300)

