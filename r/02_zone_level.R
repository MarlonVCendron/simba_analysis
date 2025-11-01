# Zone-Level Rearing Analysis: Group × Session × Zone
# Spatial rearing patterns across different experimental phases

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

# Data preparation
s1_data <- read.csv("data/spss/S1_rearing.csv", header=TRUE, na.strings = "NA", sep=";", dec=",")
s2_data <- read.csv("data/spss/S2_rearing.csv", header=TRUE, na.strings = "NA", sep=";", dec=",")
t_data <- read.csv("data/spss/T_rearing.csv", header=TRUE, na.strings = "NA", sep=";", dec=",")

# S1: OBJ vs NO_OBJ zones
s1_long <- s1_data %>%
  select(Video, GRUPO, OBJ_1_DUR, OBJ_2_DUR, OBJ_3_DUR, OBJ_4_DUR, 
         NO_OBJ_1_DUR, NO_OBJ_2_DUR, NO_OBJ_3_DUR, NO_OBJ_4_DUR) %>%
  pivot_longer(cols = c(OBJ_1_DUR, OBJ_2_DUR, OBJ_3_DUR, OBJ_4_DUR), 
               names_to = "Zone", values_to = "Duration") %>%
  mutate(Zone = "OBJ") %>%
  bind_rows(
    s1_data %>%
      select(Video, GRUPO, OBJ_1_DUR, OBJ_2_DUR, OBJ_3_DUR, OBJ_4_DUR, 
             NO_OBJ_1_DUR, NO_OBJ_2_DUR, NO_OBJ_3_DUR, NO_OBJ_4_DUR) %>%
      pivot_longer(cols = c(NO_OBJ_1_DUR, NO_OBJ_2_DUR, NO_OBJ_3_DUR, NO_OBJ_4_DUR), 
                   names_to = "Zone", values_to = "Duration") %>%
      mutate(Zone = "NO_OBJ")
  ) %>%
  mutate(Session = "S1", Zone = as.factor(Zone))

# S2: Former, Novel, Same, Never zones
s2_long <- s2_data %>%
  select(ANIMAL, GRUPO, FORMER_1_DUR, FORMER_2_DUR, NOVEL_1_DUR, NOVEL_2_DUR, 
         SAME_1_DUR, SAME_2_DUR, NEVER_1_DUR, NEVER_2_DUR) %>%
  pivot_longer(cols = c(FORMER_1_DUR, FORMER_2_DUR), names_to = "Zone", values_to = "Duration") %>%
  mutate(Zone = "FORMER") %>%
  bind_rows(
    s2_data %>%
      select(ANIMAL, GRUPO, FORMER_1_DUR, FORMER_2_DUR, NOVEL_1_DUR, NOVEL_2_DUR, 
             SAME_1_DUR, SAME_2_DUR, NEVER_1_DUR, NEVER_2_DUR) %>%
      pivot_longer(cols = c(NOVEL_1_DUR, NOVEL_2_DUR), names_to = "Zone", values_to = "Duration") %>%
      mutate(Zone = "NOVEL")
  ) %>%
  bind_rows(
    s2_data %>%
      select(ANIMAL, GRUPO, FORMER_1_DUR, FORMER_2_DUR, NOVEL_1_DUR, NOVEL_2_DUR, 
             SAME_1_DUR, SAME_2_DUR, NEVER_1_DUR, NEVER_2_DUR) %>%
      pivot_longer(cols = c(SAME_1_DUR, SAME_2_DUR), names_to = "Zone", values_to = "Duration") %>%
      mutate(Zone = "SAME")
  ) %>%
  bind_rows(
    s2_data %>%
      select(ANIMAL, GRUPO, FORMER_1_DUR, FORMER_2_DUR, NOVEL_1_DUR, NOVEL_2_DUR, 
             SAME_1_DUR, SAME_2_DUR, NEVER_1_DUR, NEVER_2_DUR) %>%
      pivot_longer(cols = c(NEVER_1_DUR, NEVER_2_DUR), names_to = "Zone", values_to = "Duration") %>%
      mutate(Zone = "NEVER")
  ) %>%
  mutate(Session = "S2", Zone = as.factor(Zone))

# T: A1, A2, B1, B2, Former, Never zones
t_long <- t_data %>%
  select(ANIMAL, GRUPO, A1_DUR, A2_DUR, B1_DUR, B2_DUR, FORMER_DUR, NEVER_1_DUR, NEVER_2_DUR, NEVER_3_DUR) %>%
  pivot_longer(cols = c(A1_DUR, A2_DUR), names_to = "Zone", values_to = "Duration") %>%
  mutate(Zone = "A") %>%
  bind_rows(
    t_data %>%
      select(ANIMAL, GRUPO, A1_DUR, A2_DUR, B1_DUR, B2_DUR, FORMER_DUR, NEVER_1_DUR, NEVER_2_DUR, NEVER_3_DUR) %>%
      pivot_longer(cols = c(B1_DUR, B2_DUR), names_to = "Zone", values_to = "Duration") %>%
      mutate(Zone = "B")
  ) %>%
  bind_rows(
    t_data %>%
      select(ANIMAL, GRUPO, A1_DUR, A2_DUR, B1_DUR, B2_DUR, FORMER_DUR, NEVER_1_DUR, NEVER_2_DUR, NEVER_3_DUR) %>%
      pivot_longer(cols = c(FORMER_DUR), names_to = "Zone", values_to = "Duration") %>%
      mutate(Zone = "FORMER")
  ) %>%
  bind_rows(
    t_data %>%
      select(ANIMAL, GRUPO, A1_DUR, A2_DUR, B1_DUR, B2_DUR, FORMER_DUR, NEVER_1_DUR, NEVER_2_DUR, NEVER_3_DUR) %>%
      pivot_longer(cols = c(NEVER_1_DUR, NEVER_2_DUR, NEVER_3_DUR), names_to = "Zone", values_to = "Duration") %>%
      mutate(Zone = "NEVER")
  ) %>%
  mutate(Session = "T", Zone = as.factor(Zone))

# Combine all sessions
zone_data <- bind_rows(s1_long, s2_long, t_long) %>%
  mutate(
    Group = as.factor(GRUPO),
    Session = as.factor(Session),
    Rat = as.factor(ifelse(Session == "S1", Video, ANIMAL)),
    Rat_id = as.numeric(as.factor(ifelse(Session == "S1", Video, ANIMAL))),
    Duration = Duration + 0.001  # Add small constant for Gamma models
  )

levels(zone_data$Group) <- c("Saline", "Muscimol")
levels(zone_data$Session) <- c("S1", "S2", "T")

# Debug: Check data structure
cat("Zone data dimensions:", dim(zone_data), "\n")
cat("Zone data summary:\n")
print(str(zone_data))
cat("Zone levels:", levels(zone_data$Zone), "\n")
cat("Session-Zone combinations:\n")
print(table(zone_data$Session, zone_data$Zone))

# Exploratory plots
p1 <- ggplot(zone_data, aes(x = Zone, y = Duration, fill = Group)) +
  geom_boxplot(position = position_dodge(0.8)) +
  facet_wrap(~Session, scales = "free_x") +
  scale_fill_manual(values = c("Saline" = "#145faa", "Muscimol" = "#828287")) +
  labs(title = "Rearing Duration by Group, Session, and Zone", 
       x = "Zone", y = "Duration (s)") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Per-session models to avoid empty Session×Zone cells
sessions <- c("S1","S2","T")
zone_sets <- list(
  S1 = c("NO_OBJ","OBJ"),
  S2 = c("FORMER","NEVER","NOVEL","SAME"),
  T  = c("A","B","FORMER","NEVER")
)

fit_one <- function(df) {
  df <- droplevels(df)
  df <- as.data.frame(df)  # Convert tibble to data.frame
  form_glmm <- Duration ~ Group + Zone + Group:Zone + (1|Rat)
  m_glmm <- tryCatch(
    glmer(form_glmm, data=df, family=Gamma(link="log"),
          control=glmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5))),
    error=function(e) NULL
  )
  m_gee <- tryCatch(
    geeglm(Duration ~ Group + Zone + Group:Zone, data=df, id=Rat_id,
           family=Gamma("log"), corstr="ar1"),
    error=function(e) NULL
  )
  list(glmm=m_glmm, gee=m_gee)
}

for (sess in sessions) {
  cat("\n=== Session", sess, "===\n")
  zs <- zone_sets[[sess]]
  d <- zone_data %>% dplyr::filter(Session == sess, Zone %in% zs)
  d$Zone <- droplevels(d$Zone)

  models <- fit_one(d)

  # Compare (AIC for GLMM, QIC for GEE)
  aic <- if (!is.null(models$glmm)) AIC(models$glmm) else NA
  qic <- if (!is.null(models$gee)) {
    tryCatch(QIC(models$gee)[1], error = function(e) NA)
  } else NA
  print(data.frame(Session=sess, GLMM_Gamma_AIC=aic, GEE_Gamma_QIC=qic))

  # Choose best by whichever metric is available/prefer GLMM if both present
  best <- if (!is.na(aic)) "glmm" else if (!is.na(qic)) "gee" else NA

  if (identical(best,"glmm")) {
    cat("Best:", best, "\n")
    print(summary(models$glmm))
    print(anova(models$glmm))
    # Effects
    cat("EMMEANS: Group differences within Zone\n")
    print(emmeans(models$glmm, pairwise ~ Group | Zone, type="response"))
    cat("EMMEANS: Zone differences within Group\n")
    print(emmeans(models$glmm, pairwise ~ Zone | Group, type="response"))
  } else if (identical(best,"gee")) {
    cat("Best:", best, "\n")
    print(summary(models$gee))
    # For GEE, use emmeans with caution (population-average)
    cat("EMMEANS (GEE): Group within Zone\n")
    print(emmeans(models$gee, pairwise ~ Group | Zone, type="response"))
    cat("EMMEANS (GEE): Zone within Group\n")
    print(emmeans(models$gee, pairwise ~ Zone | Group, type="response"))
  } else {
    cat("No estimable model for session", sess, "\n")
  }
}

# Save plots
ggsave("figures/glmm/zone_duration_boxplot.png", p1, width = 12, height = 8, dpi = 300)
if(exists("p2")) ggsave("figures/glmm/zone_duration_qq.png", p2, width = 6, height = 6, dpi = 300)
