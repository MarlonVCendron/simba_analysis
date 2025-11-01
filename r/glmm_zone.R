#---- R Packages Needed ------------------
rm(list=ls()) #Clears all objects from the workspace
options(repos=c(CRAN="https://vps.fmvz.usp.br/CRAN/"))

# Suppress package loading warnings
suppressPackageStartupMessages({
  if(!require(geepack, quietly = TRUE))install.packages("geepack")
  if(!require(ggplot2, quietly = TRUE))install.packages("ggplot2")
  if(!require(readr, quietly = TRUE))install.packages("readr")
  if(!require(tidyr, quietly = TRUE))install.packages("tidyr")
  if(!require(dplyr, quietly = TRUE))install.packages("dplyr")
  if(!require(emmeans, quietly = TRUE))install.packages("emmeans")
  if(!require(lme4, quietly = TRUE))install.packages("lme4")
  if(!require(lmerTest, quietly = TRUE))install.packages("lmerTest")
  if(!require(nlme, quietly = TRUE))install.packages("nlme")
  library(tidyr, quietly = TRUE)
  library(glmmTMB, quietly = TRUE)
  library(DHARMa, quietly = TRUE)
})

db = read.csv("data/processed/time_bins.csv", header=TRUE, na.strings = "NA")

db$group = as.factor(db$group)
# db$second <- as.factor(db$second)
db$zone <- NA
db$zone <- apply(db[, c("A1", "A2", "B1", "B2", "FORMER", "NEVER_1", "NEVER_2", "NEVER_3")], 1, function(row) {
  values <- c(
    "A1" = row["A1"],
    "A2" = row["A2"], 
    "B1" = row["B1"],
    "B2" = row["B2"],
    "Former" = row["FORMER"],
    "Never" = max(row[c("NEVER_1", "NEVER_2", "NEVER_3")])
  )
  names(which.max(values))
})
db$zone <- as.factor(db$zone)
names(db)[names(db) == 'video'] <- 'rat'
names(db)[names(db) == 'second'] <- 'time'
levels(db$group) = c("Saline", "Muscimol")

db$rearing <- db$rearing + 1

db$rat_id <- as.numeric(as.factor(db$rat))

print(str(db[, c("rearing", "group", "zone", "time", "rat", "rat_id")]))

formula <- formula(rearing ~ group + zone + time + (1|rat))

formula_gee <- formula(rearing ~ group + zone + time + group*time)

analyze_model <- function(model) {
    print(summary(model))
    print(anova(model))

    model_residuals <- residuals(model)
    # print(shapiro.test(model_residuals))

    # model_desc <- emmeans(model, ~ Session|Group, type = "response")
    # print(model_desc)

    plot <- ggplot(mapping = aes(sample = model_residuals)) +
        stat_qq_line (size = 2, col = "steelblue") +
        stat_qq(pch = 1, size = 6) +
        ggtitle(paste("QQ Plot of", deparse(substitute(model)))) +
        labs (x = "Theoretical Quantiles", y = "Sample Quantiles" ) +
        theme_classic() +
        theme(plot.title = element_text(hjust = 0.5, size = 25),
              text = element_text(size = 20),
              axis.text = element_text(colour = "#000000")) +
        guides(fill = "none")
    plot
}



# glmmNormal <- lmer(formula, data = db)
# analyze_model(glmmNormal)

# glmm2 <- glmer(formula, data = db, family = Gamma(link = "inverse"))
# glmm3 <- glmer(formula, data = db, family = Gamma(link = "log"))
# glmm4 <- glmer(formula, data = db, family = gaussian(link = "log"))
# glmm5 <- glmer(formula, data = db, family = inverse.gaussian(link = "1/mu^2"))
# glmm6 <- glmer(formula, data = db, family = gaussian(link = "inverse"))
# fit_nb <- glmmTMB(formula, data = db, family = nbinom2)
# fit_zinb <- glmmTMB(formula, data = db, ziformula=~1, family=nbinom2)

 
# print("#------- Model 2 - GLMM with Gamma distribution --------------------------------------#")
# glmmGammaInverse <- glmer(formula, data = db, family = Gamma(link = "inverse"))
# analyze_model(glmmGammaInverse)

# print("#------- Model 3 - GLMM with Gamma distribution with log link function --------------------------------------#")
# glmmGammaLog <- glmer(formula, data = db, family = Gamma(link = "log"))
# analyze_model(glmmGammaLog)

# print("#------- Model 4 - GLMM with Poisson distribution --------------------------------------#")
# glmmPoisson <- glmer(formula, data = db, family = poisson(link = "log"))
# analyze_model(glmmPoisson)

# print("#------- Model 5 - GLMM with Inverse Gaussian distribution --------------------------------------#")
# glmmInverseGaussian <- glmer(formula, data = db, family = inverse.gaussian(link = "1/mu^2"))
# analyze_model(glmmInverseGaussian)


# print("#------- Model 6 - GLMM with Binomial distribution --------------------------------------#")
# glmmBinomial <- glmer(formula, data = db, family = binomial(link = "logit"))
# analyze_model(glmmBinomial)

geeNormal <- geeglm(formula_gee, data = db, id = rat_id, family = gaussian("log"), corstr = "ar1")
# geeNormal <- geeglm(formula_gee, data = db, id = rat_id, family = Gamma("log"), corstr = "ar1")
analyze_model(geeNormal)

# models <- list(glmmNormal, glmm2, glmm3, glmm4, glmm5, glmm6)
# for (model in models) {
#     # print(paste("AIC:", AIC(model), "BIC:", BIC(model), "Log-Likelihood:", logLik(model)))

#     simres <- simulateResiduals(fittedModel = model)
#     plotSimulatedResiduals(simres)
# }



# models <- list(glmmBinomial)
# smallest_AIC <- Inf
# smallest_BIC <- Inf
# biggest_logLik <- -Inf
# for (model in models) {
#     if (AIC(model) < smallest_AIC) {
#         smallest_AIC <- AIC(model)
#         smallest_model <- model
#     }
#     if (BIC(model) < smallest_BIC) {
#         smallest_BIC <- BIC(model)
#         smallest_model <- model
#     }
#     if (logLik(model) > biggest_logLik) {
#         print(paste("logLik:", logLik(model)))
#         biggest_logLik <- logLik(model)
#         biggest_model <- model
#     }
# }
# print(smallest_model)

# # ranef(glmmNormal)
# # VarCorr(glmmNormal)
# # isSingular(glmmNormal, tol = 1e-4)   


# boxplot_plot <- ggplot(r_DB, aes(x = Session, y = Duration, fill = Group)) +
#   stat_boxplot(geom = "errorbar", width = 0.2, color = "black", position = position_dodge(width = 0.8)) +
#   geom_boxplot(position = position_dodge(width = 0.8)) +
#   scale_fill_manual(values = c("Saline" = "#145faa", "Muscimol" = "#828287")) +
#   labs(title = "Rearing Duration by Group and Session",
#        x = "Session", 
#        y = "Rearing Duration (seconds)",
#        fill = "Group") +
#   theme_classic() +
#   theme(plot.title = element_text(hjust = 0.5, size = 14), text = element_text(size = 12))
# print(boxplot_plot)

# model_emmeans <- emmeans(glmmNormal, ~ Group|Session, type = "response")
# pred_data <- as.data.frame(model_emmeans)
# pred_data$Session <- factor(pred_data$Session, levels = c("S1", "S2", "T"))

# prediction_plot <- ggplot(pred_data, aes(x = Session, y = emmean, color = Group, group = Group)) +
#   geom_line(size = 1) +
#   geom_point(size = 3) +
#   geom_errorbar(aes(ymin = lower.CL, ymax = upper.CL), width = 0.1) +
#   scale_color_manual(values = c("Saline" = "#145faa", "Muscimol" = "#828287")) +
#   labs(title = "Model Predictions: Rearing Duration",
#        x = "Session", 
#        y = "Predicted Rearing Duration (seconds)",
#        color = "Group") +
#   theme_classic() +
#   theme(plot.title = element_text(hjust = 0.5, size = 14),
#         text = element_text(size = 12))
# print(prediction_plot)

# ggsave("figures/glmm/boxplot_rearing_duration.png", boxplot_plot, width = 8, height = 6, dpi = 300)

# ggsave("figures/glmm/predictions_rearing_duration.png", prediction_plot, width = 8, height = 6, dpi = 300)
