#---- R Packages Needed ------------------
rm(list=ls()) #Clears all objects from the workspace
options(repos=c(CRAN="https://vps.fmvz.usp.br/CRAN/"))

if(!require(geepack))install.packages("geepack")
if(!require(ggplot2))install.packages("ggplot2")
if(!require(readr))install.packages("readr")
if(!require(tidyr))install.packages("tidyr")
if(!require(dplyr))install.packages("dplyr")
if(!require(emmeans))install.packages("emmeans")
if(!require(lme4))install.packages("lme4")
if(!require(lmerTest))install.packages("lmerTest")
if(!require(nlme))install.packages("nlme")
library(DHARMa)

# DB = read.csv("data/summary_data.csv", header=TRUE, na.strings = "NA")
# t_DB = read.csv("data/spss/T_exp.csv", header=TRUE, na.strings = "NA", sep=";", dec=",")
# s1_DB = read.csv("data/spss/S1_exp.csv", header=TRUE, na.strings = "NA", sep=";", dec=",")
# s2_DB = read.csv("data/spss/S2_exp.csv", header=TRUE, na.strings = "NA", sep=";", dec=",")
r_DB = read.csv("data/spss/rearing.csv", header=TRUE, na.strings = "NA", sep=";", dec=",")

library(tidyr)
r_DB <- r_DB %>%
  select(Video, GRUPO, DUR_S1, DUR_S2, DUR_T) %>%
  pivot_longer(cols = c(DUR_S1, DUR_S2, DUR_T), 
               names_to = "Session", 
               values_to = "Duration") %>%
  mutate(Session = case_when(
    Session == "DUR_S1" ~ "S1",
    Session == "DUR_S2" ~ "S2", 
    Session == "DUR_T" ~ "T"
  ),
  Session = as.factor(Session),
  Group = as.factor(GRUPO),
  Id = as.numeric(as.factor(Video)))

names(r_DB)[names(r_DB) == 'Video'] <- 'Rat'
levels(r_DB$Group) = c("Saline", "Muscimol")
levels(r_DB$Session) = c("S1", "S2", "T")

r_DB$Rat = as.factor(r_DB$Rat)

formula <- formula(Duration ~ Group + Session + Session*Group + (1|Rat))
formula2 <- formula(Duration ~ Group + Session + Session*Group)

analyze_model <- function(model) {
    print(summary(model))
    print(anova(model))

    model_residuals <- residuals(model)
    print(shapiro.test(model_residuals))

    model_desc <- emmeans(model, ~ Session|Group, type = "response")
    print(model_desc)

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



print("#------- Model 1 - GLMM with Normal distribution --------------------------------------#")
# p valores de lmerTest (Satterthwaite)
glmmNormal <- lmer(formula, data = r_DB) # LMM ao invés de GLMM
analyze_model(glmmNormal)
 
# print("#------- Model 2 - GLMM with Gamma distribution --------------------------------------#")
# glmmGaussianLog <- glmer(formula, data = r_DB, family = Gamma(link = "identity"))
# analyze_model(glmmGaussianLog)

print("#------- Model 3 - GLMM with Gamma distribution with log link function --------------------------------------#")
glmmGammaLog <- glmer(formula, data = r_DB, family = Gamma(link = "log"))
analyze_model(glmmGammaLog)

# print("#------- Model 4 - GEE with Gamma distribution with log link function --------------------------------------#")
geeGamma = geeglm(formula2, data = r_DB, family = Gamma("log"), id = Rat)
analyze_model(geeGamma)



# print("#------- Model 4 - GLMM with Poisson distribution --------------------------------------#")
# glmmPoisson <- glmer(formula, data = r_DB, family = poisson(link = "log"))
# analyze_model(glmmPoisson)

# print("#------- Model 5 - GLMM with Inverse Gaussian distribution --------------------------------------#")
# glmmInverseGaussian <- glmer(formula, data = r_DB, family = inverse.gaussian(link = "1/mu^2"))
# analyze_model(glmmInverseGaussian)


print("AIC:")
print(AIC(glmmNormal))
print(AIC(glmmGaussianLog))
print(AIC(glmmGammaLog))

print("BIC:")
print(BIC(glmmNormal))
print(BIC(glmmGaussianLog))
print(BIC(glmmGammaLog))

print("Log-Likelihood:")
print(logLik(glmmNormal))
print(logLik(glmmGaussianLog))
print(logLik(glmmGammaLog))

plot(glmmGammaLog)
qqnorm(resid(glmmGammaLog))
qqline(resid(glmmGammaLog))

# ranef(glmmGammaLog)
# VarCorr(glmmGammaLog)
# isSingular(glmmGammaLog, tol = 1e-4)   


boxplot_plot <- ggplot(r_DB, aes(x = Session, y = Duration, fill = Group)) +
  stat_boxplot(geom = "errorbar", width = 0.2, color = "black", position = position_dodge(width = 0.8)) +
  geom_boxplot(position = position_dodge(width = 0.8)) +
  scale_fill_manual(values = c("Saline" = "#145faa", "Muscimol" = "#828287")) +
  labs(title = "Rearing Duration by Group and Session",
       x = "Session", 
       y = "Rearing Duration (seconds)",
       fill = "Group") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5, size = 14), text = element_text(size = 12))
print(boxplot_plot)

model_emmeans <- emmeans(glmmGammaLog, ~ Group|Session, type = "response")
pred_data <- as.data.frame(model_emmeans)
pred_data$Session <- factor(pred_data$Session, levels = c("S1", "S2", "T"))

prediction_plot <- ggplot(pred_data, aes(x = Session, y = response, color = Group, group = Group)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = asymp.LCL, ymax = asymp.UCL), width = 0.1) +
  scale_color_manual(values = c("Saline" = "#145faa", "Muscimol" = "#828287")) +
  labs(title = "Model Predictions: Rearing Duration",
       x = "Session", 
       y = "Predicted Rearing Duration (seconds)",
       color = "Group") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        text = element_text(size = 12))
print(prediction_plot)

# simres <- simulateResiduals(fittedModel = glmmGammaLog)
# plot(simres)
# testDispersion(simres)
# testZeroInflation(simres)


ggsave("figures/glmm/boxplot_rearing_duration.png", boxplot_plot, width = 8, height = 6, dpi = 300)

ggsave("figures/glmm/predictions_rearing_duration.png", prediction_plot, width = 8, height = 6, dpi = 300)
