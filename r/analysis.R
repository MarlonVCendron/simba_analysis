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

# DB = read.csv("data/summary_data.csv", header=TRUE, na.strings = "NA")
# t_DB = read.csv("data/spss/T_exp.csv", header=TRUE, na.strings = "NA", sep=";", dec=",")
# s1_DB = read.csv("data/spss/S1_exp.csv", header=TRUE, na.strings = "NA", sep=";", dec=",")
# s2_DB = read.csv("data/spss/S2_exp.csv", header=TRUE, na.strings = "NA", sep=";", dec=",")
r_DB = read.csv("data/spss/rearing.csv", header=TRUE, na.strings = "NA", sep=";", dec=",")

library(tidyr)
r_DB_long <- r_DB %>%
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

r_DB_long$Duration_plus1 = r_DB_long$Duration + 1

str(r_DB_long)

Plot_D1 <- 
        ggplot(r_DB_long, aes(x = Duration_plus1)) + 
        geom_histogram(aes(y = ..density..), fill = "cornflowerblue", binwidth = 3, boundary = 1) +
        geom_density(alpha = .2, fill = "cornflowerblue", position = "stack", size = 0.75) +
        xlim(-5, 60) +
        ggtitle("Rearing Duration with Added Constant (+1)") +
        labs (x = "Rearing Duration (+1)", y = element_blank() ) +
        theme_classic() +
        theme(plot.title = element_text(hjust = 0.5, size = 25), 
              text = element_text(size = 20), 
              axis.text = element_text(colour = "#000000")) +
        guides(fill = "none")
Plot_D1
paste("Rearing Duration with Added Constant: ",
      "Mean = ", mean(r_DB_long$Duration_plus1), 
      "Std. Error = ", sd(r_DB_long$Duration_plus1)/sqrt(length(r_DB_long$Duration_plus1)),
      "N = ", length(r_DB_long$Duration_plus1)
)

Plot_D2 <- 
        ggplot(r_DB_long, aes(x = Duration)) + 
        geom_histogram(aes(y = ..density..), fill = "darkseagreen2", binwidth = 3, boundary = 1) +
        geom_density(alpha = .2, fill = "darkseagreen2", position = "stack", size = 0.75) +
        xlim(-5, 60) +
        ggtitle("Raw Rearing Duration Data") +
        labs (x = "Rearing Duration", y = element_blank() ) +
        theme_classic() +
                theme(plot.title = element_text(hjust = 0.5, size = 25), 
                text = element_text(size = 20),
        axis.text = element_text(colour = "#000000")) +
        guides(fill = "none")
Plot_D2
paste("Raw Rearing Duration Data: ",
      "Mean = ", mean(r_DB_long$Duration), 
      "Std. Error = ", sd(r_DB_long$Duration)/sqrt(length(r_DB_long$Duration)),
      "N = ", length(r_DB_long$Duration)
)

#--------------------------------------------------------------------------------#
#---------Data Analysis----------------------------------------------------------#
#--------------------------------------------------------------------------------#

#-------Creating the formulas used in the analysis-------------------------------#
# F1 <- formula(Duration ~ Session * Group + Error(Id/(Session*Group))) #Formula for repeated measure Anova (rmANOVA)
# F2 <- formula(Duration ~ Group + Session) #Formula for GEE
# F3 <- formula(Duration_plus1 ~ Group + Session) #Formula for GEE with Duration with a constant (+1) added 
# #Dependent Variable: Rearing Duration or Duration with constant.
# # Independent variables: Session, Group

# # print("#--------Model 1 - Repeated Measure ANOVA (rmANOVA)-------------------------------#")
# # Model01_rmANOVA <- aov(F1, data = DB)
        
# #         summary(Model01_rmANOVA) #ANOVA table
# #         coef(Model01_rmANOVA) #ANOVA coefficients

# print("#------- Model 2 - Gee with Normal distribution --------------------------------------#")
# #Link Function: Identity
# #Correlation structure: Unstructured
# Model02_geeNORMAL <- geeglm(F2, data = r_DB_long, id = Id, family = gaussian("identity"), corstr = "unstructured", )
#         anova(Model02_geeNORMAL) #GEE main effects (Wald Statistics)
#         summary(Model02_geeNORMAL) #GEE table
#         QIC(Model02_geeNORMAL) #QIC values for the model
        
#         #Residual Analysis - Q-Q Plot and Normality test (Shapiro-Wilk)    
#         Model02_R <- Model02_geeNORMAL$residuals #Residuals for the model
#         shapiro.test(Model02_R) #Shapiro-Wilk test
        
#         Model02_R_Plot <- # Q-Q plot
#                 ggplot(mapping = aes(sample = Model02_R)) +
#                 stat_qq_line (size = 2, col = "steelblue") +
#                 stat_qq(pch = 1, size = 6) +
#                 ggtitle("Normal GEE") +
#                 labs (x = "Theoretical Quantiles", y = "Sample Quantiles" ) +
#                 theme_classic() +
#                 theme(plot.title = element_text(hjust = 0.5, size = 25),
#                       text = element_text(size = 20),
#                       axis.text = element_text(colour = "#000000")) +
#                 guides(fill = "none")
#         Model02_R_Plot
        
#         #Descriptive data (Table 2 of the manuscript) with Mean and Standard Error for the Interaction Session*Group
#         Model02_desc <- emmeans(Model02_geeNORMAL, ~ Session|Group, type = "response")
#         Model02_desc 
        
        
# print("#------- Model 3 - Gee with Gamma distribution --------------------------------------#")
# #Link Function: Identity
# #Correlation structure: Unstructured
# #The Gamma distribution only allows positive values; therefore, excluded the cases (i.e., lines) with any value = 0.

# r_DB_long_onlyPositive <- r_DB_long %>% filter(Duration > 0) #Exclude from the database cases with values = 0

# Model03_geeGAMMA1 = geeglm(F2, data = r_DB_long_onlyPositive, id = Id, family = Gamma("identity"), corstr = "unstructured")
#         anova(Model03_geeGAMMA1)
#         summary(Model03_geeGAMMA1)
#         QIC(Model03_geeGAMMA1)
        
#         #Residual Analysis - Q-Q Plot and Normality test (Shapiro-Wilk)    
#         Model03_R <- Model03_geeGAMMA1$residuals
#         shapiro.test(Model03_R)
        
#         Model03_R_Plot <- 
#                 ggplot(mapping = aes(sample = Model03_R)) +
#                 stat_qq_line (size = 2, col = "steelblue") +
#                 stat_qq(pch = 1, size = 6) +
#                 ggtitle("Gamma GEE") +
#                 labs (x = "Theoretical Quantiles", y = "Sample Quantiles" ) +
#                 theme_classic() +
#                 theme(plot.title = element_text(hjust = 0.5, size = 25),
#                       text = element_text(size = 20),
#                       axis.text = element_text(colour = "#000000")) +
#                 guides(fill = "none")
#         Model03_R_Plot
        
#         Model03_desc <- emmeans(Model03_geeGAMMA1, ~ Session|Group, type = "response")
#         Model03_desc 

# print("#------- Model 4 - Gee with Gamma distribution with constant added --------------------------------------#")
# #Link Function: Identity
# #Correlation structure: Unstructured
# #Instead of excluding cases with 0, it is possible to add a constant (e.g., +1) in the dependent variable to avoid values = 0.
# Model04_geeGAMMAc = geeglm(F3, data = r_DB_long, id = Id, family = Gamma("identity"), corstr = "unstructured")
#         anova(Model04_geeGAMMAc)
#         summary(Model04_geeGAMMAc)
#         QIC(Model04_geeGAMMAc)
        
#         #Residual Analysis - Q-Q Plot and Normality test (Shapiro-Wilk)    
#         Model04_R <- Model04_geeGAMMAc$residuals
#         shapiro.test(Model04_R)
        
#         Model04_R_Plot <- 
#                 ggplot(mapping = aes(sample = Model04_R)) +
#                 stat_qq_line (size = 2, col = "steelblue") +
#                 stat_qq(pch = 1, size = 6) +
#                 ggtitle("Gamma GEE + Constant") +
#                 labs (x = "Theoretical Quantiles", y = "Sample Quantiles" ) +
#                 theme_classic() +
#                 theme(plot.title = element_text(hjust = 0.5, size = 25),
#                       text = element_text(size = 20),
#                       axis.text = element_text(colour = "#000000")) +
#                 guides(fill = "none")
#         Model04_R_Plot
        
#         Model04_desc <- emmeans(Model04_geeGAMMAc, ~ Session|Group, type = "response")
#         Model04_desc

print("#------- Model 5 - GLMM with Normal distribution --------------------------------------#")
Model05_glmmNORMAL <- lmer(Duration ~ Group + Session + Session*Group + (1|Video), data = r_DB_long)
        summary(Model05_glmmNORMAL)
        anova(Model05_glmmNORMAL)
        
        Model05_R <- residuals(Model05_glmmNORMAL) #Residuals for the model
        shapiro.test(Model05_R) #Shapiro-Wilk test
        
        Model05_R_Plot <- 
                ggplot(mapping = aes(sample = Model05_R)) +
                stat_qq_line (size = 2, col = "steelblue") +
                stat_qq(pch = 1, size = 6) +
                ggtitle("Normal GLMM") +
                labs (x = "Theoretical Quantiles", y = "Sample Quantiles" ) +
                theme_classic() +
                theme(plot.title = element_text(hjust = 0.5, size = 25),
                      text = element_text(size = 20),
                      axis.text = element_text(colour = "#000000")) +
                guides(fill = "none")
        Model05_R_Plot
        
        Model05_desc <- emmeans(Model05_glmmNORMAL, ~ Session|Group, type = "response")
        Model05_desc
        