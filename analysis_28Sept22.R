#---- R Packages Needed ------------------
rm(list=ls()) #Clears all objects from the workspace
if(!require(geepack))install.packages("geepack")
if(!require(ggplot2))install.packages("ggplot2")
if(!require(readr))install.packages("readr")
if(!require(tidyr))install.packages("tidyr")
if(!require(dplyr))install.packages("dplyr")
if(!require(emmeans))install.packages("emmeans")

#--------------------------------------------------------------------------------#
#-----------------Database Creation----------------------------------------------#
#--------------------------------------------------------------------------------#
#The database is also available in wide and long formats in TODD (LINK).
#Variables:
#Id - Identification of animals (21 levels)
#Group - Experimental groups (3 levels)
#Time - Time of test, minute by minute (5 levels)
#Freezing - Time spent in freezing behavior.
#Freezing_Constant - Same as "Freezing", but with a constant of +1 added to each observation to avoid non-positive values in the database (i.e.,)
Table = ("Id	Group	Time	Freezing	Freezing_Constant
1	3	1	27	28
1	3	2	34	35
1	3	3	18	19
1	3	4	28	29
1	3	5	5	6
2	2	1	17	18
2	2	2	53	54
2	2	3	15	16
2	2	4	19	20
2	2	5	11	12
3	1	1	8	9
3	1	2	15	16
3	1	3	4	5
3	1	4	8	9
3	1	5	4	5
4	2	1	30	31
4	2	2	26	27
4	2	3	43	44
4	2	4	56	57
4	2	5	29	30
5	1	1	21	22
5	1	2	24	25
5	1	3	23	24
5	1	4	32	33
5	1	5	13	14
6	3	1	1	2
6	3	2	2	3
6	3	3	0	1
6	3	4	2	3
6	3	5	0	1
7	3	1	18	19
7	3	2	20	21
7	3	3	30	31
7	3	4	38	39
7	3	5	47	48
8	2	1	27	28
8	2	2	7	8
8	2	3	4	5
8	2	4	0	1
8	2	5	4	5
9	1	1	50	51
9	1	2	51	52
9	1	3	59	60
9	1	4	50	51
9	1	5	59	60
10	3	1	2	3
10	3	2	21	22
10	3	3	10	11
10	3	4	2	3
10	3	5	2	3
11	3	1	2	3
11	3	2	1	2
11	3	3	1	2
11	3	4	0	1
11	3	5	1	2
12	1	1	24	25
12	1	2	37	38
12	1	3	32	33
12	1	4	21	22
12	1	5	25	26
13	2	1	12	13
13	2	2	31	32
13	2	3	20	21
13	2	4	16	17
13	2	5	9	10
14	1	1	51	52
14	1	2	49	50
14	1	3	55	56
14	1	4	49	50
14	1	5	59	60
15	3	1	8	9
15	3	2	31	32
15	3	3	29	30
15	3	4	44	45
15	3	5	36	37
16	3	1	17	18
16	3	2	18	19
16	3	3	43	44
16	3	4	31	32
16	3	5	13	14
17	1	1	34	35
17	1	2	42	43
17	1	3	47	48
17	1	4	42	43
17	1	5	43	44
18	2	1	23	24
18	2	2	29	30
18	2	3	28	29
18	2	4	18	19
18	2	5	18	19
19	1	1	40	41
19	1	2	54	55
19	1	3	51	52
19	1	4	37	38
19	1	5	41	42
20	2	1	43	44
20	2	2	44	45
20	2	3	21	22
20	2	4	24	25
20	2	5	8	9
21	2	1	38	39
21	2	2	48	49
21	2	3	26	27
21	2	4	27	28
21	2	5	20	21
")

#-------Database as an data.frame object--------------------------#
DB = read.table(textConnection(Table), header=TRUE, na.string = "NA")

#Transforming variables into factors
DB$Group = as.factor(DB$Group)
DB$Time = as.factor(DB$Time)
DB$Id = as.factor(DB$Id)

#Naming the levels of Groups and Time Variables
levels(DB$Group) = c("Saline", "Muscimol", "AP5")
levels(DB$Time) = c("Min_01","Min_02","Min_03","Min_04","Min_05")

str(DB) #Overview of the finalized database.

#--------------------------------------------------------------------------------#
#Creating Figure 1 - Density plot, histogram, and descriptive statistics---------#
#--------------------------------------------------------------------------------#

#Freezing with constant - Histogram and density plot
Plot_D1 <- 
        ggplot(DB, aes(x = Freezing_Constant)) + 
        geom_histogram(aes(y = ..density..), fill = "cornflowerblue", binwidth = 3, boundary = 1) +
        geom_density(alpha = .2, fill = "cornflowerblue", position = "stack", size = 0.75) +
        xlim(-5, 60) +
        ggtitle("Data with Added Constant (+1)") +
        labs (x = "Freezing (+1)", y = element_blank() ) +
        theme_classic() +
        theme(plot.title = element_text(hjust = 0.5, size = 25), 
              text = element_text(size = 20), 
              axis.text = element_text(colour = "#000000")) +
        guides(fill = "none")
Plot_D1
#Freezing with constant - Descriptive statistics
paste("Data with Added Constant: ",
      "Mean = ", mean(DB$Freezing_Constant), 
      "Std. Error = ", sd(DB$Freezing)/sqrt(length(DB$Freezing_Constant)),
      "N = ", length(DB$Freezing_Constant)
)

#Freezing - Histogram and density plot
Plot_D2 <- 
        ggplot(DB, aes(x = Freezing)) + 
        geom_histogram(aes(y = ..density..), fill = "darkseagreen2", binwidth = 3, boundary = 1) +
        geom_density(alpha = .2, fill = "darkseagreen2", position = "stack", size = 0.75) +
        xlim(-5, 60) +
        ggtitle("Raw Data") +
        labs (x = "Freezing", y = element_blank() ) +
        theme_classic() +
                theme(plot.title = element_text(hjust = 0.5, size = 25), 
                text = element_text(size = 20),
        axis.text = element_text(colour = "#000000")) +
        guides(fill = "none")
Plot_D2
#Freezing - Descriptive statistics
paste("Raw Data: ",
      "Mean = ", mean(DB$Freezing), 
      "Std. Error = ", sd(DB$Freezing)/sqrt(length(DB$Freezing)),
      "N = ", length(DB$Freezing)
)

#--------------------------------------------------------------------------------#
#---------Data Analysis----------------------------------------------------------#
#--------------------------------------------------------------------------------#

#-------Creating the formulas used in the analysis-------------------------------#
F1 <- formula(Freezing ~ Time * Group + Error(Id/(Time*Group))) #Formula for repeated measure Anova (rmANOVA)
F2 <- formula(Freezing ~ Group + Time + Time*Group) #Formula for GEE
F3 <- formula(Freezing_Constant ~ Group + Time + Time*Group) #Formula for GEE with Freezing with a constant (+1) added 
#Dependent Variable: Freezing or Freezing with constant.
# Independent variables: Time, Group and Interaction between Time and Group

#--------Model 1 - Repeated Measure ANOVA (rmANOVA)-------------------------------#
Model01_rmANOVA <- aov(F1, data = DB)
        
        summary(Model01_rmANOVA) #ANOVA table
        coef(Model01_rmANOVA) #ANOVA coefficients

#------- Model 2 - Gee with Normal distribution --------------------------------------#
#Link Function: Identity
#Correlation structure: Unstructured
Model02_geeNORMAL <- geeglm(F2, data = DB, id = Id, family = gaussian("identity"), corstr = "unstructured", )
        anova(Model02_geeNORMAL) #GEE main effects (Wald Statistics)
        summary(Model02_geeNORMAL) #GEE table
        QIC(Model02_geeNORMAL) #QIC values for the model
        
        #Residual Analysis - Q-Q Plot and Normality test (Shapiro-Wilk)    
        Model02_R <- Model02_geeNORMAL$residuals #Residuals for the model
        shapiro.test(Model02_R) #Shapiro-Wilk test
        
        Model02_R_Plot <- # Q-Q plot
                ggplot(mapping = aes(sample = Model02_R)) +
                stat_qq_line (size = 2, col = "steelblue") +
                stat_qq(pch = 1, size = 6) +
                ggtitle("Normal GEE") +
                labs (x = "Theoretical Quantiles", y = "Sample Quantiles" ) +
                theme_classic() +
                theme(plot.title = element_text(hjust = 0.5, size = 25),
                      text = element_text(size = 20),
                      axis.text = element_text(colour = "#000000")) +
                guides(fill = "none")
        Model02_R_Plot
        
        #Descriptive data (Table 2 of the manuscript) with Mean and Standard Error for the Interaction Time*Group
        Model02_desc <- emmeans(Model02_geeNORMAL, ~ Time|Group, type = "response")
        Model02_desc 
        
        
#------- Model 3 - Gee with Gamma distribution --------------------------------------#
#Link Function: Identity
#Correlation structure: Unstructured
#The Gamma distribution only allows positive values; therefore, excluded the cases (i.e., lines) with any value = 0.

DB_onlyPositive <- filter_if(DB, is.numeric, all_vars((.) != 0)) #Exclude from the database cases with values = 0

Model03_geeGAMMA1 = geeglm(F2, data = DB_onlyPositive, id = Id, family = Gamma("identity"), corstr = "unstructured")
        anova(Model03_geeGAMMA1)
        summary(Model03_geeGAMMA1)
        QIC(Model03_geeGAMMA1)
        
        #Residual Analysis - Q-Q Plot and Normality test (Shapiro-Wilk)    
        Model03_R <- Model03_geeGAMMA1$residuals
        shapiro.test(Model03_R)
        
        Model03_R_Plot <- 
                ggplot(mapping = aes(sample = Model03_R)) +
                stat_qq_line (size = 2, col = "steelblue") +
                stat_qq(pch = 1, size = 6) +
                ggtitle("Gamma GEE") +
                labs (x = "Theoretical Quantiles", y = "Sample Quantiles" ) +
                theme_classic() +
                theme(plot.title = element_text(hjust = 0.5, size = 25),
                      text = element_text(size = 20),
                      axis.text = element_text(colour = "#000000")) +
                guides(fill = "none")
        Model03_R_Plot
        
        Model03_desc <- emmeans(Model03_geeGAMMA1, ~ Time|Group, type = "response")
        Model03_desc 

#------- Model 4 - Gee with Gamma distribution with constant added --------------------------------------#
#Link Function: Identity
#Correlation structure: Unstructured
#Instead of excluding cases with 0, it is possible to add a constant (e.g., +1) in the dependent variable to avoid values = 0.
Model04_geeGAMMAc = geeglm(F3, data = DB, id = Id, family = Gamma("identity"), corstr = "unstructured")
        anova(Model04_geeGAMMAc)
        summary(Model04_geeGAMMAc)
        QIC(Model04_geeGAMMAc)
        
        #Residual Analysis - Q-Q Plot and Normality test (Shapiro-Wilk)    
        Model04_R <- Model04_geeGAMMAc$residuals
        shapiro.test(Model04_R)
        
        Model04_R_Plot <- 
                ggplot(mapping = aes(sample = Model04_R)) +
                stat_qq_line (size = 2, col = "steelblue") +
                stat_qq(pch = 1, size = 6) +
                ggtitle("Gamma GEE + Constant") +
                labs (x = "Theoretical Quantiles", y = "Sample Quantiles" ) +
                theme_classic() +
                theme(plot.title = element_text(hjust = 0.5, size = 25),
                      text = element_text(size = 20),
                      axis.text = element_text(colour = "#000000")) +
                guides(fill = "none")
        Model04_R_Plot
        
        Model04_desc <- emmeans(Model04_geeGAMMAc, ~ Time|Group, type = "response")
        Model04_desc 
