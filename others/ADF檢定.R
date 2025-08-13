setwd('C:\\Users\\User\\Desktop\\專題\\113-2負責的部分\\總體經濟指標\\macroeconomic factors\\1\\ADF資料集')
IPI <- read.csv('IPI.csv',header=T,row.names = 1)
CPI <- read.csv('CPI.csv',header=T,row.names = 1)
CLI <- read.csv('CLI.csv',header=T,row.names = 1)
Unep <- read.csv('Unemployment.csv',header=T,row.names = 1)
Real <- read.csv('manufacturering.csv',header=T,row.names = 1)
Initial <- read.csv('Initial.csv',header=T,row.names = 1)

#ADF檢定(有三種模型因此先看資料屬性)
CPI_ts <- ts(CPI$CPI, start = c(2006, 4), frequency = 12)  # 如果是月資料
plot.ts(CPI_ts, main="CPI Time Series", ylab="CPI", xlab="Time")
#看有無明顯趨勢
time_cpi <- 1:length(CPI_ts)
model_cpi <- lm(CPI_ts ~ time_cpi)
summary(model_cpi)#有截距且有趨勢=>tau3
library(urca)
adf_result <- ur.df(CPI_ts, type = "trend", selectlags = "AIC")
summary(adf_result)#t=-0.2061(p-value=0.837)=>無充分理由拒絕虛無假設H0:γ=0 → 資料有單根（非平穩）

CLI_ts <- ts(CLI$CLI, start = c(2006, 4), frequency = 12)  # 如果是月資料
plot.ts(CLI_ts, main="CLI Time Series", ylab="CLI", xlab="Time")
#看有無明顯趨勢
time_cli <- 1:length(CLI_ts)
model_cli <- lm(CLI_ts ~ time_cli)
summary(model_cli)#有截距且無趨勢=>tau2
library(urca)
adf_result <- ur.df(CLI_ts, type = "trend", selectlags = "AIC")
summary(adf_result)#t=-3.4898(p-value=0.000582)=>有充分理由拒絕虛無假設H0:γ=0 → 資料無單根（平穩）

IPI_ts <- ts(IPI$IPI, start = c(2006, 4), frequency = 12)  # 如果是月資料
plot.ts(IPI_ts, main="IPI Time Series", ylab="IPI", xlab="Time")
#看有無明顯趨勢
time_ipi <- 1:length(IPI_ts)
model_ipi <- lm(IPI_ts ~ time_ipi)
summary(model_ipi)#有截距且有趨勢=>tau3
library(urca)
adf_result <- ur.df(IPI_ts, type = "trend", selectlags = "AIC")
summary(adf_result)#t=-2.9971(p-value=0.003036)=>有充分理由拒絕虛無假設H0:γ=0 → 資料無單根（平穩）

Initial_ts <- ts(Initial$Initial.Claims, start = c(2006, 4), frequency = 12)  # 如果是月資料
plot.ts(Initial_ts, main="Initial Time Series", ylab="Initial", xlab="Time")
#看有無明顯趨勢
time_initial <- 1:length(Initial_ts)
model_initial <- lm(Initial_ts ~ time_initial)
summary(model_initial)#有截距且無趨勢=>tau2
library(urca)
adf_result <- ur.df(Initial_ts, type = "drift", selectlags = "AIC")
summary(adf_result)#t=-6.830(p-value=0.0000000000796)=>有充分理由拒絕虛無假設H0:γ=0 → 資料無單根（平穩）

Real_ts <- ts(Real$Real.Manufacturing.and.Trade.Industries.Sales, start = c(2006, 4), frequency = 12)  # 如果是月資料
plot.ts(Real_ts, main="Real Time Series", ylab="Real", xlab="Time")
#看有無明顯趨勢
time_Real <- 1:length(Real_ts)
model_real <- lm(Real_ts ~ time_Real)
summary(model_real)#有截距且有趨勢=>tau3
library(urca)
adf_result <- ur.df(Real_ts, type = "trend", selectlags = "AIC")
summary(adf_result)#t=-3.132(p-value=0.00197)=>有充分理由拒絕虛無假設H0:γ=0 → 資料無單根（平穩）

Unep_ts <- ts(Unep$Unemployment.rate, start = c(2006, 4), frequency = 12)  # 如果是月資料
plot.ts(Unep_ts, main="Unep Time Series", ylab="Unep", xlab="Time")
#看有無明顯趨勢
time_unep <- 1:length(Unep_ts)
model_unep <- lm(Unep_ts ~ time_unep)
summary(model_unep)#有截距且無趨勢=>tau2
library(urca)
adf_result <- ur.df(Unep_ts, type = "trend", selectlags = "AIC")
summary(adf_result)#t=-3.302(p-value=0.00112)=>有充分理由拒絕虛無假設H0:γ=0 → 資料無單根（平穩）
