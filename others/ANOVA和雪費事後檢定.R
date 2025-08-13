setwd('C:\\Users\\User\\Desktop\\macroeconomic factors\\2\\ANOVA檢定使用之資料集\\CSV')
IPI <- read.csv('IPI.csv',header=T,row.names = 1)
CPI <- read.csv('CPI.csv',header=T,row.names = 1)
CLI <- read.csv('CLI.csv',header=T,row.names = 1)
Unep <- read.csv('Unemployment.csv',header=T,row.names = 1)
Real <- read.csv('manufacturering.csv',header=T,row.names = 1)
Initial <- read.csv('Initial.csv',header=T,row.names = 1)

IPI$Label <- as.factor(IPI$Label)
CPI$Label <- as.factor(CPI$Label)
CLI$Label <- as.factor(CLI$Label)
Unep$Label <- as.factor(Unep$Label)
Real$Label <- as.factor(Real$Label)
Initial$Label <- as.factor(Initial$Label)


cli_result=aov(CLI~Label,data=CLI)
summary(cli_result)
#事後檢定(Post hoc)：用Tukey法檢定兩兩之間是否有顯著性差異
TukeyHSD(cli_result,conf.level=0.90)
# 拿到 TukeyHSD 的值
tukey_result <- TukeyHSD(cli_result, conf.level = 0.90)
df <- as.data.frame(tukey_result$Label)

# 手動算出每組的 SE = (upper - lower) / (2 * qt)
# qt 是 t 分布的臨界值（雙尾 90% 信賴區間，df 自己取）

df$SE <- (df$upr - df$lwr) / (2 * qt(0.95, df = df.residual(cli_result)))
print(df)

cpi_result=aov(CPI~Label,data=CPI)
summary(cpi_result)
#事後檢定(Post hoc)：用Tukey法檢定兩兩之間是否有顯著性差異
TukeyHSD(cpi_result,conf.level=0.90)
# 拿到 TukeyHSD 的值
tukey_result <- TukeyHSD(cpi_result, conf.level = 0.90)#
df <- as.data.frame(tukey_result$Label)

# 手動算出每組的 SE = (upper - lower) / (2 * qt)
# qt 是 t 分布的臨界值（雙尾 90% 信賴區間，df 自己取）

df$SE <- (df$upr - df$lwr) / (2 * qt(0.95, df = df.residual(cpi_result)))#
print(df)

initial_result=aov(Initial.Claims~Label,data=Initial)
summary(initial_result)
#事後檢定(Post hoc)：用Tukey法檢定兩兩之間是否有顯著性差異
TukeyHSD(initial_result,conf.level=0.90)
# 拿到 TukeyHSD 的值
tukey_result <- TukeyHSD(initial_result, conf.level = 0.90)#
df <- as.data.frame(tukey_result$Label)

# 手動算出每組的 SE = (upper - lower) / (2 * qt)
# qt 是 t 分布的臨界值（雙尾 90% 信賴區間，df 自己取）

df$SE <- (df$upr - df$lwr) / (2 * qt(0.95, df = df.residual(initial_result)))#
print(df)

real_result=aov(Real.Manufacturing.and.Trade.Industries.Sales~Label,data=Real)
summary(real_result)
#事後檢定(Post hoc)：用Tukey法檢定兩兩之間是否有顯著性差異
TukeyHSD(real_result,conf.level=0.90)
# 拿到 TukeyHSD 的值
tukey_result <- TukeyHSD(real_result, conf.level = 0.90)#
df <- as.data.frame(tukey_result$Label)

# 手動算出每組的 SE = (upper - lower) / (2 * qt)
# qt 是 t 分布的臨界值（雙尾 90% 信賴區間，df 自己取）

df$SE <- (df$upr - df$lwr) / (2 * qt(0.95, df = df.residual(real_result)))#
print(df)

unep_result=aov(Unemployment.rate~Label,data=Unep)
summary(unep_result)
#事後檢定(Post hoc)：用Tukey法檢定兩兩之間是否有顯著性差異
TukeyHSD(unep_result,conf.level=0.90)
# 拿到 TukeyHSD 的值
tukey_result <- TukeyHSD(unep_result, conf.level = 0.90)#
df <- as.data.frame(tukey_result$Label)

# 手動算出每組的 SE = (upper - lower) / (2 * qt)
# qt 是 t 分布的臨界值（雙尾 90% 信賴區間，df 自己取）

df$SE <- (df$upr - df$lwr) / (2 * qt(0.95, df = df.residual(unep_result)))#
print(df)

ipi_result=aov(IPI~Label,data=IPI)
summary(ipi_result)
#事後檢定(Post hoc)：用Tukey法檢定兩兩之間是否有顯著性差異
TukeyHSD(ipi_result,conf.level=0.90)
# 拿到 TukeyHSD 的值
tukey_result <- TukeyHSD(ipi_result, conf.level = 0.90)#
df <- as.data.frame(tukey_result$Label)

# 手動算出每組的 SE = (upper - lower) / (2 * qt)
# qt 是 t 分布的臨界值（雙尾 90% 信賴區間，df 自己取）

df$SE <- (df$upr - df$lwr) / (2 * qt(0.95, df = df.residual(ipi_result)))#
print(df)