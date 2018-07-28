library(RColorBrewer)
library(devtools)
library(ggplot2)
library(scales)
install_github("easyGgplot2", "kassambara")
library(easyGgplot2)
# Error bars represent standard error of the mean
AUROC1vs1 <- data.frame(Algorithm = c(' XGBDTI','DTINet','HNM','NetLapRLS'),
                  AUC = c(0.9264,0.9141,0.8551,0.829),
                  sd = c(0.009,0.010,0.016,0.012))
AUPR1vs1 <- data.frame(Algorithm = c(' XGBDTI','DTINet','HNM','NetLapRLS'),
                        AUC = c(0.9350,0.9322,0.8798,0.8821),
                        sd = c(0.012,0.008,0.009,0.010))
AUPR1vs10 <- data.frame(Algorithm = c(' XGBDTI','DTINet','HNM','NetLapRLS'),
                       AUPR = c(0.8211,0.7602,0.5520,0.6042),
                       sd = c(0.012,0.008,0.009,0.010))
AUPR_homo <- data.frame(Algorithm = c(' XGBDTI','DTINet','HNM','NetLapRLS'),
                        AUPR = c(0.8108,0.6102,0.4789,0.4213),
                        sd = c(0.012,0.008,0.009,0.010))
AUPR_drug <- data.frame(Algorithm = c(' XGBDTI','DTINet','HNM','NetLapRLS'),
                        AUPR = c(0.7620,0.7182,0.5023,0.5210),
                        sd = c(0.012,0.008,0.009,0.010))
AUPR_diease <- data.frame(Algorithm = c(' XGBDTI','DTINet','HNM','NetLapRLS'),
                        AUPR = c(0.7600,0.7230,0.5455,0.5302),
                        sd = c(0.012,0.008,0.009,0.010))
AUPR_sideeffects <- data.frame(Algorithm = c(' XGBDTI','DTINet','HNM','NetLapRLS'),
                        AUPR = c(0.8054,0.7620,0.5462,0.6003),
                        sd = c(0.012,0.008,0.009,0.010))
AUPR_unique <- data.frame(Algorithm = c(' XGBDTI','DTINet','HNM','NetLapRLS'),
                        AUPR = c(0.3208,0.2519,0.1643,0.1682),
                        sd = c(0.012,0.008,0.009,0.010))


plot1 <- ggplot(AUROC1vs1, aes(x=Algorithm, y=AUC)) + 
  geom_bar(position=position_dodge(), stat="identity",width = 0.5,fill=c('lightblue','lightpink','grey','lightgreen')) +
  geom_errorbar(aes(ymin=AUC-sd, ymax=AUC+sd),
                width=.1,
                position=position_dodge(.9))+
  scale_y_continuous(limits=c(0.5,1), oob=rescale_none)+
  ggtitle("(a)AUROC\nNo.positive:No.negtive=1:1")+
  theme(plot.title = element_text(hjust = 0.5)) 

plot2 <- ggplot(AUPR1vs1, aes(x=Algorithm, y=AUC)) + 
  geom_bar(position=position_dodge(), stat="identity",width = 0.5,fill=c('lightblue','lightpink','grey','lightgreen')) +
  geom_errorbar(aes(ymin=AUC-sd, ymax=AUC+sd),
                width=.1,
                position=position_dodge(.9))+
  scale_y_continuous(limits=c(0.5,1), oob=rescale_none)+
  ggtitle("(b)AUPR\nNo.positive:No.negtive=1:1")+
  theme(plot.title = element_text(hjust = 0.5)) 
plot3 <- ggplot(AUPR1vs10, aes(x=Algorithm, y=AUPR)) + 
  geom_bar(position=position_dodge(), stat="identity",width = 0.5,fill=c('lightblue','lightpink','grey','lightgreen')) +
  geom_errorbar(aes(ymin=AUPR-sd, ymax=AUPR+sd),
                width=.1,
                position=position_dodge(.9))+
  scale_y_continuous(limits=c(0.5,0.9), oob=rescale_none)+
  ggtitle("(c)AUPR\nNo.positive:No.negtive=1:10")+
  theme(plot.title = element_text(hjust = 0.5)) 
plot4 <- ggplot(AUPR_homo, aes(x=Algorithm, y=AUPR)) + 
  geom_bar(position=position_dodge(), stat="identity",width = 0.5,fill=c('lightblue','lightpink','grey','lightgreen')) +
  geom_errorbar(aes(ymin=AUPR-sd, ymax=AUPR+sd),
                width=.05,
                position=position_dodge(.3))+
  scale_y_continuous(limits=c(0.2,0.8), oob=rescale_none)+
  ggtitle("(d)AUPR\nNo.positive:No.negtive=1:10\nDTIs with similar drugs or\nproteins were removed")+
  theme(plot.title = element_text(hjust = 0.5)) 
plot5 <- ggplot(AUPR_drug, aes(x=Algorithm, y=AUPR)) + 
  geom_bar(position=position_dodge(), stat="identity",width = 0.5,fill=c('lightblue','lightpink','grey','lightgreen')) +
  geom_errorbar(aes(ymin=AUPR-sd, ymax=AUPR+sd),
                width=.05,
                position=position_dodge(.9))+
  scale_y_continuous(limits=c(0.2,0.9), oob=rescale_none)+
  ggtitle("(e)AUPR\nNo.positive:No.negtive=1:10\nDTIs with drugs sharing similar\ndrug interactions were removed")+
  theme(plot.title = element_text(hjust = 0.5)) 
plot6 <- ggplot(AUPR_sideeffects, aes(x=Algorithm, y=AUPR)) + 
  geom_bar(position=position_dodge(), stat="identity",width = 0.5,fill=c('lightblue','lightpink','grey','lightgreen')) +
  geom_errorbar(aes(ymin=AUPR-sd, ymax=AUPR+sd),
                width=.05,
                position=position_dodge(.9))+
  scale_y_continuous(limits=c(0.2,0.9), oob=rescale_none)+
  ggtitle("(f)AUPR\nNo.positive:No.negtive=1:10\nDTIs with drugs sharing\nsimilar side-effects were removed")+
  theme(plot.title = element_text(hjust = 0.5))
plot7 <- ggplot(AUPR_diease, aes(x=Algorithm, y=AUPR)) + 
  geom_bar(position=position_dodge(), stat="identity",width = 0.5,fill=c('lightblue','lightpink','grey','lightgreen')) +
  geom_errorbar(aes(ymin=AUPR-sd, ymax=AUPR+sd),
                width=.05,
                position=position_dodge(.9))+
  scale_y_continuous(limits=c(0.2,0.8), oob=rescale_none)+
  ggtitle("(g)AUPR\nNo.positive:No.negtive=1:10\nDTIs with drugs sharing\nsimilar disease were removed")+
  theme(plot.title = element_text(hjust = 0.5)) 
plot8 <- ggplot(AUPR_unique, aes(x=Algorithm, y=AUPR)) + 
  geom_bar(position=position_dodge(), stat="identity",width = 0.5,fill=c('lightblue','lightpink','grey','lightgreen')) +
  geom_errorbar(aes(ymin=AUPR-sd, ymax=AUPR+sd),
                width=.1,
                position=position_dodge(.9))+
  scale_y_continuous(limits=c(0,0.5), oob=rescale_none)+
  ggtitle("Trained on non-unique interactions,\ntested on unique interactions")+
  theme(plot.title = element_text(hjust = 0.5)) 
ggplot2.multiplot(plot1,plot2,plot3,plot4,plot5,plot6,plot7,plot8,cols = 2)
ggplot2.multiplot(plot4,plot5,plot6,plot7,cols = 4)
