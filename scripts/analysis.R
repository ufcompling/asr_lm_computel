library(brms)
library(ggplot2)
library(lmerTest)
library(MuMIn)
library(car)
library(broom)

data <- read.csv('results/evaluation.txt', header = T, sep = '\t')
data_random <- subset(data, Quality == 'NONE' & Merge == 'NONE' & LM_order == 'NONE')

ggplot(data_random, aes(Language, WER, group = Size, color = Size)) +
  geom_point() +
  #  geom_text(aes(label = Score), vjust = 1) +
  scale_color_manual(values=c("#0072B2", "#009E73")) + #, "#CC79A7", "#D55E00"))+
  xlab("") +
  ylab("") +
  theme(axis.text.y = element_text(size = 10),
        #       axis.text.x = element_blank(),
        axis.title.y = element_text(size = 10),
        axis.title.x = element_text(size = 10),
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 10),
        legend.position = 'top') +
  ggtitle('Random')

### Simulating Hupa size
data_tier = subset(data, Quality %in% c('top_tier', 'second_tier')  & Merge == 'NONE' & LM_order == 'NONE')

ggplot(data_tier, aes(Language, WER, group = Size, color = Size)) +
  geom_point() +
  #  geom_text(aes(label = Score), vjust = 1) +
  scale_color_manual(values=c("#0072B2", "#009E73", "#CC79A7")) + #, "#D55E00"))+
  xlab("") +
  ylab("") +
  facet_wrap(~ Quality, ncol = 4) +
  theme(axis.text.y = element_text(size = 10),
 #       axis.text.x = element_blank(),
        axis.title.y = element_text(size = 10),
        axis.title.x = element_text(size = 10),
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 10),
        legend.position = 'top') +
  ggtitle('')

### BPE
data_bpe = subset(data, Quality == 'NONE'  & Merge != 'NONE' & LM_order != 'NONE')
data_bpe$Merge = as.numeric(data_bpe$Merge)
data_bpe$LM_order = as.numeric(data_bpe$LM_order)
fongbe_bpe = subset(data_bpe, Language == 'fongbe')
wolof_bpe = subset(data_bpe, Language == 'wolof')
iban_bpe = subset(data_bpe, Language == 'iban')
swahili_bpe = subset(data_bpe, Language == 'swahili')

ggplot(fongbe_bpe, aes(LM_order, WER, group = Size, color = Size)) +
  geom_point() +
  #  geom_text(aes(label = Score), vjust = 1) +
  scale_color_manual(values=c("#0072B2", "#009E73", "#CC79A7")) + #, "#D55E00"))+
  xlab("") +
  ylab("") +
  facet_wrap(Language ~ Merge, ncol = 4) +
  theme(axis.text.y = element_text(size = 10),
        #       axis.text.x = element_blank(),
        axis.title.y = element_text(size = 10),
        axis.title.x = element_text(size = 10),
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 10),
        legend.position = 'top') +
  ggtitle('Fongbe BPE')

ggplot(wolof_bpe, aes(LM_order, WER, group = Size, color = Size)) +
  geom_point() +
  #  geom_text(aes(label = Score), vjust = 1) +
  scale_color_manual(values=c("#0072B2", "#009E73", "#CC79A7")) + #, "#D55E00"))+
  xlab("") +
  ylab("") +
  facet_wrap(Language ~ Merge, ncol = 4) +
  theme(axis.text.y = element_text(size = 10),
        #       axis.text.x = element_blank(),
        axis.title.y = element_text(size = 10),
        axis.title.x = element_text(size = 10),
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 10),
        legend.position = 'top') +
  ggtitle('Wolof BPE')

ggplot(iban_bpe, aes(LM_order, WER, group = Size, color = Size)) +
  geom_point() +
  #  geom_text(aes(label = Score), vjust = 1) +
  scale_color_manual(values=c("#0072B2", "#009E73", "#CC79A7")) + #, "#D55E00"))+
  xlab("") +
  ylab("") +
  facet_wrap(Language ~ Merge, ncol = 4) +
  theme(axis.text.y = element_text(size = 10),
        #       axis.text.x = element_blank(),
        axis.title.y = element_text(size = 10),
        axis.title.x = element_text(size = 10),
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 10),
        legend.position = 'top') +
  ggtitle('Iban BPE')

ggplot(swahili_bpe, aes(LM_order, WER, group = Size, color = Size)) +
  geom_point() +
  #  geom_text(aes(label = Score), vjust = 1) +
  scale_color_manual(values=c("#0072B2", "#009E73", "#CC79A7")) + #, "#D55E00"))+
  xlab("") +
  ylab("") +
  facet_wrap(Language ~ Merge, ncol = 4) +
  theme(axis.text.y = element_text(size = 10),
        #       axis.text.x = element_blank(),
        axis.title.y = element_text(size = 10),
        axis.title.x = element_text(size = 10),
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 10),
        legend.position = 'top') +
  ggtitle('Swahili BPE')

