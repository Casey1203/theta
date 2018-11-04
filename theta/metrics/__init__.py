# coding: utf-8
# @Time    : 18-11-1 上午11:19
# @Author  : jia

# confusion matrix

#                          predicted
# 		           +1,                 -1
# actual
# +1      true positive(TP)      false negative(FN)
# -1      false positive(FP)     true negative(TN)


# precision = TP / (TP + FP) 猜得多准
# recall = TP / (TP + FN) 猜出了多少

# measure classification imbalance: ROC curve
# x-axis: number of false positive
# y-axis: number of true positive
