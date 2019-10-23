# coding: utf-8
# @Time    : 18-10-29 下午5:00
# @Author  : jia

# bagging: build S datasets from original dataset
# by randomly selecting an example from the original with replacement
# then train S classifiers by these S dataset
# last, apply S classifiers to new piece of data and take majority vote
# e.g: random forest

# boosting: different classifierss are trained sequentially
# new classifiers is trained based on the performance of those already trained by focusing on misclassified data
# e.g: Adaboost

