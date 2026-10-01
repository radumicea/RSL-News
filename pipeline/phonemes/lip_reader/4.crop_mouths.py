# Whatever, too much code, but it uses a class similar to the one given in Auto-AVSR and using MediaPipe it extracts face keypoints and crops the mouth region.
# And removes "phoneme" "."" and turns "c" into "k"
# And we remain with:
# 1
# @
# S
# Z
# a
# b
# d
# e
# e_X
# f
# g
# gZ
# g_j
# gz
# h
# i
# i_0
# j
# je
# k
# k_j
# ks
# l
# m
# n
# o
# o_X
# p
# r
# s
# t
# tS
# ts
# u
# v
# w
# z
# Which are tokenized so we have these 37 tokens + token 0 which is blank for ctc
# So we have pairs of videos of mouth crop and sequence of the tokens of the phonemes
