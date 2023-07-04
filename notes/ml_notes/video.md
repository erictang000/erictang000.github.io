---
layout: post
title: SOTA on Kinetics and SSV2 - 
---
## Task Overview and Datasets
The task of video action recognition generally is framed as a classification task, where given a video, the goal is to predict the action that occurs in the video. A common additional task includes predicting the start and end times of each actions in longer videos. Videos used for benchmarking video understanding models have most commonly been scraped from platforms like Youtube, and then manually annotated with action labels. A number of datasets exist for measuring the abilities of models to understand human action in videos - the following are the most commonly used in somewhat chronological order:
<p align="center">
    <img src="/images/video_datasets.png" width="500">
</p>
Anecdotally, it seems like Kinetics-400, Kinetics-600, and SomethingSomethingV2 are the most commonly currently used datasets for benchmarking new video architectures. Kinetics videos are collected from youtube, whereas SSV2 videos are recorded by crowdworkers.