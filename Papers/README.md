# Papers 

## [Is it worth it? Comparing six deep and classical methods for unsupervised anomaly detection in time series](https://arxiv.org/abs/2212.11080)
مقایسه ۶ روش تشخیص ناهنجاری سری‌های زمانی (UCR)
این مقاله با عنوان "آیا ارزشش را دارد؟ مقایسه شش روش عمیق و کلاسیک برای تشخیص ناهنجاری بدون نظارت در سری‌های زمانی" در مجله Applied Sciences در سال ۲۰۲۳ منتشر شده است. نسخه اولیه این مقاله در دسامبر ۲۰۲۲ در arXiv منتشر شده است. 
ARXIV

در این پژوهش، شش روش مختلف برای تشخیص ناهنجاری در داده‌های سری‌های زمانی مقایسه شده‌اند. سه روش از این میان، تکنیک‌های کلاسیک یادگیری ماشین هستند:

جنگل برش تصادفی مقاوم (RRCF)
فواصل حداکثر واگرا (MDI)
MERLIN
سه روش دیگر مبتنی بر یادگیری عمیق هستند:

خودرمزگذار (AE)
جریان‌های نرمال‌سازی تقویت‌شده با گراف (GANF)
شبکه‌های ترنسفورمر برای تشخیص ناهنجاری (TranAD)
مجموعه داده‌های مورد استفاده از UCR Anomaly Archive گرفته شده‌اند که یکی از منابع معتبر برای تحلیل سری‌های زمانی است. 
PAPERS WITH CODE

نتایج این مطالعه نشان می‌دهد که روش‌های کلاسیک یادگیری ماشین در تشخیص ناهنجاری‌ها عملکرد بهتری نسبت به روش‌های یادگیری عمیق دارند. 
ARXIV

## [TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data](https://arxiv.org/abs/2201.07284)
افزودن TranAD: مدل ترنسفورمر برای تشخیص ناهنجاری در سری‌های زمانی چندمتغیره (VLDB 2021)
مقاله TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series در سال 2021 در کنفرانس VLDB (Very Large Data Bases) منتشر شده است.
این مقاله مدلی به نام TranAD را معرفی می‌کند که از شبکه‌های ترنسفورمر برای تشخیص ناهنجاری در سری‌های زمانی چندمتغیره استفاده می‌کند.
TranAD از یک ساختار بازخورد مبتنی بر بازسازی و پیش‌بینی بهره می‌برد که به طور همزمان ناهنجاری‌ها را در داده‌های پیچیده شناسایی می‌کند.
https://github.com/imperial-qore/tranad

## [AER: Auto-Encoder with Regression for Time Series Anomaly Detection](https://arxiv.org/abs/2212.13558)

افزودن مدل AER: خودرمزگذار با رگرسیون LSTM برای تشخیص ناهنجاری در سری‌های زمانی
مقاله AER: Auto-Encoder with Regression for Time Series Anomaly Detection در کنفرانس بین‌المللی IEEE در زمینه داده‌های بزرگ (IEEE International Conference on Big Data) در دسامبر ۲۰۲۲ ارائه شده است. 
DAI LIDS

این مقاله یک مدل ترکیبی به نام AER معرفی می‌کند که از یک خودرمزگذار (Auto-Encoder) و یک رگرسور LSTM برای تشخیص ناهنجاری در داده‌های سری زمانی استفاده می‌کند. 
ARXIV

## [Graph-Augmented Normalizing Flow](https://arxiv.org/abs/2202.07857)

The paper titled "Graph-Augmented Normalizing Flows for Anomaly Detection of Multiple Time Series" was presented at the International Conference on Learning Representations (ICLR) in 2022. 
ARXIV

This research introduces GANF (Graph-Augmented Normalizing Flow), a model that enhances normalizing flows with a Bayesian network to capture the intricate interdependencies among multiple time series. By factorizing the joint probability into conditional probabilities, GANF effectively identifies anomalies in complex datasets.
https://github.com/enyandai/ganf

## [Explainable Time Series Anomaly Detection using Masked Latent Generative Modeling](https://arxiv.org/abs/2311.12550)
Uses VQ-VAE for time series
https://github.com/ML4ITS/TimeVQVAE-AnomalyDetection

## [Unraveling the "Anomaly" in Time Series Anomaly Detection: A Self-supervised Tri-domain Solution](https://arxiv.org/abs/2311.11235)
وضیحات برای دیکریپشن:
این مقاله با عنوان "گشودن رمز ‘ناهنجاری’ در تشخیص ناهنجاری سری‌های زمانی: یک راه‌حل خودنظارتی سه‌دامنه‌ای" در arXiv منتشر شده است.
مدل TriAD (تشخیص‌دهنده ناهنجاری سه‌دامنه‌ای) معرفی شده که با استفاده از یادگیری خودنظارتی، ناهنجاری‌ها را از داده‌های سری‌های زمانی شناسایی می‌کند. این مدل سه دامنه زمانی، فرکانسی و باقیمانده را تحلیل کرده و از روش‌های کنتراست بین و درون دامنه‌ها بهره می‌برد.
این مدل یک الگوریتم کشف ناهماهنگی را برای شناسایی ناهنجاری‌های با طول متغیر ارائه می‌دهد.
کد منبع مدل در GitHub در دسترس است:
TriAD GitHub Repository
https://github.com/pseudo-Skye/TriAD

## [Unravel Anomalies: An End-to-End Seasonal-Trend Decomposition Approach for Time Series Anomaly Detection](https://arxiv.org/abs/2310.00268)
Add seasonal-trend decomposition model for time-series anomaly detection TADNet is an end-to-end time-series anomaly detection model that leverages seasonal-trend decomposition to link various types of anomalies to specific decomposition components, simplifying the analysis of complex time-series data and enhancing detection performance. IEEE 2023
https://github.com/zhangzw16/TADNet

## [Unsupervised Model Selection for Time-series Anomaly Detection](https://arxiv.org/abs/2210.01078)
Identify three classes of surrogate (unsupervised) metrics, namely, prediction error, model centrality, and performance on injected synthetic anomalies

## [MOMENT: A Family of Open Time-series Foundation Models](https://arxiv.org/abs/2402.03885)
Foundation models for time series. ICML 2024 
https://huggingface.co/AutonLab
https://github.com/moment-timeseries-foundation-model/moment

## [Deep Contrastive One-Class Time Series Anomaly Detection](https://arxiv.org/abs/2207.01472)
Contrastive Learning for time series. Introduces Contrastive One-Class Anomaly (COCA)
https://github.com/ruiking04/COCA
