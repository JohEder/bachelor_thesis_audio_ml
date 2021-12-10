#Code Repository for my Bachelor thesis: Anomalous Sound Detection for AcousticTraffic Monitoring using Transformers

Abstract:
Acoustic Traffic Monitoring (ATM) helps to make road traffic safer by detecting accidentsautomatically and helping to better estimate traffic load. 
Anomalous sound detection(ASD) has been sparsely used in ATM, which relies mostly on classification algorithms.
There are however certain situations, where classifications are not feasible, because notenough data is available.
Transformer Models have long been state-of-the art in Language Modeling Tasks due totheir ability to capture the relationship 
between different input tokens accurately even ifthey are very distant. Recently however, they have also outperformed previous methodsin different domains 
such as Computer Vision or Audio Understanding.This thesis examines whether Transformer Models are also suitable for ASD, 
by introduc-ing a purely attention-based masked-reconstruction architecture called Anomalous SoundDetection Transformer (ASDT). 
The model is trained on normal data in a self-supervisedway to reconstruct rectangular patches of log-mel-spectrograms. 
During inference, thereconstruction error can be interpreted as anomaly score.  
It is hypothesized that theproposed model outperforms traditional deep-learning ASD methods because of the at-tention mechanism which allows it to better leverage the spatial properties of a log-mel-spectrogram.
The model is evaluated on the recently published IDMT-Traffic dataset. 
The dataset con-sists of over 4000 annotated sounds of passing vehicles, in total 2.5 hours. 
The anomalydetection settings used to evaluate the model include vehicle type detection, 
speed detec-tion and wet road conditions detection. The ASDT-Model is compared to two baselinemodels, 
an Autoencoder and a Deep Interpolation Neural Network (IDNN).The proposed ASDT-Model beats the IDNN in all settings 
by a high amount and theAutoencoder in almost all settings by a small margin.
