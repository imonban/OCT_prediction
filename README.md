# A Deep-learning Approach for Prognosis of Age-Related Macular Degeneration Disease using SD-OCT Imaging Biomarkers 

The code loads the trained weights and a sample dataset, and predicts the AMD progression using a longitudinal deep learning model. Finally, it computes the preformance as AUC ROC.

## Dependencies needed

1. Python 2
2. Pandas '0.19.2'
3. sklearn - '0.19.1'
4. keras -  2.1.6 with tensorflow 1.8.0
5. matplotlib - 1.5.1

## Execution (Very simple running!! Trust me)

1. unzip weights
2. I will upload the sample dataset features (computed by Cirrus Review Software), put it under weights
2. Run the Testing_OCT.py from terminal as: python Testing_OCT.py
3. And that's it!! It will read the sample data, predict the outcome and plot the AUC ROC for 3 months and 21 months prediction.


## Output
Produce ROC curves over the prediction results - both short and long term progression of AMD.


