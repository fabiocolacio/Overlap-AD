
# Unsupervised Anomaly Detection

This repository contains a reference implementation for our algorithm for anomaly detection.

## Rationale 

Traditional machine-learning solutions classify data based entirely off of past experiences.
While they are effective at learning new patterns and accurately classifying real anomalies, they are susceptible to experiencing a high frequency of false-positives.
These occur when new patterns arise in the data because those patterns have not been seen before.

We attempt to reduce the number of false-positives experienced by withholding judgment of suspicious data.
When a data is suspected of being anomalous, its classification is "pending", and we wait for more information before classifying it as an anomaly.
This helps us to determine of the data is part of a new, normal pattern, or if it is truly an anomaly.
In other words, our classifications are based on past and future data, unlike other algorithms which make classifications based only on past data.

