# Web and Social Event Signals for AI-Driven Mobile Network Demand Forecasting

This is the repositoty to replicate the experiments of a paper we published in Open Research Europe (https://open-research-europe.ec.europa.eu).

## Data

Data derives from the Netmob 2023 challenge dataset (https://arxiv.org/abs/2305.06933):

- **service_clustering.csv** contains the clustering of application services into 5 clusters

- **nantes_antenna_clustering.csv** contains information about the network graph and its cluserting into ageggated regions (far edge and near edge areas)

- **prediction_clusto_i.csv** files contains network demand real data and prediction for multiple class of application services; *i*-file is associated with the *i*th cluster of application services.  


## Code

- **event_error.py** Code to run experiments to understand the impact of events in demand forecasting

- **optimization_and_event_errors.py** Code to run experiments to understand the impact of events in network optimization (i.e., allocate resources based on demand foreacting)

- **ml_exps.py** Code to run experiment to compare multiple ML models.



