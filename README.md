# Anomaly Detection Using Machine Learning: An Application to Automated Vehicle Health Monitoring
Anomaly detection has been instrumental in identifying unusual patterns in the data which can be indicative of health deterioration of a system, potential component failure, or a random defect. Anomaly detection using machine learning techniques which represents emerging trends while retaining historical knowledge helps in continuous condition monitoring of these systems. It enables to utilize growth in technology towards better control, more reliability, and reduced downtime using proactive fault diagnosis and optimized maintenance schedules.

This repository focuses on condition based health monitoring of vehicles and automatic early fault diagnosis by casting this as an anomaly detection problem. Here, the vehicle under test is equipped with five vertical acceleration sensors namely FR_LH (front left side), FR_RH (front right side), RR_LH (rear left side), RR_RH (rear right side) & Cabin. This vehicle is made to run over road surface with alternating rough & smooth patches. The data is segmented into two road surfaces (rough & smooth) along with some preliminary analysis for each of these road surfaces.  

For anomaly detection, Mahalanobis distance and Autoencoder based models are used over these segmented road surfaces. The sensor data from the known healthy vehicles
is used to train and learn useful patterns from this healthy data which models the normal behaviour of vehicles over that surface. Thereafter, unknown vehicle data
over the same surface is tested against the learned normal behaviour. Any data point deviating from this normal behaviour beyond the selected threshold is marked as an
anomaly in the test data assuming these deviating patterns in the data are a result of a fault condition. It is very difficult to learn from raw vibration signals because of the presence of noise and transients. Therefore, time domain features which are known to explain various statistical properties of the data are extracted from these raw signals. After feature extraction, feature selection is done using Principal component analysis (PCA) to derive the most significant features influencing the output. This transformed feature space is then used as an input to our Mahalanobis distance based anomaly detection algorithm. In Mahalanobis distance based method, two models are developed one is single sensor model which considers inputs from any one sensor at a time and another is multiple sensor model which considers inputs from all the sensors together. However, for the Autoencoder network, feature extraction and feature selection are not needed as it is itself said to learn useful representation from vibration data. Therefore, the Autoencoder is trained on smoothed vibration data from a healthy vehicle and is used to determine whether the test data from an unknown vehicle is normal or anomalous.

In this experiment, anomaly detection is performed on two datasets, ”Feb” and ”March”, belonging to the same vehicle which are segmented into rough surface data and smooth surface data. One of them, the ”Feb” dataset is confirmed to be from the healthy vehicle. The ”March” dataset is known to have anomalies as a breakdown occurred in the front part of the vehicle while this dataset was recorded. So it is expected that unusual patterns reflecting breakdown might have been captured in the raw vibration signals. Here, a dataset known to be from a healthy vehicle is used to train (Train Dataset) the model and the ”Feb” dataset (healthy termed as Test1 Dataset) and the ”March” dataset (unhealthy termed as Test2 Dataset) is used as test datasets.

The following provides details of the data and code files used for anomaly detection algorithms. Raw input data files used has been pre-processed and transformed suitably for them to be used as inputs for anomaly detection algorithms.
•	There are two types of files: Jupyter (ipynb) notebooks and csv files.
	o	All the ipynb files consists of python code files written for specific tasks as indicated in the nomenclature of each file.
	o	Some of the csv files are used as input data files and the others are used to save output data.

•	There are total of 6 ipynb files, each for three anomaly detection approaches (Multiple Sensor Mahalanobis Distance, Single Sensor Mahalanobis Distance & Autoencoder) used over two road surfaces (Smooth & Rough).
	o	Anomaly_Detection_Smooth_PCA_Mahalanobis_Distance_Multiple_Sensor_Approach – Multiple senor Mahalanobis distance model training and testing with feature engineering & feature selection over smooth road surface.
	o	Anomaly_Detection_Rough_PCA_Mahalanobis_Distance_Multiple_Sensor_Approach – Multiple senor Mahalanobis distance model training and testing with feature engineering & feature selection over rough road surface.
	o	Anomaly_Detection_Smooth_Mahalanobis_Distance_Single_Sensor_Approach – Single senor Mahalanobis distance model training and testing with feature engineering over smooth road surface.
	o	Anomaly_Detection_Rough_Mahalanobis_Distance_Single_Sensor_Approach – Single sensor Mahalanobis distance model training and testing with feature engineering over rough road surface.
	o	Anomaly_Detection_Smooth_Multivariate_Autoencoder – Autoencoder model training and testing over smooth road surface.
	o	Anomaly_Detection_Rough_Multivariate_Autoencoder – Autoencoder model training and testing over rough road surface.

•	Out of all the csv files, 6 files are used as input files (Input_Data_Files) to our above explained notebooks.
	o	SmoothDataTrain - Training vertical acceleration sensors input data from distinct subset of Feb vehicle over smooth road surface (Train)
	o	SmoothDataTest_Feb - Testing vertical acceleration sensors input data from Feb vehicle (healthy) over smooth road surface (Test1)
	o	SmoothDataTest_March - Testing vertical acceleration sensors input data from March vehicle (faulty) over smooth road surface (Test2)
	o	RoughDataTrain - Training vertical acceleration sensors input data from distinct subset of Feb vehicle over rough road surface (Train)
	o	RoughDataTest_Feb - Testing vertical acceleration sensors input data from Feb vehicle (healthy) over rough road surface (Test1)
	o	RoughDataTest_March – Testing vertical acceleration sensors input data from March vehicle (faulty) over rough road surface (Test2)

•	Rest of the csv files are output (Output_Data_Files) from the above stated Jupyter notebooks. These are basically generated in code to finally draw a comparison between the two test sets for each of the mahalanobis distance-based anomaly detection approaches.
	o	Multivariate_MahaDist_Smooth_With PCA_Feb - Multiple sensor Mahalanobis distances calculated for Feb vehicle (healthy termed as Test1) data over smooth road surface.
	o	Multivariate_MahaDist_Smooth_With PCA_March - Multiple sensor Mahalanobis distances calculated for March vehicle (unhealthy termed as Test2) data over smooth road surface.
	o	Multivariate_MahaDist_Rough_With PCA_Feb - Multiple sensor Mahalanobis distances calculated for Feb vehicle (healthy termed as Test1) data over rough road surface.
	o	Multivariate_MahaDist_Rough_With PCA_March - Multiple sensor Mahalanobis distances calculated for March vehicle (unhealthy termed as Test2) data over rough road surface.
	o	MahaDist_Smooth_Feb – Single sensor Mahalanobis distances calculated for Feb vehicle (healthy termed as Test1) data over smooth road surface. 
	o	MahaDist_Smooth_March - Single sensor Mahalanobis distances calculated for March vehicle (unhealthy termed as Test2) data over smooth road surface.
	o	MahaDist_Rough_Feb - Single sensor Mahalanobis distances calculated for Feb vehicle (healthy termed as Test1) data over rough road surface.
	o	MahaDist_Rough_March - Single sensor Mahalanobis distances calculated for March vehicle (unhealthy termed as Test2) data over rough road surface.
