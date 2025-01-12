# Anomaly Detection Using Machine Learning: An Application to Automated Vehicle Health Monitoring
Detecting anomalies has played a crucial role in spotting irregular patterns in data that may signal system health decline, possible component failures, or random defects. By employing machine learning techniques for anomaly detection, which capture new trends while preserving historical data, continuous monitoring of these systems is enhanced. This approach leverages technological advancements to achieve better control, increased reliability, and minimized downtime through proactive fault diagnosis and optimized maintenance planning.

This repository is dedicated to monitoring the health of vehicles based on their condition and automatically diagnosing early faults by treating this as an anomaly detection problem. Here, the vehicle under test is equipped with five vertical acceleration sensors namely FR_LH (front left), FR_RH (front right), RR_LH (rear left), RR_RH (rear right) & Cabin. This vehicle is made to run over alternating rough & smooth road surfaces. The entire data is segmented into these two rough & smooth road surfaces along with some preliminary analysis for each of these road surfaces.  

For anomaly detection, models based on Mahalanobis distance and Autoencoders are applied to these segmented road surfaces. Sensor data from a known healthy vehicle is used to train and identify meaningful patterns, modeling the normal behavior of a vehicle on that surface. Subsequently, data from an unknown vehicle on the same surface is tested against this learned behavior which is known to be normal. Any data point that deviates beyond a set threshold is flagged as an anomaly, assuming these deviations indicate a fault. Due to the presence of noise and transients, learning from raw vibration signals is challenging. Therefore, time domain features, which explain various statistical properties of the data are extracted. After feature extraction, Principal Component Analysis (PCA) is used for feature selection to identify the most influencing features determining the output. Mahalanobis distance-based anomaly detection algorithm then utilizes this feature space as input. In this method, two models are developed: a single sensor model, which considers inputs from one sensor at a time, and a multiple sensor model, which considers inputs from all sensors together. For the Autoencoder model, feature extraction and selection are not required, as it derives meaningful representations from the vibration data itself. The Autoencoder model is trained on vibration data coming from a healthy vehicle which is then used to determine whether the unknown vehicle's test data is normal or anomalous.

In this study, anomaly detection is conducted on two datasets, 'Feb' and 'March', from the same vehicle, segmented into rough and smooth surface data. The 'Feb' dataset is known to be from a healthy vehicle. The 'March' dataset contains anomalies, as a breakdown occurred in the vehicle's front part during its recording. It is expected that the raw vibration signals might capture unusual patterns indicating the breakdown. A dataset coming from a healthy vehicle is used to train the model (Train Data), while the 'Feb' dataset (healthy vehicle, called as Test1 Dataset) and the 'March' dataset (unhealthy vehicle, called as Test2 Dataset) are used to test the model.

The details of all the concepts used in this study along with performed experiments and their results are explained in Chapter "Anomaly Detection Using Machine Learning: An Application to Automated Vehicle Health Monitoring" as a part of a book titled "Data Science for Decision Making".

The following provides details of the data and code files used for anomaly detection algorithms. Raw input data files used has been pre-processed and transformed suitably for them to be used as inputs for anomaly detection algorithms.

There are two types of files: Jupyter (ipynb) notebooks and csv files. References to respective sections of the mentioned chapter are provided below: 

	o	All the ipynb files consists of python code files written for specific tasks as indicated in the nomenclature of each file.
	o	Some of the csv files are used as input data files and the others are used to save output data.

There are a total of 6 ipynb files, each for three anomaly detection approaches (Multiple Sensor Mahalanobis Distance, Single Sensor Mahalanobis Distance & Autoencoder) used over two road surfaces (Smooth & Rough).

 	o	Anomaly_Detection_Smooth_PCA_Mahalanobis_Distance_Multiple_Sensor_Approach – Multiple senor Mahalanobis distance model training and testing with feature engineering & feature selection over smooth road surface (Refer Section 4.1 & 5.1).
	o	Anomaly_Detection_Rough_PCA_Mahalanobis_Distance_Multiple_Sensor_Approach – Multiple senor Mahalanobis distance model training and testing with feature engineering & feature selection over rough road surface (Refer Section 4.1 & 5.1).
	o	Anomaly_Detection_Smooth_Mahalanobis_Distance_Single_Sensor_Approach – Single senor Mahalanobis distance model training and testing with feature engineering over smooth road surface (Refer Section 4.1 & 5.2).
	o	Anomaly_Detection_Rough_Mahalanobis_Distance_Single_Sensor_Approach – Single sensor Mahalanobis distance model training and testing with feature engineering over rough road surface (Refer Section 4.1 & 5.2).
	o	Anomaly_Detection_Smooth_Multivariate_Autoencoder – Autoencoder model training and testing over smooth road surface (Refer Section 4.2 & 5.3).
	o	Anomaly_Detection_Rough_Multivariate_Autoencoder – Autoencoder model training and testing over rough road surface (Refer Section 4.2 & 5.3).

Out of all the csv files, 6 files are used as input files (Input_Data_Files) to our above explained notebooks.
	
 	o	SmoothDataTrain - Training vertical acceleration sensors input data from distinct subset of Feb vehicle over smooth road surface (Train).
	o	SmoothDataTest_Feb - Testing vertical acceleration sensors input data from Feb vehicle (healthy) over smooth road surface (Test1).
	o	SmoothDataTest_March - Testing vertical acceleration sensors input data from March vehicle (faulty) over smooth road surface (Test2).
	o	RoughDataTrain - Training vertical acceleration sensors input data from distinct subset of Feb vehicle over rough road surface (Train).
	o	RoughDataTest_Feb - Testing vertical acceleration sensors input data from Feb vehicle (healthy) over rough road surface (Test1).
	o	RoughDataTest_March – Testing vertical acceleration sensors input data from March vehicle (faulty) over rough road surface (Test2).

Rest of the csv files are output (Output_Data_Files) from the above stated Jupyter notebooks. These are basically generated in code to finally draw a comparison between the two test sets for each of the mahalanobis distance-based anomaly detection approaches.
	
 	o	Multivariate_MahaDist_Smooth_With PCA_Feb - Multiple sensor Mahalanobis distances calculated for Feb vehicle (healthy termed as Test1) data over smooth road surface.
	o	Multivariate_MahaDist_Smooth_With PCA_March - Multiple sensor Mahalanobis distances calculated for March vehicle (unhealthy termed as Test2) data over smooth road surface.
	o	Multivariate_MahaDist_Rough_With PCA_Feb - Multiple sensor Mahalanobis distances calculated for Feb vehicle (healthy termed as Test1) data over rough road surface.
	o	Multivariate_MahaDist_Rough_With PCA_March - Multiple sensor Mahalanobis distances calculated for March vehicle (unhealthy termed as Test2) data over rough road surface.
	o	MahaDist_Smooth_Feb – Single sensor Mahalanobis distances calculated for Feb vehicle (healthy termed as Test1) data over smooth road surface. 
	o	MahaDist_Smooth_March - Single sensor Mahalanobis distances calculated for March vehicle (unhealthy termed as Test2) data over smooth road surface.
	o	MahaDist_Rough_Feb - Single sensor Mahalanobis distances calculated for Feb vehicle (healthy termed as Test1) data over rough road surface.
	o	MahaDist_Rough_March - Single sensor Mahalanobis distances calculated for March vehicle (unhealthy termed as Test2) data over rough road surface.
