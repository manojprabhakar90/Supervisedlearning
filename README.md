**ABOUT**
This is part of MS Data Science Course at University of Colorado Boulder.

**WHY**

This is part of the supervised learning final project (as part of the Machine Learning Supervised Learning course). The package consists of data folder which contains the dataset, jupyter notebooks folder which has the EDA as part of the assignment. Then a DockerFile which provides the list of steps to be carried out for the application to run. 

The model is a RandomForest Model which has been trained with a small dataset. 

**HOW**

To build the model, you can run python model.py to generate the RandomForestModel.pkl file. 

The package management is done through poetry. 

As part of this, I have build the repo as a Docker Image. 

To run the application, you can type docker run -p 8501:8501 streamlittest:latest

**SNAPSHOT:**

![image](https://github.com/user-attachments/assets/563c31b5-7fff-48c0-a0f6-dd77f3bffd92)
