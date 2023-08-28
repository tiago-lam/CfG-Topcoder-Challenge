## Topcoder - Crowd For Good - Breast Cancer Prediction Challenge

Topcoder organized a [Data Science / Machine Learning](https://www.topcoder.com/challenges/09f40d2a-29f2-4952-b846-87e9717c34dc) project to classify the risk of breast cancer development. 
The challenge happened in October 2022, the month selected by the United Nations to remember the world every year about the fight against breast cancer. 

## Models

In this repository, you can find the entries I submitted to the challenge. 
Since it is a classification problem, I started with a simple Logistic Regression model, and then moved to a XGBoost. 
The final result got 96% accuracy. 13th place out of around 500 competitors. 
Topcoder community designed a badge to award the top finalist, mine is shown belown.

![img](https://github.com/tiago-lam/CfG-Topcoder-Challenge/blob/main/topcoder_badge.png)

## How to run

The easiest way to run the models and reproduce the results is using Docker.

- Open your terminal and point to any one of the following folders 
	- [Logistic Regression](https://github.com/tiago-lam/CfG-Topcoder-Challenge/tree/main/submission/Logistic%20Regression)
	- [XGBoost](https://github.com/tiago-lam/CfG-Topcoder-Challenge/tree/main/submission/XGBoost)
- Then go to the folder "code" and type: `docker image build -t image_name .` image_name can be any name you want to
- After the image is created, type: `docker run --rm -it --entrypoint bash image_name` 
- For training, type: `./train.sh data/training.csv`
- After training is concluded, type `./test.sh data/testing.csv data/solution.csv` for testing. Final results will be available at `data/solution.csv`

## Future work

- Main goal: reduce incidence of false negatives
- Try a categorical booster
- Data processing may require an extra category for the 
