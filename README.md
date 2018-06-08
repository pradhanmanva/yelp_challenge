**Yelp Dataset Challenge**

This is an experimental project to predict the rush hours for a specific type of business using the Yelp dataset.

Libraries Used: numpy, pandas, scikit-learn, matplotlib, seaborn, json, os, math

Dataset Used: Yelp Dataset – https://www.yelp.com/dataset/download

**How to run the code**
1. Keep the whole folder intact.
2. Install the above packages using pip. 
2.	For windows users: Open Command Line  type "python -m pip install  <package_name> " 
3.	For Linux users: Open Terminal  type "pip install <package_name>" 
4.	Keep the downloaded data (json files) in the data folder.
5. Run the notebooks and that should show you graphs and results on its own.
6. Run the model_name.py file to view performance of various model with our data.

**CS 6375.004 – Machine Learning**
 
**Project – Predicting the busyness of a business based on reviews and check-ins**
 
**Introduction:**
You have a business that might be doing well. But knowing when you are going to get maximum the crowd or when there is going to be a dry spell is a very important information to know. It prepares the entrepreneur to handle the big crowd during rush hours and alerts the user when should he go to a particular business to avoid a rush. This project will benefit both the user and the customer in shaping their schedule to the business. It will not only save customers’ time but also benefit the business to perform well during the rush hours.

**Dataset details, such as number of features, instances, data distribution (Dataset description)**

The dataset is divided into three files business.json, user.json and review.json.
●	The business.json file describes a business. It contains the name of the business, location of the business,average rating, location and certain features like the category of business and amenities available at the business. There are 156,639 business present in the dataset.
●	The user.json file describes the user base of Yelp. The file contains the user’s name, total reviews given by the user, average rating, friends of the user, compliments for the user. There are 1,183,362 users in the dataset.
●	The review.json file contains the reviews for business, reference to business, reference to the reviewer and the usefulness of the review. The file contains json objects of 4,736,897 reviews.
●	The checkin.json file contains the times over a week when customers have checked-into a business. There are 135,148 check-ins present in the dataset.

**Data Distribution:**

1. Ratings vs No. of Reviews:
This graph maps the total number of reviews for each possible rating value. The dataset has ratings from 1 to 5.

2. Year vs No. of Reviews:
This graph maps the number of reviews that were given to businesses against each year.

3. Year vs No. of Users joining Yelp:
The below graph plots the number of users joining Yelp every year

4. Day of the week vs No. of Check-ins:
The following four graphs show the check-ins of four businesses over a week.

**Planned Techniques:**

The plan to achieve our aim is as follows:
1. Extract the information from the business.json about the features of a business like opening hours, closing hours, dinner, lunch, cafe and other important information using the attribute "attribute"
2.	Mine the details of the checkin.json and find out the rush hours and non-rush hours.
3.	Map the business to their check-in details and prepare the training, testing and validation data.
4.	Create a model using the review, checkin and business data.
5.	Predict the busyness of a business at a given time or day for a given location.

We intend to try out different techniques, in addition to sentiment analysis, like

1.	Artificial Neural Networks Regressor
2.	Bagging Regressor
3.	Stochastic Gradient Descent
4.	Ensemble Methods
5.	AdaBoosting

**Experimental methodology (how you plan to pre-process, create training, validation, and test datasets, and other such details)**

In the pre-processing step, we intend to combine the three json files into a single csv file analogous to joining three tables in a relational database with only the desired attributes. Textual categorical features, if any, will be converted to numerical features. Records with null values for any of its features, if any, will be omitted. All features will be normalized to some equivalent range.
 Lengthy reviews have a lot of unnecessary words and characters like ‘\n’ and stop words. Such words and special characters will be removed, without loss of information, for the sentiment analyser to perform optimally.
On the processed data, we plan to try out the following methods to create training and validation and test data sets and choose the best performing one -

1.	Since the dataset is large enough containing nearly 1M reviews, the hold-out method can provide a simple and fast way to generate data.
2.	K-fold cross-validation with large enough K. Leave-one-out cross validation can be very time consuming to perform on such huge dataset.
3.	Bootstrapping resampling.


**Coding language / technique to be used**

The language used to code the whole project is Python. We intend to use libraries like TensorFlow or Keras, NLTK, scikit-learn, and data processing libraries like Pandas, Numpy and Matplotlib. For sentiment analysis, we have planned to use senti_classifier library. We intend to use MongoDB as the data is in json format.
