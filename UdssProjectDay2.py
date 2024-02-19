#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# installing dependencies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[4]:


data = pd.read_csv("plant_yield.csv")


# In[6]:


data.head()


# In[7]:


data.tail()


# In[8]:


data.columns


# In[9]:


data.shape


# In[10]:


data.describe()


# In[11]:


data.isnull().sum()


# In[14]:


#visualization
data.head()


# In[16]:


data.columns


# In[17]:


data = data[['Year','average_rain_fall_mm_per_year','pesticides_tonnes','avg_temp','plant_yield']]


# In[18]:


data.columns


# In[20]:


x = np.array(data.iloc[:, :-1])
y = np.array(data['plant_yield'])


# In[23]:


# Create scatter plots for each feature against the target variable (Sales)
plt.figure(figsize=(12, 6))
plt.subplot(1, 4, 1)
sns.scatterplot(x='Year', y='plant_yield', data=data)
plt.title('Year vs plant_yield')

plt.subplot(1, 4, 2)
sns.scatterplot(x='average_rain_fall_mm_per_year', y='plant_yield', data=data)
plt.title('average_rain_fall_mm_per_year vs plant_yield')

plt.subplot(1, 4, 3)
sns.scatterplot(x='pesticides_tonnes', y='plant_yield', data=data)
plt.title('pesticides_tonnes vs plant_yield')

plt.subplot(1, 4, 4)
sns.scatterplot(x='avg_temp', y='plant_yield', data=data)
plt.title('avg_temp vs plant_yield')

plt.tight_layout()
plt.show()


# In[25]:


# Create a correlation matrix
corr_matrix = data.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 12})
plt.title('Correlation Matrix')
plt.show()


# In[26]:


print(corr_matrix["plant_yield"].sort_values(ascending=False))


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Building the model

# In[29]:


#Import the linear Regression Model
from sklearn.linear_model import LinearRegression


# In[30]:


#Initialize the Linear Regression Model
Lin_Regres = LinearRegression()


# In[31]:


#Fit the Linear Regression Model to the split dataset
Lin_Regres.fit(X_train, y_train)


# In[33]:


#To import another model called Random Forest Regressor, this will enable us to choose the better model to use for our plant_yield prediction
from sklearn.ensemble import RandomForestRegressor


# In[34]:


#To initialize another model called Random Forest Regressor
rf_regressor = RandomForestRegressor()


# In[35]:


#Fit the Random Forest Regressor Model to the split dataset
rf_regressor.fit(X_train, y_train)


# In[36]:


#To see the accuracy rate of Linear regression model with the test dataset

print(Lin_Regres.score(X_test, y_test))


# In[37]:


#To also see the accuracy rate of the Random Forest Regressor Model with the same test dataset
print(rf_regressor.score(X_test, y_test))


# In[38]:


# Get feature importance from the trained RandomForestRegressor
feature_importances = rf_regressor.feature_importances_

# Sort feature importance in descending order
indices = np.argsort(feature_importances)

# Rearrange feature names based on feature importance
feature_names = data.columns
sorted_feature_names = [feature_names[i] for i in indices]

# Plot feature importances horizontally
plt.figure(figsize=(10, 6))
plt.title("Feature Importance - RandomForestRegressor")
plt.barh(range(len(feature_importances)), feature_importances[indices])
plt.yticks(range(len(feature_importances)), sorted_feature_names)
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()


# In[39]:


print(X_test)


# In[40]:


#Import the evaluation metrics first
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[45]:


# Compute evaluation metrics
mse = mean_squared_error(y_test, y_test)
mae = mean_absolute_error(y_test, y_test)
r2 = r2_score(y_test, y_test)

# Print evaluation metrics
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2): {r2:.2f}")


# In[46]:


# Make predictions on the test set
y_pred = rf_regressor.predict(X_test)


# In[49]:


print(y_pred)


# In[ ]:





# In[ ]:




