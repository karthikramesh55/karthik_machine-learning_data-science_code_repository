## Implementing K-Nearest Neighbours Model on AirBnB Listings To Determine the Pricing Strategy of Hosts ##

import pandas as pd
import numpy as np
np.random.seed(1)

dc_listings = pd.read_csv('dc_airbnb.csv')
dc_listings = dc_listings.loc[np.random.permutation(len(dc_listings))]
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')

dc_listings.info()

## 2. Removing features ##

import pandas as pd 

dc_listings = dc_listings.drop(['room_type', 'city', 'state', 'latitude', 'longitude', 'zipcode', 'host_acceptance_rate', 'host_response_rate', 'host_listings_count'], axis=1)

print(dc_listings.isnull().sum())

## 3. Handling missing values ##

dc_listings = dc_listings.drop(['cleaning_fee','security_deposit'], axis=1)
dc_listings = dc_listings.dropna(axis=0)

print(dc_listings.isnull().sum())

## 4. Normalize columns ##

normalized_listings = (dc_listings - dc_listings.mean()) / (dc_listings.std())
normalized_listings['price']  = dc_listings['price']
print(normalized_listings.head(3))

## 5. Euclidean distance for multivariate case ##

from scipy.spatial import distance
first_listing = normalized_listings.iloc[0][['accommodates','bathrooms']]
fifth_listing = normalized_listings.iloc[4][['accommodates','bathrooms']]

first_fifth_distance = distance.euclidean(first_listing, fifth_listing)
print(first_fifth_distance)

## 7. Fitting a model and making predictions ##

from sklearn.neighbors import KNeighborsRegressor

train_df = normalized_listings.iloc[0:2792]
test_df = normalized_listings.iloc[2792:]

train_features = train_df[['accommodates', 'bathrooms']]
train_target = train_df['price']

knn = KNeighborsRegressor(n_neighbors = 5, algorithm = 'brute')
knn.fit(train_features,train_target)
predictions = knn.predict(test_df[['accommodates', 'bathrooms']])
print(predictions)


## 8. Calculating MSE using Scikit-Learn ##

from sklearn.metrics import mean_squared_error

train_columns = ['accommodates', 'bathrooms']
knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute', metric='euclidean')
knn.fit(train_df[train_columns], train_df['price'])
predictions = knn.predict(test_df[train_columns])

two_features_mse = mean_squared_error(test_df['price'], predictions)
two_features_rmse = (two_features_mse)**(0.5)
print(two_features_mse)
print(two_features_rmse)

## 9. Using more features ##

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

features = ['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']

knn = KNeighborsRegressor(n_neighbors = 5, algorithm = 'brute', metric = 'euclidean')
knn.fit(train_df[features], train_df['price'])
four_predictions = knn.predict(test_df[features])

four_mse = mean_squared_error(test_df['price'], four_predictions)
four_rmse = (four_mse)**(0.5)

print(four_mse)
print(four_rmse)



## 10. Using all features ##

knn = KNeighborsRegressor(n_neighbors=5, algorithm='brute')

features = train_df.columns.tolist()
features.remove('price')

knn.fit(train_df[features], train_df['price'])
all_features_predictions = knn.predict(test_df[features])
all_features_mse = mean_squared_error(test_df['price'], all_features_predictions)
all_features_rmse = all_features_mse ** (1/2)
print(all_features_mse)
print(all_features_rmse)