import pandas as pd
import numpy as np
import streamlit as st
import io
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score


imdb = pd.read_csv("IMDb Movies India.csv", encoding='latin-1')
# st.title("Movie Rating Predictions")
# st.write("Through this model, we have attempted to identify a number of factors that may influence a film's ranking.","This model is designed to forecast a future film's rating based on prior ratings that are established using a number of different criteria.")

# print(imdb.shape)
# print(imdb.head())

# st.header("Information about the Dataset")
buffer =io.StringIO()
imdb.info(buf=buffer)
info_str=buffer.getvalue()
# st.text(info_str)


# Checking null values
# print('Null values in the columns:')
# print(imdb.isna().sum())


#Checking if there are any typos

# for col in imdb.select_dtypes(include = "object"):
#     print(f"Name of Column: {col}")
#     print(imdb[col].unique())
#     print('\n', '-'*60, '\n')


# Handling the null values
imdb.dropna(subset=['Name', 'Year', 'Duration', 'Rating', 'Votes', 'Director', 'Actor 1', 'Actor 2', 'Actor 3'], inplace=True)

#Extracting only the text part from the Name column
imdb['Name'] = imdb['Name'].str.extract('([0-9A-Za-z\s\'\-]+)')

# Replacing the brackets from year column as observed above
imdb['Year'] = imdb['Year'].str.replace(r'[()]', '', regex=True).astype(int)

# Convert 'Duration' to numeric and replacing the min, while keeping only numerical part
imdb['Duration'] = pd.to_numeric(imdb['Duration'].str.replace(r' min', '', regex=True), errors='coerce')


# Splitting the genre by , to keep only unique genres and replacing the null values with mode
imdb['Genre'] = imdb['Genre'].str.split(', ')
imdb = imdb.explode('Genre')
imdb['Genre'].fillna(imdb['Genre'].mode()[0], inplace=True)


# Convert 'Votes' to numeric and replace the , to keep only numerical part
imdb['Votes'] = pd.to_numeric(imdb['Votes'].str.replace(',', ''), errors='coerce')


#checking duplicate values by Name and Year
duplicate = imdb.groupby(['Name', 'Year']).filter(lambda x: len(x) > 1)
duplicate.head(5)


# Dropping the duplicated values by Name
imdb = imdb.drop_duplicates(subset=['Name'])

# print(imdb.describe())

# print(imdb.describe(include = 'O'))


# imdb.drop('Name', axis = 1, inplace = True)

# Grouping the columns with their average rating and then creating a new feature

genre_mean_rating = imdb.groupby('Genre')['Rating'].transform('mean')
imdb['Genre_mean_rating'] = genre_mean_rating

director_mean_rating = imdb.groupby('Director')['Rating'].transform('mean')
imdb['Director_encoded'] = director_mean_rating

actor1_mean_rating = imdb.groupby('Actor 1')['Rating'].transform('mean')
imdb['Actor1_encoded'] = actor1_mean_rating

actor2_mean_rating = imdb.groupby('Actor 2')['Rating'].transform('mean')
imdb['Actor2_encoded'] = actor2_mean_rating

actor3_mean_rating = imdb.groupby('Actor 3')['Rating'].transform('mean')
imdb['Actor3_encoded'] = actor3_mean_rating

# Keeping the predictor and target variable

X = imdb[[ 'Year', 'Votes', 'Duration', 'Genre_mean_rating','Director_encoded','Actor1_encoded', 'Actor2_encoded', 'Actor3_encoded']]
y = imdb['Rating']
# st.write("Input Features")
# st.write(X.head())
# st.write("Target Features")
# st.write(y.head())

# Splitting the dataset into training and testing parts

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Building 2 machine learning models and training them

lr = LinearRegression()
lr.fit(X_train,y_train)
lr_pred = lr.predict(X_test)

rf = RandomForestRegressor()
rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test)

# st.header("Evaluating the performance of trained algorithms")
# st.write('The performance evaluation of Logistic Regression is below: ', '\n')
# st.write('Mean squared error: ',mean_squared_error(y_test, lr_pred))
# st.write('Mean absolute error: ',mean_absolute_error(y_test, lr_pred))
# st.write('R2 score: ',r2_score(y_test, lr_pred))

# st.write('The performance evaluation of Random Forest Regressor is below: ', '\n')
# st.write('Mean squared error: ',mean_squared_error(y_test, rf_pred))
# st.write('Mean absolute error: ',mean_absolute_error(y_test, rf_pred))
# st.write('R2 score: ',r2_score(y_test, rf_pred))

# Creating a new dataframe with values close to the 3rd row according to the sample above 
st.header("Movie Rating Prediction")
name = st.text_input("Enter the name of the movie:", key="name_input")
year = st.number_input("Enter the year of the movie:", key="year_input")
votes = st.number_input("Enter the votes of the movie:", key="votes_input")
duration = st.number_input("Enter the duration of the movie:", key="duration_input")
genre = st.text_input("Enter the genre of the movie:", key="genre_input")
director = st.text_input("Enter the director of the movie:", key="director_input")
actor1 = st.text_input("Enter the actor1 of the movie:", key="actor1_input")
actor2 = st.text_input("Enter the actor2 of the movie:", key="actor2_input")
actor3 = st.text_input("Enter the actor3 of the movie:", key="actor3_input")

genre_mean=imdb.groupby('Genre')['Rating'].mean()
director_mean=imdb.groupby('Director')['Rating'].mean()
actor1_mean=imdb.groupby('Actor 1')['Rating'].mean()
actor2_mean=imdb.groupby('Actor 2')['Rating'].mean()
actor3_mean=imdb.groupby('Actor 3')['Rating'].mean()

# Initialize session state for predicted_rating if it doesn't exist
if 'predicted_rating' not in st.session_state:
    st.session_state['predicted_rating'] = 0  # Initialize to 0 or None

# Button to submit movie details and make prediction
submit = st.button("Predict")

if submit:
    # Example: Simulating inputs (you would get these from actual user inputs)
    genre_rating = genre_mean.get(genre, None)
    director_rating = director_mean.get(director, None)
    actor1_rating = actor1_mean.get(actor1, None)
    actor2_rating = actor2_mean.get(actor2, None)
    actor3_rating = actor3_mean.get(actor3, None)

    # Create DataFrame for prediction
    data = {
        'Year': [year],
        'Votes': [votes],
        'Duration': [duration],
        'Genre_mean_rating': [genre_rating],
        'Director_encoded': [director_rating],
        'Actor1_encoded': [actor1_rating],
        'Actor2_encoded': [actor2_rating],
        'Actor3_encoded': [actor3_rating]
    }
    
    df = pd.DataFrame(data)
    st.write(df)

    # Predict the movie rating using the Random Forest model
    try:
        predicted_rating = rf.predict(df)[0]  # Store predicted rating
        st.session_state['predicted_rating'] = predicted_rating  # Store in session state
        st.write("Predicted Rating:", predicted_rating)
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Sentiment-based Rating Adjustment (NLP Processing)
user_review = st.text_area("Enter your review of the movie:", key="review_input")

# Button to submit the review and finalize the rating adjustment
review_submit = st.button("Submit Review")

if review_submit:
    # Fetch the predicted rating from session state
    predicted_rating = st.session_state['predicted_rating']

    # Ensure a rating was predicted before proceeding
    if predicted_rating == 0:
        st.error("Please predict the rating first.")
    else:
        # Check if the user provided a review
        if user_review.strip():  # Check if the review is not just whitespace
            # Perform sentiment analysis on the user review
            blob = TextBlob(user_review)
            sentiment_score = blob.sentiment.polarity
            st.write(f"Adjustment done: {sentiment_score}")

            # Adjust the rating based on the sentiment polarity
            sentiment_adjustment = sentiment_score * 1.5  # You can adjust this factor
            final_rating = predicted_rating + sentiment_adjustment

            # Display the adjusted rating based on the review
            st.write(f"Adjusted Rating based on review: {final_rating}")
        else:
            final_rating = predicted_rating
            st.write(f"No review provided, using the predicted rating as final rating: {final_rating}")

        # Check if the movie exists in the dataset
        st.write(f"Checking for movie: {name} ({year})")
        movie_exists = imdb[(imdb['Name'] == name) & (imdb['Year'] == year)]
        st.write(f"Movie Exists: {not movie_exists.empty}")

        if not movie_exists.empty:
            # Update the rating if the movie exists
            imdb.loc[movie_exists.index, 'Rating'] = final_rating
            st.write("Rating updated in the dataset!")
        else:
            # Append the new movie to the dataset
            st.write("Movie does not exist, adding a new entry.")
            new_movie_data = {
                'Name': name,
                'Year': year,
                'Votes': votes,
                'Duration': duration,
                'Genre': genre,
                'Director': director,
                'Actor 1': actor1,
                'Actor 2': actor2,
                'Actor 3': actor3,
                'Rating': final_rating
            }
            new_movie_df = pd.DataFrame([new_movie_data])
            imdb = pd.concat([imdb, new_movie_df], ignore_index=True)
            st.write("Added the new movie with the predicted rating to the dataset.")

        # Optionally, save the updated dataset to a file
        imdb.to_csv("IMDb_Movies_Updated.csv", index=False)
        st.write("Dataset has been updated and saved.")

