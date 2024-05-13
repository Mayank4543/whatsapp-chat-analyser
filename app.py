import streamlit as st
import pandas as pd
import preprocessor
import helper
import sentiment
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import requests
import numpy as np
import pymongo
from streamlit_lottie import st_lottie

# MongoDB setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["feedback_database"]
collection = db["feedback_collection"]

# Streamlit page configuration
st.set_page_config(layout='wide', page_title='Whatsapp Chat Analyzer')
st.sidebar.title("Whatsapp Chat Analyzer")
plt.rcParams['font.family'] = 'Arial'

# File upload
uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    # Convert to string
    data = bytes_data.decode('utf-8')
    df = preprocessor.preprocess(data)

    # Unique users
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show analysis"):
        st.snow()
        
        num_messages, words, num_media_messages, nums_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)

        with col2:
            st.header("Total Words")
            st.title(words)

        with col3:
            st.header("Total images")
            st.title(num_media_messages)

        with col4:
            st.header("Links shared")
            st.title(nums_links)

        # Monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='red')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily timeline
        st.title("Daily timeline")
        timeline1 = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline1['only_date'], timeline1['message'], color='green')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Activity Map
        st.title("Activity Map")
        col1, col2 = st.columns(2)
        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            custom_colors = ['#F2C53D', '#6BBE45', '#2AABE2', '#F95D6A', '#A64AC9', '#E57373', '#47D1B5']
            ax.bar(busy_day.index, busy_day.values, color=custom_colors)

            plt.xticks(rotation=45)
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color=['#F2C53D', '#6BBE45', '#2AABE2', '#F95D6A',
                                                                '#A64AC9', '#E57373', '#42A5F5', '#FF9800',
                                                                '#4CAF50', '#FF5722', '#9C27B0', '#795548'])
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # Heatmap
        st.title("Activity Heatmap")
        activity_map = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        ax = sns.heatmap(activity_map, cmap='viridis')
        st.pyplot(fig)

        # Finding the busiest user in the group
        if selected_user == 'Overall':
            st.title('Most Busy Users 🗽')
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values,color=['#F2C53D', '#6BBE45', '#2AABE2', '#F95D6A',
                                                                '#A64AC9', '#E57373', '#42A5F5', '#FF9800',
                                                                '#4CAF50', '#FF5722', '#9C27B0', '#795548'])
                plt.xticks(rotation=45)
                st.pyplot(fig)

            with col2:
                st.dataframe(new_df)

        # Wordcloud
        st.title("Word Cloud ☁")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # Most common word
        mdf = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.barh(mdf[0], mdf[1])
        plt.xticks(rotation='vertical')
        st.title('Most common words')
        st.pyplot(fig)

        # Emoji analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")
        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)

        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df['Count'].head(),labels= emoji_df['Emoji'].head(), autopct="%0.2f")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # Sentiment Analysis
    if st.sidebar.button("Sentiment Analysis"):
        st.title("Sentiment Analysis")
        df_sentiment = sentiment.polarity_score(selected_user, df)
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(df_sentiment)
        with col2:
            l = [np.mean(df_sentiment['Negative']), np.mean(df_sentiment['Neutral']),
                 np.mean(df_sentiment['Positive'])]
            labels = ['Negative', 'Neutral', 'Positive']
            fig, ax = plt.subplots(figsize=(4, 6))
          
            ax.pie(l, labels=labels, autopct='%1.1f%%', startangle=140, colors=['red', 'gray', 'green'])
            ax.axis('equal')
            st.pyplot(fig)
    if st.sidebar.button("Feedback"):

     st.sidebar.title("Feedback")
    full_name = st.text_input("Full Name", key='full_name')
    email = st.text_input("Email", key='email')
    feedback_text = st.text_area("Please share your feedback here:", key='feedback_text')
    # Store feedback data in MongoDB
    if st.button("submit"):

       feedback_data = {
        "Full Name": full_name,
        "Email": email,
        "Feedback": feedback_text
       }
    # Insert feedback data into MongoDB collection
       collection.insert_one(feedback_data)

       full_name = ""
       email = ""
       feedback_text = ""
    
       st.success("Thank you for your feedback!")

    

         
        

else:
    url = "https://assets9.lottiefiles.com/packages/lf20_M9p23l.json"
    lottie_json = requests.get(url).json()
    st_lottie(lottie_json, loop=True)

    
# Feedback Form
