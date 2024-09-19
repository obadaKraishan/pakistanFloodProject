import pandas as pd
import json
import ast
import re
import html
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from datetime import datetime
import pytz
from ast import literal_eval


class DataPreprocessor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.tweets_df = None
        self.stop_words = set(stopwords.words('english'))
        self.ps = PorterStemmer()

    def load_data(self):
        print("Loading data...")
        try:
            self.tweets_df = pd.read_csv(self.dataset_path, encoding='utf-8')
        except UnicodeDecodeError:
            print("UTF-8 encoding failed. Trying with ISO-8859-1...")
            self.tweets_df = pd.read_csv(self.dataset_path, encoding='ISO-8859-1')
        print("Data loaded successfully.")

    def handle_missing_values(self):
        print("Handling missing values...")
        self.tweets_df['hashtags'] = self.tweets_df['hashtags'].fillna('NoHashtag')
        print("Missing values handled.")

    def extract_json_fields(self):
        print("Extracting JSON fields...")

        def extract_followers_count(user_str):
            try:
                user_dict = json.loads(user_str)
            except json.JSONDecodeError:
                try:
                    user_dict = ast.literal_eval(user_str)
                except ValueError:
                    return 0
            return user_dict.get('followersCount', 0)

        self.tweets_df['user_followersCount'] = self.tweets_df['user'].apply(extract_followers_count)
        print("JSON fields extracted.")

    def convert_dates(self):
        print("Converting dates to datetime format...")
        self.tweets_df['date'] = pd.to_datetime(self.tweets_df['date'])
        print("Dates converted.")

    def clean_encoding_issues(self, text):
        # Decode HTML entities
        text = html.unescape(text)

        # Attempt to replace common problematic sequences
        text = re.sub(r'[‰Û¢å£‰Ý¼•ü‰Âà•ü_ÙÒ_Ù´÷_Ù_]', '', text)

        # Remove URLs
        text = re.sub(r'http\S+', '', text)

        # Remove any remaining non-ASCII characters
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)

        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def clean_text(self, text):
        text = self.clean_encoding_issues(text)
        words = word_tokenize(text)
        cleaned_words = [self.ps.stem(w) for w in words if w.lower() not in self.stop_words and w.isalpha()]
        return ' '.join(cleaned_words)

    def preprocess_text(self):
        print("Preprocessing text data...")
        self.tweets_df['cleaned_content'] = self.tweets_df['content'].apply(self.clean_text)
        print("Text data preprocessed.")

    def expand_user_details(self):
        print("Expanding user details into separate columns...")

        # Pre-process the 'user' column to convert string representations to dictionaries
        def preprocess_user_string(user_str):
            try:
                # Convert datetime strings to an evaluable form by removing the datetime part
                user_str_processed = re.sub(r'datetime\.datetime\([^)]+\)', 'None', user_str)
                user_dict = literal_eval(user_str_processed)
                return user_dict
            except Exception as e:
                print(f"Error processing user string: {e}")
                return {}

        self.tweets_df['user_processed'] = self.tweets_df['user'].apply(preprocess_user_string)

        # Normalize and expand user details
        user_details_df = pd.json_normalize(self.tweets_df['user_processed'])

        # Combine the expanded user details with the original DataFrame
        self.tweets_df = pd.concat([self.tweets_df.drop(['user', 'user_processed'], axis=1), user_details_df], axis=1)
        print("User details expanded.")

    def format_hashtags(self):
        print("Formatting hashtags...")

        def format_hashtag_list(hashtag_str):
            # Convert string representation of list to list
            # Check if the hashtag_str is already a list (it might not need conversion)
            if isinstance(hashtag_str, str):
                hashtag_list = ast.literal_eval(hashtag_str)
            else:
                hashtag_list = hashtag_str  # Assuming it's already a list

            # Prepend # to each hashtag and join them with commas
            formatted_hashtags = ', '.join([f"#{tag}" for tag in hashtag_list])
            return formatted_hashtags

        self.tweets_df['hashtags'] = self.tweets_df['hashtags'].apply(format_hashtag_list)
        print("Hashtags formatted.")

    def extract_media_fullUrls(self):
        print("Extracting media fullUrls...")

        def extract_fullUrl(media_entry):
            # Check if the entry is NaN (float in pandas) or None
            if pd.isna(media_entry):
                return None  # or return an empty string '', based on your preference

            try:
                # If the entry is a string, convert it to a list of dictionaries
                if isinstance(media_entry, str):
                    media_list = ast.literal_eval(media_entry)
                else:
                    media_list = media_entry  # Assuming it's already the correct format

                # Extract 'fullUrl' from each dictionary in the list
                fullUrls = [media['fullUrl'] for media in media_list if 'fullUrl' in media]

                # Join the URLs into a single string, separated by commas
                return ', '.join(fullUrls)
            except Exception as e:
                print(f"Error processing media entry: {e}")
                return None  # or return an empty string '', based on your preference

        self.tweets_df['media'] = self.tweets_df['media'].apply(extract_fullUrl)
        print("Media fullUrls extracted.")

    def format_tcooutlinks(self):
        print("Formatting tcooutlinks...")

        def format_link(tcooutlink_str):
            # Check if the entry is not NaN (which in pandas is float)
            if pd.isna(tcooutlink_str):
                return None  # or '', depending on how you want to handle NaNs

            # If the entry is a string, evaluate it to a list
            if isinstance(tcooutlink_str, str):
                link_list = ast.literal_eval(tcooutlink_str)
            else:
                link_list = tcooutlink_str  # Assuming it's already a list

            # Return the first link if the list is not empty, else return None or ''
            return link_list[0] if link_list else None

        self.tweets_df['tcooutlinks'] = self.tweets_df['tcooutlinks'].apply(format_link)
        print("tcooutlinks formatted.")

    def format_outlinks(self):
        print("Formatting outlinks...")

        def format_link(outlink_str):
            # Check if the entry is not NaN
            if pd.isna(outlink_str):
                return None  # or '', depending on how you want to handle NaNs

            # If the entry is a string, evaluate it to a list
            if isinstance(outlink_str, str):
                link_list = ast.literal_eval(outlink_str)
            else:
                link_list = outlink_str  # Assuming it's already a list

            # Return the first link if the list is not empty, else return None or ''
            return link_list[0] if link_list else None

        self.tweets_df['outlinks'] = self.tweets_df['outlinks'].apply(format_link)
        print("Outlinks formatted.")

    def format_mentionedUsers(self):
        print("Formatting mentionedUsers...")

        def format_user_list(mentionedUsers_str):
            # Check if the entry is not NaN
            if pd.isna(mentionedUsers_str):
                return None  # or '', depending on how you want to handle NaNs

            # Convert string representation of list to actual list
            if isinstance(mentionedUsers_str, str):
                mentionedUsers_list = ast.literal_eval(mentionedUsers_str)
            else:
                mentionedUsers_list = mentionedUsers_str  # Assuming it's already a list

            # Extract and format username, id, and displayname for each user
            formatted_users = []
            for user in mentionedUsers_list:
                formatted_user = f"username: {user['username']}, id: {user['id']}, displayname: {user['displayname']}"
                formatted_users.append(formatted_user)

            # Join all formatted user strings with a semicolon
            return '; '.join(formatted_users)

        self.tweets_df['mentionedUsers'] = self.tweets_df['mentionedUsers'].apply(format_user_list)
        print("mentionedUsers formatted.")

    def format_inReplyToUser(self):
        print("Formatting inReplyToUser...")

        def format_user(user_str):
            if pd.isna(user_str) or not isinstance(user_str, str):
                return None  # Handle missing or non-string data

            # Regular expression to match and extract the relevant fields
            username_match = re.search(r"'username': '([^']*)'", user_str)
            id_match = re.search(r"'id': (\d+)", user_str)
            displayname_match = re.search(r"'displayname': '([^']*)'", user_str)

            # Extracting the matched groups if they exist
            username = username_match.group(1) if username_match else 'N/A'
            user_id = id_match.group(1) if id_match else 'N/A'
            displayname = displayname_match.group(1) if displayname_match else 'N/A'

            # Formatting the extracted information
            formatted = f"username: {username}, id: {user_id}, displayname: {displayname}"
            return formatted

        self.tweets_df['inReplyToUser'] = self.tweets_df['inReplyToUser'].apply(format_user)
        print("inReplyToUser formatted.")

    def preprocess(self):
        print("Starting preprocessing...")
        self.load_data()
        self.handle_missing_values()
        self.format_hashtags()
        self.extract_media_fullUrls()
        self.format_tcooutlinks()
        self.format_outlinks()
        self.format_mentionedUsers()
        self.format_inReplyToUser()
        self.extract_json_fields()
        self.convert_dates()
        self.preprocess_text()
        self.expand_user_details()
        self.tweets_df.fillna("NA", inplace=True)
        print("Data preprocessing completed.")

    def save_preprocessed_data(self, preprocessed_path):
        print(f"Saving preprocessed data to {preprocessed_path}...")
        self.tweets_df.to_csv(preprocessed_path, index=False, encoding='utf-8')
        print("Preprocessed data saved successfully.")


# Example usage
if __name__ == "__main__":
    dataset_path = '/Volumes/Kraishan 1/TTU//Thesis/NEW/Pakistan/PakistanFloodsAppeal-tweets.csv'  # Replace with your dataset's path
    preprocessed_path = '/Volumes/Kraishan 1/TTU//Thesis/NEW/Pakistan/ProcessedFloodsInPakistan-tweets.csv'  # Desired path for the preprocessed data

    preprocessor = DataPreprocessor(dataset_path)
    preprocessor.preprocess()  # Perform preprocessing
    preprocessor.save_preprocessed_data(preprocessed_path)  # Save the preprocessed data
