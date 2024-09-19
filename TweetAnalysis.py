import pandas as pd
import chardet
from collections import Counter
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer


class TweetAnalysis:
    def __init__(self, processed_dataset_path):
        self.dataset_path = processed_dataset_path
        self.tweets_df = self.load_processed_data()

    def load_processed_data(self):
        """Load the processed dataset from an Excel file."""
        try:
            return pd.read_excel(self.dataset_path)
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            return None

    def keyword_extraction(self):
        """Perform keyword extraction using TF-IDF."""
        tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(
            self.tweets_df['cleaned_content'])  # Ensure 'cleaned_content' is present
        keywords = tfidf_vectorizer.get_feature_names_out()
        print("Extracted Keywords:", keywords[:10])  # Display top 10 keywords as example

    def hashtag_analysis(self):
        """Analyze hashtags for trending topics."""
        all_hashtags = ','.join(self.tweets_df['hashtags'].dropna()).split(',')
        hashtag_counts = Counter(all_hashtags)
        print("Most Common Hashtags:", hashtag_counts.most_common(10))

    def sentiment_analysis(self):
        """Apply sentiment analysis and categorize tweets."""
        self.tweets_df['sentiment'] = self.tweets_df['cleaned_content'].apply(
            lambda tweet: TextBlob(tweet).sentiment.polarity)
        self.tweets_df['sentiment_category'] = self.tweets_df['sentiment'].apply(
            lambda score: 'Positive' if score > 0 else ('Neutral' if score == 0 else 'Negative'))
        print("Sentiment Analysis Completed")

    # Placeholder for future analysis methods
    def engagement_analysis(self):
        pass  # To be implemented

    def temporal_analysis(self):
        pass  # To be implemented

    def network_analysis(self):
        pass  # To be implemented

    def perform_all_analyses(self):
        """Run all analysis methods."""
        print("Starting Content Analysis...")
        self.keyword_extraction()
        self.hashtag_analysis()
        self.sentiment_analysis()
        # Uncomment the following lines as you implement the corresponding methods
        # self.engagement_analysis()
        # self.temporal_analysis()
        # self.network_analysis()
        print("All analyses completed.")


# Example usage
if __name__ == "__main__":
    processed_dataset_path = '/ProcessedFloodsInPakistan-tweets.xlsx'
    analysis = TweetAnalysis(processed_dataset_path)
    analysis.perform_all_analyses()
