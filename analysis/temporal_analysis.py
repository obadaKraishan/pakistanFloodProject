import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class TemporalAnalysis:
    def __init__(self, tweets_df):
        self.tweets_df = tweets_df

    def load_processed_data(self):
        # Assuming the dates are in ISO 8601 format, for example, '2022-09-13 20:22:57+00:00'
        # Adjust the format as per your dataset's date format
        try:
            df = pd.read_csv(self.dataset_path, parse_dates=['date'],
                             date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S%z', errors='coerce'))
            print("Date Range:", df['date'].min(), df['date'].max())
            print("Date types:", df['date'].dtypes)
            return df
        except Exception as e:
            print(f"Failed to load or parse dataset: {e}")
            return None

    def tweet_volume_over_time(self):
        if 'date' not in self.tweets_df.columns or self.tweets_df['date'].isna().all():
            print("No valid 'date' data available for analysis. Exiting method.")
            return

        # Convert 'date' column to datetime, including parsing the timezone
        self.tweets_df['date'] = pd.to_datetime(self.tweets_df['date'], errors='coerce', format='%Y-%m-%d %H:%M:%S+00:00')

        # Check if after converting, dates are appropriate
        print("Date Range:", self.tweets_df['date'].min(), self.tweets_df['date'].max())
        print("Date types:", self.tweets_df['date'].dtypes)

        # Filter out rows with NaT in 'date' column
        valid_tweets_df = self.tweets_df.dropna(subset=['date'])
        print("Number of valid tweets after filtering:", len(valid_tweets_df))

        if valid_tweets_df.empty:
            print("No valid tweets available for analysis. Exiting method.")
            return

        # Ensure DataFrame is sorted by date
        valid_tweets_df = valid_tweets_df.sort_values(by='date')

        # Set the 'date' column as the DataFrame index
        valid_tweets_df.set_index('date', inplace=True)

        # Resample to daily frequency and count tweets per day
        daily_tweet_volume = valid_tweets_df.resample('D').size()

        if daily_tweet_volume.empty:
            print("No data available for daily tweet volume. Exiting method.")
            return

        plt.figure(figsize=(12, 6))
        daily_tweet_volume.plot(title='Daily Tweet Volume Over Time', color='skyblue')
        plt.xlabel('Date')
        plt.ylabel('Number of Tweets')
        plt.grid(True)
        plt.show()

    def sentiment_over_time(self, frequency='W'):
        # Convert 'date' to datetime and localize to UTC
        self.tweets_df['date'] = pd.to_datetime(self.tweets_df['date'], errors='coerce', utc=True)

        # Drop rows with NaT in 'date' column
        valid_tweets_df = self.tweets_df.dropna(subset=['date'])

        if valid_tweets_df.empty:
            print("No valid tweets available for sentiment analysis. Exiting method.")
            return

        # Resample and compute mean sentiment
        sentiment_over_time = valid_tweets_df.resample(frequency, on='date')['sentiment'].mean()

        plt.figure(figsize=(12, 6))
        sentiment_over_time.plot(title='Sentiment Over Time', color='salmon')
        plt.xlabel('Date')
        plt.ylabel('Average Sentiment')
        plt.show()

    def high_engagement_over_time(self, metric='likeCount', threshold_quantile=0.9):
        """Identifies periods of high engagement."""
        threshold = self.tweets_df[metric].quantile(threshold_quantile)
        high_engagement_df = self.tweets_df[self.tweets_df[metric] >= threshold]

        high_engagement_daily = high_engagement_df.resample('D', on='date').size()

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=high_engagement_daily, label='High Engagement Volume', color='green')
        plt.title(f'High {metric.capitalize()} Tweets Over Time')
        plt.xlabel('Date')
        plt.ylabel('High Engagement Tweet Volume')
        plt.legend()
        plt.show()

    def perform_analysis(self):
        if self.tweets_df is None or self.tweets_df.empty:
            print("No valid dataset loaded. Exiting.")
            return

        """Wrapper method to perform all analyses."""
        print("Starting Temporal Analysis...")
        self.tweet_volume_over_time()
        self.sentiment_over_time()
        self.high_engagement_over_time()
        print("Temporal Analysis Completed.")
        