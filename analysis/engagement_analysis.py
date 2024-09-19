# engagement_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class EngagementAnalysis:
    def __init__(self, tweets_df):
        self.tweets_df = tweets_df

    def analyze_engagement(self):
        """
        Analyze engagement levels (likes, retweets, replies) in relation to crisis communication strategies.
        """
        strategies = ['denial', 'diminishment', 'rebuilding', 'bolstering']
        engagement_metrics = ['likeCount', 'retweetCount', 'replyCount']

        # Calculate average engagement metrics for each strategy
        engagement_summary = {}
        for strategy in strategies:
            filtered_df = self.tweets_df[self.tweets_df[strategy] == 1]
            print(f"{strategy} tweets count: {len(filtered_df)}")
            if len(filtered_df) > 0:
                averages = {metric: filtered_df[metric].mean() for metric in engagement_metrics}
            else:
                averages = {metric: 0 for metric in engagement_metrics}  # Or use NaN, depending on your analysis needs
            engagement_summary[strategy] = averages

        print("Engagement Analysis Summary:")
        for strategy, metrics in engagement_summary.items():
            print(f"{strategy}: {metrics}")

    # Implementation for visualizing engagement distributions
    def analyze_engagement_distribution(self):
        metrics = ['likeCount', 'retweetCount', 'replyCount']
        fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 6 * len(metrics)))

        for i, metric in enumerate(metrics):
            sns.histplot(self.tweets_df[metric], bins=50, ax=axs[i], kde=True, color='skyblue')
            axs[i].set_title(f'Distribution of {metric}')
            axs[i].set_xlabel(metric)
            axs[i].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    def identify_high_engagement_tweets(self):
        # Define a threshold for high engagement; this could be the top 10% for each metric
        thresholds = {metric: self.tweets_df[metric].quantile(0.9) for metric in
                      ['likeCount', 'retweetCount', 'replyCount']}

        high_engagement_tweets = self.tweets_df[
            (self.tweets_df['likeCount'] > thresholds['likeCount']) |
            (self.tweets_df['retweetCount'] > thresholds['retweetCount']) |
            (self.tweets_df['replyCount'] > thresholds['replyCount'])
            ]

        print(f"Number of high engagement tweets: {len(high_engagement_tweets)}")
        # Example analysis: Most common words in high engagement tweets
        high_engagement_words = ' '.join(high_engagement_tweets['cleaned_content']).split()
        word_counts = pd.Series(high_engagement_words).value_counts().head(10)
        print("Most common words in high engagement tweets:", word_counts)

    def analyze_author_profiles(self):
        # Group tweets by 'verified' status and calculate average engagement
        avg_engagement_by_verified = self.tweets_df.groupby('verified')[
            ['likeCount', 'retweetCount', 'replyCount']].mean()

        print("Average engagement by verified status:")
        print(avg_engagement_by_verified)

        # Further analysis could include examining the influence of followers count, media count, etc.
        # Example: Average engagement by followers count quartile
        followers_quartiles = pd.qcut(self.tweets_df['followersCount'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        avg_engagement_by_followers = self.tweets_df.groupby(followers_quartiles)[
            ['likeCount', 'retweetCount', 'replyCount']].mean()

        print("\nAverage engagement by followers count quartile:")
        print(avg_engagement_by_followers)

    def perform_detailed_engagement_analysis(self):
        self.analyze_engagement_distribution()
        self.identify_high_engagement_tweets()
        self.analyze_author_profiles()
        # Invoke additional analysis methods as needed