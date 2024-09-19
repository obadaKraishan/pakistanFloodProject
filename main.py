# main.py
from analysis.content_analysis import ContentAnalysis
from analysis.engagement_analysis import EngagementAnalysis
from analysis.temporal_analysis import TemporalAnalysis
from analysis.network_analysis import NetworkAnalysis
import pandas as pd

if __name__ == "__main__":
    processed_dataset_path = 'DS_PATH'

    # Assuming the ContentAnalysis class handles loading the dataset, you might not need to load it again here.

    content_analysis = ContentAnalysis(processed_dataset_path)
    content_analysis.perform_all_analyses()
    content_analysis.save_analysis_to_doc()

    # If you have the DataFrame accessible after content analysis (e.g., through a property or method),
    # you can pass it to other analysis classes. Otherwise, you might need to reload the dataset.
    tweets_df = content_analysis.tweets_df  # Example; adjust based on your actual implementation.

    # Perform temporal analysis
    temporal_analysis = TemporalAnalysis(content_analysis.tweets_df)
    temporal_analysis.tweet_volume_over_time()
    temporal_analysis.sentiment_over_time()
    temporal_analysis.high_engagement_over_time()

    # Perform engagement analysis
    engagement_analysis = EngagementAnalysis(tweets_df)
    engagement_analysis.analyze_engagement()

    # Perform network analysis
    network_analysis = NetworkAnalysis(tweets_df)
    network_analysis.perform_all_analyses()

