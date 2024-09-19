import pandas as pd
import re
import html


def clean_text(text):
    """
    Clean the text to remove any encoding-related issues, including HTML entities and emojis.
    """
    if pd.isna(text):
        return text
    # Decode HTML entities
    text = html.unescape(text)
    # Remove emojis: Matches most emojis by using unicode blocks
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text

try:
    # Attempt to read with specified encoding
    df = pd.read_csv('/Volumes/Kraishan 1/TTU//Thesis/NEW/Pakistan/FloodsInPakistan-tweets.csv', encoding='mac_roman')
except Exception as e:
    print(f"Error reading file: {e}")

# Apply cleaning to the content field and any other fields as necessary
df['content'] = df['content'].apply(clean_text)

# Save the cleaned data
df.to_csv('/Volumes/Kraishan 1/TTU//Thesis/NEW/Pakistan/CleanedFloodsInPakistan-tweets.csv', index=False, encoding='utf-8-sig')
