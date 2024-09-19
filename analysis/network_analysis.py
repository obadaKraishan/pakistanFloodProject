import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from ast import literal_eval


class NetworkAnalysis:
    def __init__(self, tweets_df):
        self.tweets_df = tweets_df

    def parse_mentioned_users(self, mentioned_users_str):
        """Parse the custom format of mentioned users to a list of dicts."""
        users = []
        if pd.isna(mentioned_users_str) or mentioned_users_str.strip() == "":
            return users

        # Splitting the string by semicolon and comma to extract users
        user_entries = mentioned_users_str.split(';')
        for entry in user_entries:
            user_info = entry.split(',')
            user_dict = {}
            for info in user_info:
                key_value = info.split(':')
                if len(key_value) == 2:
                    key, value = key_value
                    user_dict[key.strip()] = value.strip()
            if user_dict:
                users.append(user_dict)
        return users

    def build_interaction_network(self):
        """Builds a network graph based on mentions and replies."""
        G = nx.DiGraph()

        for index, row in self.tweets_df.iterrows():
            source = row['username']
            mentioned_users_str = row.get('mentionedUsers', '')

            mentioned_users = self.parse_mentioned_users(mentioned_users_str)
            for user in mentioned_users:
                target = user.get('username')
                if source and target:
                    G.add_edge(source, target)

        return G

    def visualize_network(self, G):
        """Visualizes the interaction network with text labels and saves it as a PNG file."""
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, k=0.15, iterations=20)
        nx.draw(G, pos, with_labels=False, node_size=20, alpha=0.6, arrows=True)

        # Add text labels to the nodes
        nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')

        plt.title("Twitter Interaction Network")

        # Specify the filename and path where you want to save the plot
        plt.savefig('/Volumes/Kraishan 1/TTU//Thesis/NEW/Pakistan/twitter_interaction_network.png', format='png', dpi=300)

        # Display the plot
        plt.show()

    def perform_all_analyses(self):
        print("Building Twitter Interaction Network...")
        G = self.build_interaction_network()
        print(f"Network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        self.visualize_network(G)


# Example usage
if __name__ == "__main__":
    # Load your dataset here
    dataset_path = '/Volumes/Kraishan 1/TTU//Thesis/NEW/Pakistan/ProcessedFloodsInPakistan-tweets.xlsx'
    tweets_df = pd.read_csv(dataset_path)
    analysis = NetworkAnalysis(tweets_df)
    analysis.perform_all_analyses()