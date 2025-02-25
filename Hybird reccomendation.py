import pandas as pd

def hybrid_recommendations(input_song_name, num_recommendations=5, alpha=0.5):
    if input_song_name not in music_df['Track Name'].values:
        print(f"'{input_song_name}' not found in the dataset. Please enter a valid song name.")
        return

    content_based_rec = content_based_recommendations(input_song_name, num_recommendations)

    popularity_score = music_df.loc[music_df['Track Name'] == input_song_name, 'Popularity'].values[0]

    weighted_popularity_score = popularity_score * calculate_weighted_popularity(
        music_df.loc[music_df['Track Name'] == input_song_name, 'Release Date'].values[0]
    )

    new_entry = pd.DataFrame({
        'Track Name': [input_song_name],
        'Artists': [music_df.loc[music_df['Track Name'] == input_song_name, 'Artists'].values[0]],
        'Album Name': [music_df.loc[music_df['Track Name'] == input_song_name, 'Album Name'].values[0]],
        'Release Date': [music_df.loc[music_df['Track Name'] == input_song_name, 'Release Date'].values[0]],
        'Popularity': [weighted_popularity_score]
    })

    hybrid_recommendations = pd.concat([content_based_rec, new_entry], ignore_index=True)

    hybrid_recommendations = hybrid_recommendations.sort_values(by='Popularity', ascending=False)

    hybrid_recommendations = hybrid_recommendations[hybrid_recommendations['Track Name'] != input_song_name]

    return hybrid_recommendations

input_song_name = "I'm Good (Blue)"
recommendations = hybrid_recommendations(input_song_name, num_recommendations=5)
print(f"Hybrid recommended songs for '{input_song_name}':")
print(recommendations)