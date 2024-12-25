from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt

# Sample user feedback dataset
data = {
    "Feedback": [
        "The website is user-friendly and easy to navigate.",
        "The loading time is very slow and frustrating.",
        "I love the design and color scheme.",
        "The website is cluttered and hard to use.",
        "The content is very informative and well-organized.",
        "Too many ads make it annoying to browse.",
        "Great website! Very intuitive and helpful.",
        "Not mobile-friendly at all."
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Function to analyze sentiment
def analyze_sentiment(feedback):
    blob = TextBlob(feedback)
    return blob.sentiment.polarity  # Polarity ranges from -1 (negative) to 1 (positive)

# Apply sentiment analysis
df['Sentiment Score'] = df['Feedback'].apply(analyze_sentiment)

# Categorize sentiment
def categorize_sentiment(score):
    if score > 0.1:
        return 'Positive'
    elif score < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

df['Sentiment Category'] = df['Sentiment Score'].apply(categorize_sentiment)

# Print the DataFrame
print(df)

# Summary statistics
summary = df['Sentiment Category'].value_counts()
print("\nSummary:")
print(summary)

# Visualization
summary.plot(kind='bar', color=['green', 'red', 'blue'], title='Website Sentiment Analysis')
plt.xlabel('Sentiment Category')
plt.ylabel('Number of Feedbacks')
plt.show()
