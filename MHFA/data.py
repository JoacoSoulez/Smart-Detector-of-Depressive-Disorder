import pandas as pd
import os

raw_data_path = '/home/leoasad/code/JoacoSoulez/MHFA/raw_data/'

def get_tweets(raw_data_path=raw_data_path):
    """"Import a dataset of Tweets labeled as not depressive (0) and depressive (1)"""
    # Import data from raw_data folder
    data = pd.read_csv(f"{raw_data_path}tweets/training.processed.noemoticon.csv",
                      encoding_errors='ignore',
                      usecols=[0,5],
                      header=None,
                      names=['label','tweets'])

    # Remove missing values
    data = data.dropna()

    # Filter only possitive Tweets
    positive_tweets = data[data['label']==4].drop(columns='label').reset_index(drop=True)

    # Assign 0 to not depressive tweets
    positive_tweets['label'] = 0

    # Import depressive tweets
    depressive_tweets = pd.read_csv(f"{raw_data_path}tweets/depressive_tweets_processed.csv",
                                   sep = '|',
                                   header = None,
                                   usecols = [5],
                                   names=['tweets'])
    # Remove missing values
    depressive_tweets = depressive_tweets.dropna()
    # Assign 1 to depressive tweets
    depressive_tweets['label'] = 1

    # Concat possitive + depressive tweets
    # Get the length of depressive tweets
    n_depressive = len(depressive_tweets)
    # Undersample possitive tweets
    positive_tweets_reduced = positive_tweets.sample(n=n_depressive)
    # Concat both datasets + shuffle their rows
    tweets = pd.concat([positive_tweets_reduced, depressive_tweets]).sample(frac=1).reset_index(drop=True)

    return tweets

def get_tweets_and_reddit(raw_data_path=raw_data_path):
    """"Import a dataset of Tweets and Reddit Posts labeled as not depressive (0) and depressive (1)"""
    # Import depressive Reddit posts
    reddit = pd.read_csv(f'{raw_data_path}reddit/reddit_depression_suicidewatch.csv')

    # Remove missing values
    reddit = reddit.dropna()

    # Filter post from subreddit depression
    depressive_reddit = reddit[reddit['label']=='depression'].copy()
    # Assign to depressive post the label 1
    depressive_reddit.loc[:,'label'] = depressive_reddit['label'].map({'depression':1})

    # Import depressive tweets
    depressive_tweets = pd.read_csv(f"{raw_data_path}tweets/depressive_tweets_processed.csv",
                                   sep = '|',
                                   header = None,
                                   usecols = [5],
                                   names=['text'])
    # Remove missing values
    depressive_tweets = depressive_tweets.dropna()
    # Assign 1 to depressive tweets
    depressive_tweets['label'] = 1

    # Concat depressive tweets + depressive reddit posts
    depressive_text = pd.concat([depressive_reddit, depressive_tweets,]).reset_index(drop=True)

    # Get the length of depressive tweets
    n_depressive_text = len(depressive_text)

    # Get possitive tweets
    # Import data from raw_data folder
    data = pd.read_csv(f"{raw_data_path}tweets/training.processed.noemoticon.csv",
                      encoding_errors='ignore',
                      usecols=[0,5],
                      header=None,
                      names=['label','text'])

    # Remove missing values
    data = data.dropna()

    # Filter only possitive Tweets
    positive_tweets = data[data['label']==4].drop(columns='label').reset_index(drop=True)

    # Assign 0 to not depressive tweets
    positive_tweets['label'] = 0

    # Undersample possitive tweets to get a balanced dataset
    positive_text_reduced = positive_tweets.sample(n=n_depressive_text)

    # Concat both depressing and not_depressing text and shuffle
    twiter_reddit = pd.concat([depressive_text, positive_text_reduced]).sample(frac=1).reset_index(drop=True)

    return twiter_reddit

if __name__ == "__main__":
    #print(os.path.abspath(__file__))
    df = get_tweets_and_reddit()
    print(df)
