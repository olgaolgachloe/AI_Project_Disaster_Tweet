import re
import string
from collections import Counter

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import nlpaug.augmenter.word as naw


# Helper functions to clean texts. Remove urls, emojis, html tags and punctuations.

def remove_url(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


def remove_emoji(text):
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_html(text):
    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(html, '', text)


def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)


def clean_v2_data(text):
    # lower case
    text = text.lower()

    # remove url
    text = re.sub('https?://.+', '', text)

    # punctuation
    text = re.sub(r'\[.*?.\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)

    # remove ending […]
    text = re.sub('[…“”’]', '', text)

    # remove leading or ending space
    text = text.strip()
    return text


def extract_valid_target(targets):
    if len(targets) == 1:
        return targets[0]
    counter = Counter(targets)
    if counter[0] == counter[1]:
        return None
    if counter[1] > counter[0]:
        return 1
    return 0


def get_v1_data():
    df_v1 = pd.read_csv('../data/2015_train.csv')
    df_v1['text_clean'] = df_v1['text'].apply(lambda x: remove_url(x))
    df_v1['text_clean'] = df_v1['text_clean'].apply(lambda x: remove_emoji(x))
    df_v1['text_clean'] = df_v1['text_clean'].apply(lambda x: remove_html(x))
    df_v1['text_clean'] = df_v1['text_clean'].apply(lambda x: remove_punct(x))

    df_v1 = df_v1[['text_clean', 'target']].rename(columns={'text_clean': 'text', 'target': 'label'})
    # apply v2 function to remove cases with \n
    df_v1['text'] = df_v1['text'].apply(lambda x: clean_v2_data(x))
    df_v1 = df_v1[df_v1['text'] != '']
    df_v1['label'] = df_v1['label'].astype(int)
    return df_v1


def get_v2_data():
    df = pd.read_csv("../data/2020_train.csv")
    df['text'] = df['text'].apply(lambda x: clean_v2_data(x))
    df = df[df['text'] != '']
    df_clean = df.groupby('text')['target'].apply(list).reset_index(name='targets')
    df_clean['target'] = df_clean['targets'].apply(lambda x: extract_valid_target(x))
    df_clean = df_clean.dropna(subset=['target'])[['text', 'target']].rename(columns={'target': 'label'})
    df_clean['label'] = df_clean['label'].astype(int)
    return df_clean


# negative vs positive: 2.5:1
def data_augment(df: DataFrame):
    df_neg = df[df['label'] == 0]
    df_pos = df[df['label'] == 1]
    aug = naw.SynonymAug(aug_src='wordnet')
    df_aug_pos = pd.DataFrame({"text": pd.Series(aug.augment(df_pos['text'].tolist()))})
    df_aug_pos['label'] = 1
    df = pd.concat([df_pos, df_aug_pos, df_neg])
    return df


def main():
    df_v1 = get_v1_data()
    df_v2 = get_v2_data()
    df_join = pd.concat([df_v1, df_v2])
    x_train, x_test, y_train, y_test = train_test_split(df_join['text'],
                                                        df_join['label'],
                                                        random_state=200,
                                                        test_size=0.3)
    df_join_train = pd.concat([x_train, y_train], axis=1)
    # df_join_train = data_augment(df_join_train)
    df_join_train.to_csv("../data/joined_train.csv", index=False)
    df_join_test = pd.concat([x_test, y_test], axis=1)
    df_join_test.to_csv("../data/joined_test.csv", index=False)
    print(df_join_train.groupby(['label']).count())


if __name__ == '__main__':
    main()
