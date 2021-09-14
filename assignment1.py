# This is the file you will need to edit in order to complete assignment 1
# You may create additional functions, but all code must be contained within this file


# Some starting imports are provided, these will be accessible by all functions.
# You may need to import additional items
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import json
import re
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# You should use these two variable to refer the location of the JSON data file and the folder containing the news articles.
# Under no circumstances should you hardcode a path to the folder on your computer (e.g. C:\Chris\Assignment\data\data.json) as this path will not exist on any machine but yours.
datafilepath = 'data/data.json'
articlespath = 'data/football'

def task1():
    with open(datafilepath, encoding='utf-8') as f:
        j = json.loads(f.read())
        df = pd.json_normalize(j['clubs']).sort_values(by=['club_code'])
        return list(df['club_code'])

def task2():
    with open(datafilepath, encoding='utf-8') as f:
        j = json.loads(f.read())
        df = pd.json_normalize(j['clubs']).sort_values(by=['club_code'])
        df[['club_code', 'goals_scored', 'goals_conceded']].to_csv('task2.csv', encoding='utf-8',
            index=False, header=['team code', 'goals scored by team', 'goals scored against team'])
    return df[['club_code', 'goals_scored', 'goals_conceded']]

def task3():
    reg = re.compile('\s(\d{1,2})-(\d{1,2})')
    out = []
    for file in os.listdir(articlespath):
        with open(articlespath + '/' + file, encoding='utf-8') as f:
            '''
            long code alert:
             1) adds the regex obtained goal count,
             2) takes the max of the sums - if empty, take 0
             3) appends [file, max] to out (2-dim array)
            '''
            out.append([file, (max([0]+list(map(lambda x: int(x[0])+int(x[1]),reg.findall(f.read())))))])
    df = pd.DataFrame(out, columns=['file', 'largest match score'])
    df.to_csv('task3.csv', encoding='utf-8', index=False)
    return df

def task4():
    data = pd.read_csv('task3.csv', encoding='utf-8')['largest match score']
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, 1, 'D', labels=[''], vert=False)
    # plt.xscale('log') # outlier view
    ax.yaxis.set_ticks_position('none')
    ax.set(
        axisbelow=True,
        title='Distribution of Maximum Total Goals per Article',
        xlabel='Total Goals',
        ylabel='',
    )
    plt.savefig('task4.png')
    return

def task5():
    with open(datafilepath, encoding='utf-8') as f:
        j = json.loads(f.read())
        df = pd.json_normalize(j['clubs'])
    reg = re.compile('|'.join([x for x in df['name']]))

    d = {x:0 for x in df['name']} # club to article count mapping
    for file in os.listdir(articlespath):
        with open(articlespath + '/' + file, encoding='utf-8') as f:
            for club in set(reg.findall(f.read())):
                d[club]+=1
    df['mentions'] = df['name'].map(d)
    df[['name', 'mentions']].to_csv('task5.csv', encoding='utf-8',
                        index=False)
    fig, ax = plt.subplots(figsize=(20, 15))
    ax.barh(y='name', width='mentions', data=df)
    ax.set(
        axisbelow=True,
        title='Number of Articles Referencing Each Football Club',
        xlabel='No. of Articles',
        ylabel='Football Clubs',
    )
    plt.gca().invert_yaxis()
    plt.savefig('task5.png')
    return df

def sim(both, one, two):
    return (2 * both)/(one+two)

def makeMatrix():
    with open(datafilepath, encoding='utf-8') as f:
        j = json.loads(f.read())
        df = pd.json_normalize(j['clubs'])
    reg = re.compile('|'.join([x for x in df['name']]))
    dirr = os.listdir(articlespath)
    equiv = {x : i for i, x in enumerate(df['name'].tolist())}
    count = np.zeros((len(df['name']), len(dirr))) # bool matrix if club in file

    for i, file in enumerate(dirr):
        with open(articlespath + '/' + file, encoding='utf-8') as f:
            for x in reg.findall(f.read()):
                count[equiv[x]][i] = 1

    matrix = np.zeros((len(df['name']), len(df['name'])))
    for i in range(len(df['name'])):
        for j in range(i+1,len(df['name'])):
            matrix[i][j] = sim(sum(count[i]*count[j]), sum(count[i]), sum(count[j]))
            matrix[j][i] = matrix[i][j] # heatmap symmetry
    return matrix

def task6():
    matrix = makeMatrix()
    df = pd.read_csv('task5.csv', encoding='utf-8')
    fig, ax = plt.subplots(figsize=(20, 15))
    im = ax.imshow(matrix, cmap='Greys') # wanted to make it look like a soccer ball
    ax.set_xticks(np.arange(len(df['name'])))
    ax.set_yticks(np.arange(len(df['name'])))
    ax.set_xticklabels(df['name'])
    ax.set_yticklabels(df['name'])

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    ax.set(
        axisbelow=True,
        title='Which football clubs are commonly mentioned together?',
    )

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Similarity Score', rotation=-90, va="bottom")

    plt.savefig('task6.png')

    return matrix

def task7():
    fig, ax = plt.subplots(figsize=(20, 15))
    df1 = pd.read_csv('task5.csv', encoding='utf-8')
    df2 = pd.read_csv('task2.csv', encoding='utf-8')
    ax.scatter(df2['goals scored by team'], df1['mentions'])
    ax.set(
        axisbelow=True,
        title='Correlation between Football Clubs\' Article Mentions and Goals Scored',
        xlabel='Goals Scored',
        ylabel='No. of Articles',
    )
    plt.savefig('task7.png')
    return


def task8(filename):
    # by bullet point
    reg1 = re.compile('[^A-Za-z\s]')
    reg2 = re.compile('\s\s+')
    with open(filename, encoding='utf-8') as f:
        one = reg1.sub(' ', f.read())
    two = reg2.sub(' ', one)
    three = two.lower()
    four = word_tokenize(three)
    stop_words = set(stopwords.words('english'))
    five = [w for w in four if not w in stop_words]
    six = [w for w in five if len(w)>1]
    return six
