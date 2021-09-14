# GDSC-UPD Engineering Data Science Programming Interview Test

## Getting started

Before starting the assignment you must do the following:
- Create a GitHub account at https://www.github.com if you don't already have one.

- Visit https://github.com/tusvi/gdsc-upd_engg_ds_interview and clone the repo.

- Once you have completed all the tasks, please create a new public personal repository and push your submission files there. Dm/email me the repository where you submitted the files for checking.


### Repository Files
The starting repository contains the following files:
- `main.py`: This file contains code to verify your answers. You must not edit anything in main.py

- `assignment1.py`: This file contains one function for each task in the assignment. You should fill in the relevant function to complete the task.
You may choose to create additional functions to segment your code, but all the code you write must be contained in this file.

- `data/data.json`: This file contains details of recent soccer matches in the English Premier League,
which you will need in order to complete your assignment.

- `data/football`: This folder contains a number of news articles about soccer matches in the English Premier League.
You will need to load these files in order to complete your assignment.



## Your Tasks

### Task 1: Loading and interpreting a JSON file

Write a function `task1()` that loads the `data/data.json` file into Python. Your function
should return a list of teams codes, sorted in alphabetical order by team code.

You can test your implementation with the following command: `python main.py task1`



### Task 2: Data Aggregation

Write a function `task2()` that uses the information contained in the clubs objects to work out
how many goals were scored by and against each team in total throughout the season. Your
function should output this information to a csv file called `task2.csv`.
Your csv file should contain the following headings:
```
team code, goals scored by team, goals scored against team
```

Each row in the file should contain the details for one team, sorted in alphabetical order by team code.

You can test your implementation with the following command: `python main.py task2`



### Task 3: Regular Expressions

In addition to the information contained in the `data.json` file, we also have a number of
news articles written about soccer matches. Each article is located in a separate text file in
the `data/football` folder. For this task we will assume that each article is written about a
match. Write a function `task3()` to extract the largest match score identified in the article.

Add the number of goals scored by each side together to produce the total number of goals
scored in the match.

For example, if the largest match score mentioned in an article is 14-6, your program
should calculate 20 as the total number of goals. For this task we define the largest match
score as the one with the highest total number of goals, so a score of 14-6 is considered larger
than a score of 16-2.

If a suitable score cannot be found in the article, your function should return 0 as the
total number of goals for that article. You will need to use regular expressions to accomplish
this.

Your function should produce a csv file containing the filename and the total number of
goals for each article. Your csv file should contain two columns, filename and total goals.
Each row in the file should contain the detail for one article, sorted in ascending alphabetic
order by filename. Save this file as `task3.csv`

You can test your implementation with the following command: `python main.py task3`



### Task 4: Visualising Scores

We now wish to understand whether there are outliers present in the number of goals we calculated in Task 3.

Write a function `task4()` that produces a boxplot showing the distribution of values for total goals.
Any values more than 1.5 interquartile ranges above Q3 should be identified as outliers on the plot.

This boxplot should be saved as `task4.png`.

For all tasks involving visualisations, you should ensure that your plots contain a title and labels for all relevant axes.

You can test your implementation with the following command: `python main.py task4`



### Task 5: Extracting information from text data

We now wish to understand how often each club is mentioned by the media.
The `data.json` file also contained a list of club names.

Write a function `task5()` that searches through each of the news articles for mentions of each club
and counts the articles for which each club is mentioned at least once.

Your function should produce a csv file containing the club name and number of mentions for each club.

Your csv file should contain the following column headings:
```
club name, and number of mentions.
```
Save this file as `task5.csv`.

Each row in the file should contain the details for one team, sorted in ascending alphabetic order by club name.

Your function should also produce a bar chart conveying this information, saved as `task5.png`.

You can test your implementation with the following command: `python main.py task5`



### Task 6: Extracting information from text data

We also wish to understand which clubs are commonly mentioned together in the same news
articles, as we believe that clubs that are commonly mentioned together in the same news
articles are similar.

We can produce a similarity score for clubs using the following formula:

![image](https://user-images.githubusercontent.com/60842200/132987205-70aaf785-91f1-4865-82e4-a84166cafd04.png)

Write a function `task6()` that calculates the similarity score for each
pair of clubs and produce a heatmap conveying this information.

This heatmap should be saved as `task6.png`

You can test your implementation with the following command: `python main.py task6`



### Task 7: Comparing Information

We now wish to understand whether the number of times a team is mentioned in the media is related to its performance.
Write a function `task7()` that produces a scatterplot comparing
the number of articles mentioning each team (as calculated in Task 5)
with the total number of goals scored by each team (as calculated in Task 2).

This scatterplot should be saved as `task7.png`

You can test your implementation with the following command: `python main.py task7`



### Task 8: Text pre-processing

We now wish to perform the following preprocessing on each article in the in order to make them easier to analyse.
Write a function `task8(filepath)` that performs the following pre-processing.
This function takes a single argument, `filepath`, which specifies the file to be processed.

- Remove all non-alphabetic characters (for example, numbers, apostrophes and punctuation characters),
except for spacing characters such as whitespaces, tabs and newlines.
- Convert all spacing characters such as tabs and newlines to whitespace and ensure that only one
whitespace character exists between each word
- Change all uppercase characters to lower case
- Tokenize the resulting string into words
- Remove all stopwords in nltk's (Natural Language Toolkit Python lib) list of English stopwords from the resulting list
- Remove all remaining words that are only a single character long from the resulting list

The function should return a list of all the words in the article after parsing and cleaning.

You can test your implementation with the following command: `python main.py task8`



GOOD LUCK!! :)
