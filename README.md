# Using Lyrics to Determine Musical Genre: An Analysis

## Hypothesis

I have a theory that musical genres can be identified solely upon their lyrics. There has been a good bit of research[^1] that was done on musical genre recognition using various other properties such as chords, tempo, rhythm, and key, but not much has been done in terms of using solely their lyrics. In this analysis, I'm going to be using a bag of words model for text analysis to be able to run several classification machine learning algorithms.

## Data Cleaning

When I first loaded up the data, there was about 362,000 rows of songs. After I used the info method to see what the data types were, I noticed that in the lyrics column there was about 100,000 rows that had null values. I used the dropna method to get rid of those, thus bringing the number of rows down to approximately 266,000 rows. I then checked to see what the unique genres were in the dataset using the value counts method on the genre column and saw that it brought back 12 different genres. However, upon closer inspection, one of them was labeled "Not Available" and the other one "Other". Because these would not be useful in this context, I got rid of all rows that had these values for genre, bringing down the total of rows to about 237,000. 

I then used the groupby method to see how many artists there were in the dataset and what the songs count looked like for the artists in the dataset. There were about 10,000 artists after the processing done thus far. Dolly Parton had the highest count with a count of 744 songs, and a few other artists had songs in the multi-hundreds. I didn't think this was completely accurate, so I did some looking and saw that there were multiple songs of the same or similar name. To fix this, I implemented the use of a library called "fuzzywuzzy". This library allows me to compare two strings, and then return a number between 0 and 100 depending on how match one string does or does not match with the other. I then wrote a for loop 

    lastsong = ""
    for index,row in artists.iterrows():
        currentsong = row.song
    #Setting the cutoff point to 85 to "pass"
    if fuzz.ratio(currentsong, lastsong) >= 85:
        lyrics = lyrics.drop(index[1], axis=0)
    lastsong = row.song

to go through each artist and compare each song to the one before it (I sorted the songs beforehand so those with similar names were more likely to be next to each other). This cleared out about another 6,000 rows bringing us down to approximately 231,000 rows.

I then began cleaning the lyrics in the dataset. Just so you're aware, the website that was used to gather the data (metrolyrics.com) is a user-submitted site, meaning that all songs on the website were generated and submitted by individual users. This means that there was a good bit of variation in how things were named and structured in each song. I used separated the lyrics column from the dataset and ran a series of regex lines to strip what I thought were the things that would be needed to be cleared of so that the lyrics were as "clean" as possible. 

Example of regex used:

    #Getting rid of newline characters
    lyricsep2 = [re.sub(r'\n',' ', i) for i in lyricsep2]

    #Getting rid of things such as "x4" or "x2" for repeating parts of a song
    lyricsep2 = [re.sub(r'x\d','', i) for i in lyricsep2]

    #Removing punctuation or anything that is not recognized in the Unicode format
    lyricsep2 = [re.sub(r'[^\w ]','', i) for i in lyricsep2]

    #Removing the word CHORUS as it pertains more often than not to the start of the chorus as opposed to being used in the song
    lyricsep2 = [re.sub(r'CHORUS','', i) for i in lyricsep2]


After I did this, I reappended the "clean lyrics" to the dataset.

Once I did that, I looked again at the value counts of the genres and noticed that the "Rock" genre had almost three times as many rows as the next genre. To deal with this so there would not be a sense of bias, I extracted the rows that had genre values of "Rock" and used the "random" library to randomly select 40,000 rows from that, and then I wrote a loop for the dataset that checked to see if the genre value for a given row was "Rock" and the index matched that of one of the ones in the list of indexes selected randomly, it would keep that one and any other one would get tossed out.

Due to a user-error (my bad) I hadn't realized that the logic on the first run wasn't correct, and so alongside deleting rows from the "Rock" genre, it also deleted rows from other genres as well. I was able to catch it before it did too much, but there were about another 10,000 rows deleted on top of the ones from the "Rock" genre. This brought the total amount of rows down to about 145,000. 

# Getting the Data Ready for the Machine Learning Models

Once I had all this done, next was to get the data ready for analysis. I added a column called "genre label" and wrote a piece of code that assigned each genre a different number from 0-9. At the same time I utilized an NLP tool called "nltk" and after I changed all the words to lowercase, use the "word_tokenize" method to tokenize all the lyrics in a given song and put it into a list. 

Code used for that:

    from nltk.tokenize import word_tokenize

    for index, row in lyricstwo.iterrows():
    #     print(row.genre)
        if row.genre == "Pop":
            lyricstwo.genre_label[index] = 0
        elif row.genre == "Rock":
            lyricstwo.genre_label[index] = 1
        elif row.genre == "Hip-Hop":
            lyricstwo.genre_label[index] = 2
        elif row.genre == "Metal":
            lyricstwo.genre_label[index] = 3
        elif row.genre == "Country":
            lyricstwo.genre_label[index] = 4
        elif row.genre == "Jazz":
            lyricstwo.genre_label[index] = 5
        elif row.genre == "Electronic":
            lyricstwo.genre_label[index] = 6
        elif row.genre == "R&B":
            lyricstwo.genre_label[index] = 7
        elif row.genre == "Indie":
            lyricstwo.genre_label[index] = 8
        else:
            lyricstwo.genre_label[index] = 9
    
    #     print(row.lyrics)
    lyricstwo.lyrics[index] = row.lyrics.lower()
    lyricstwo.lyrics[index] = word_tokenize(lyricstwo.lyrics[index])
<br>
After that was completed, I then used a "Bag-of-words" model and created a dictionary with all the words used and their respective count in the dataset. In the end, there was about 367,000 unique words in the dataset, or about 2.5-3 unique words per song. In order to have the Machine Learning models actually run and not take up a ton of memory space, I only used the top 300 words that were in the dictionary using a library called "heapq" and its nthlargest method to extract them. Due to time constraints, I wasn't able to do things like remove stopwords or do any sort of lemmatization or stemming, but I do hope to implement this in the future and compare results. 

Once I had this dictionary, I created my vector for each of the lyrics in the dataset and was ready to start running tests. I used the "train_test_split" library from sklearn to split my data with a test size of 20% and the training size to be 80% of the data.

# Running the Machine Learning Models

I decided that it might be worth the while to try and run a few different classification methods on the dataset to see which proved to be the best in terms of classifying the genres according to their lyrics. I decided to use **Logistic Regression**, **K-Nearest Neighbors**, **Naive-Bayes**, **Support Vector Machines**, and **Schotastic Gradient Descent**. 

Due to having such a large sample size, each model took about 2 hours or more to run. As of time of this writing, only the first 3 Machine Learning models have run, and SVM is still currently running. In the meantime, this is what I have so far

|Machine Learning Model|Score|
|----------------------|-----|
|Logistic Regression Training | 0.50|
|Logistic Regression Test | 0.49|
|K-Nearest Neighbors Training | 0.58|
|K-Nearest Neighbors Test | 0.40|
|Naive-Bayes Training | 0.26|
|Naive-Bayes Test | 0.25|


The other two models will be updated on this table whenever the tests decide to finish. 

As far as analysis goes for what I have so far, there is a lot of room for improvement, but you can see where there might be potential in determining which tests might do better. It seems like Logistic Regression despite having a score 0.50, on the testing did about the same. I'm not sure if it's due to any implicit bias I'm not aware of, or if it's just because it's that good of a general classification method. The same thing goes with the Naive-Bayes model showing a good bit of potential. K-Nearest Neighbors in this example did not do as well as the others, but the n value could always be adjusted and we could see how it could improve depending on that. I would maybe try and implement a GridSearch or something to try and find the optimized n value for that and see what kind of results it would return.

Looking back, there are a few things that I would do differently (and plan on doing later on in the future). The first thing I would do is I would definitely cut down the sample size. Even by cutting the data down to nearly a third of what it was, it was pretty ambitious to think that I could run these models in any sort of manner and not have them literally take hours to complete. I would probably cut down each genre to about 2,000 or so, just so that there is still a good amount to look through at 20,000 rows, but not an excessive amount like 140,000 rows. The other thing I definitely want to try and implement the next go around is using stopwords removal and lemmitization. I'm curious to see how it would improve the bag-of-words model output.

On top of all that, there are other methods that if I had the time I probaby would have looked more into trying to implement, such as using ngrams, tf-idf, or pos to see how that would help in improving the results of the classification methods as they tend to provide a lot more context in a model as opposed to bag-of-words.  



### Studies done on Musical Genre Recognition
[^1]: Fell, Michael and Sporleder, Caroline.
    Lyrics-based Analysis and Classification of 
    Music, 2017.[URL](http://www.anthology.aclweb.org/C/
    C14/C14-1059.pdf) 
    Howard,Sam, Silla Jr., Carlos N., and John-son, Colin G. 
    Automatic Lyrics-based Music GenreClassification in a 
    Multilingual Setting.2009.[URL]
    (https://pdfs.semanticscholar.org/e48c/
    a0b7a1796747cd3e22ef0586bded9faaf4ae.pdf)<br><br>
    Mayer, Rudolph, Neumayer, Robert, and Rauber,
    Andreas. Rhyme and Style Features For Musi-cal 
    Genre Classification by Song Lyrics.2008.[URL](https:/
    /books.google.com/books?hl=en&lr=&
    id=OHp3sRnZD-oC&oi=fnd&pg=PA337&dq=lyrics+2
    classification&ots=oFKPqKdCa6&
    sig=lzGHqWB3yJ5B-i_1FBR7Q0nYkBU#v=onepage&
    q=lyrics%20classification&f=false)<br><br>
    Viswanathan, Ajay Prasadh, and Sundaraj,Sriram. Music 
    Genre Classification.2015 [URL]
    (http://www.ijecs.in/issue/v4-i10/38%20ijecs.pdf)<br><br>
    Perez-Sancho, Carlos, Rizo, 
    David and Iñesta,José M. Genre classification 
    using chords andstochastic language models.2009.[URL]
    (http://web.a.ebscohost.com/ehost/pdfviewer/
    pdfviewer?vid=6&
    sid=da8e8d82-9182-4995-bd67-4a0c6cd9df81%40sessio
    nmgr4010)<br><br>
    Perez-Sancho, Carlos, Rizoa, David, 
    Iñesta,Jose M., Ponce de Leon, Pedro J., 
    Kersten, andRamirez, Rafael.  Genre 
    Classification of Mu-sic by Tonal Harmony.2010.[URL]
    (http://web.b.ebscohost.com/ehost/pdfviewer/
    pdfviewer?vid=3&
    sid=0380a4c2-4c17-4c1a-b34a-0b179ed19196%40sessio
    nmgr101)<br><br>
    Bacgi, Ulas, and Erzin, Engin. Inter 
    GenreSimilarity Modelling For Automatic Music 
    GenreClassification.2009.[URL]
    (https://arxiv.org/pdf/0907.3220.pdf)<br><br>
    Tsaptsinos, Alexandros. 
    Lyrics-based MusicGenre Classification Using a 
    Hierarchical Atten-tion Network.2017.[URL]
    (https://arxiv.org/pdf/1707.04678.pdf3)
