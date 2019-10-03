#Things to do:

# 1) Go through the data.
# 2) Make "for" loop with a second table where the values with NA are separated (not necessary)
# 3) Check through the data and verify info.
# 4) Experiment with "stemming" the words in different table
# 5) Set up n-gram model
# 6) Run n-gram model
# 7) Split data into training and testing data
# 8) Set up SVM loop for training
# 9) Run SVM loop
# 10) Check through training data results

library(RWeka)
library(ngram)
library(stringi)
library(e1071)
library(rlist)

lyrics <- read.csv("/Users/SilverSurfer/Repositories/lyric-analysis/lyrics.csv")

lyricssep <- lyrics

lyricssep <- lyricssep$lyrics
lyricssep <- as.data.frame(lyricssep)
na.omit(lyrics)

lyricstoo <- lyrics

subs <- list('[[:punct:]]','Verse','verse','VERSE','Chorus','CHORUS','chorus','Instrumental','instrumental','INSTRUMENTAL','2x','x2','4x','x4','Bridge','Hook','HOOK','Pre','intro','Intro','INTRO','outro','Outro','OUTRO','[0-9]')

#Substitute punctuations and key words with spaces

#lyricssep <- gsub('[[:punct:]]+','',lyricstoo$lyrics)
#lyricssep <- as.data.frame(lyricssep)

# for(i in lyricssep){
#   if(is.null(lyricssep[i,1])){
#     lyricssep <- lyricssep[-i,]
#   }
# }


while(is.null(lyricssep[1,1])){
  lyricssep <- lyricssep[-1]
}
lyricssep <- as.data.frame(lyricssep)

lyricssep <- gsub('[\n]+',' ',lyricssep$lyrics)
lyricssep <- as.data.frame(lyricssep)

for(i in subs){
  lyricssep <- gsub(i,'',lyricssep$lyrics)
  lyricssep <- as.data.frame(lyricssep)
}

# lyricssep <- gsub('Verse','',lyricssep$lyrics) #also verse and VERSE separately
# lyricssep <- as.data.frame(lyricssep)
# 
# lyricssep <- gsub('verse','',lyricssep$lyrics) #also verse and VERSE separately
# lyricssep <- as.data.frame(lyricssep)
# 
# lyricssep <- gsub('VERSE','',lyricssep$lyrics) #also verse and VERSE separately
# lyricssep <- as.data.frame(lyricssep)
# 
# lyricssep <- gsub('Chorus','',lyricssep$lyrics)
# lyricssep <- as.data.frame(lyricssep)
# 
# lyricssep <- gsub('CHORUS','',lyricssep$lyrics) 
# lyricssep <- as.data.frame(lyricssep)
# 
# lyricssep <- gsub('chorus','',lyricssep$lyrics) 
# lyricssep <- as.data.frame(lyricssep)
# 
# lyricssep <- gsub('Instrumental','',lyricssep$lyrics)
# lyricssep <- as.data.frame(lyricssep)
# 
# lyricssep <- gsub('instrumental','',lyricssep$lyrics)
# lyricssep <- as.data.frame(lyricssep)
# 
# lyricssep <- gsub('INSTRUMENTAL','',lyricssep$lyrics)
# lyricssep <- as.data.frame(lyricssep)
# 
# lyricssep <- gsub('2x','',lyricssep$lyrics) #x2 also
# lyricssep <- as.data.frame(lyricssep)
# 
# lyricssep <- gsub('x2','',lyricssep$lyrics) #x2 also
# lyricssep <- as.data.frame(lyricssep)
# 
# lyricssep <- gsub('4x','',lyricssep$lyrics) #x4 also
# lyricssep <- as.data.frame(lyricssep)
# 
# lyricssep <- gsub('x4','',lyricssep$lyrics) #x2 also
# lyricssep <- as.data.frame(lyricssep)
# 
# lyricssep <- gsub('Bridge','',lyricssep$lyrics)
# lyricssep <- as.data.frame(lyricssep)
# 
# lyricssep <- gsub('Hook','',lyricssep$lyrics) #PreHook also
# lyricssep <- as.data.frame(lyricssep)
# 
# lyricssep <- gsub('HOOK','',lyricssep$lyrics) 
# lyricssep <- as.data.frame(lyricssep)
# 
# lyricssep <- gsub('Pre','',lyricssep$lyrics) #PreHook also
# lyricssep <- as.data.frame(lyricssep)
# 
# lyricssep <- gsub('Intro',' ',lyricssep$lyrics) #Outro also
# lyricssep <- as.data.frame(lyricssep)
# 
# lyricssep <- gsub('intro','',lyricssep$lyrics)
# lyricssep <- as.data.frame(lyricssep)
# 
# lyricssep <- gsub('INTRO','',lyricssep$lyrics)
# lyricssep <- as.data.frame(lyricssep)
# 
# lyricssep <- gsub('Outro','',lyricssep$lyrics) 
# lyricssep <- as.data.frame(lyricssep)
# 
# lyricssep <- gsub('outro','',lyricssep$lyrics) 
# lyricssep <- as.data.frame(lyricssep)
# 
# lyricssep <- gsub('OUTRO','',lyricssep$lyrics) 
# lyricssep <- as.data.frame(lyricssep)
# 
# lyricssep <- gsub('[0-9]','',lyricssep$lyrics) #removes numbers
# lyricssep <- as.data.frame(lyricssep)

lyricssep <- iconv(lyricssep$lyrics, "UTF-8", "ASCII", sub = "")
lyricssep <- as.data.frame(lyricssep)

#remove lyrics column
lyricstoo <- lyricstoo[,-6]
#bind clean lyrics to dataset
lyricstoo <- cbind(lyricstoo, lyricssep)

lyricstoo <- lyricstoo[order(lyricstoo$genre),] #sort lyrics by genre name

lyricstoo <- lyricstoo[-230861:-362237,] #Remove Rock Genre

lyricstoo <- lyricstoo[-121985:-175481,] #Remove Not Available and Other Genre

lyricstoo <- lyricstoo[order(lyricstoo$lyricssep),] #sort lyrics by letter

lyricstoo <- lyricstoo[-1:-50872,] #remove songs with no lyrics
lyricstoo <- lyricstoo[-7:-24,]
lyricstoo <- lyricstoo[-9:-131,]

lyricstoo <- lyricstoo[order(lyricstoo$index),] #sort lyrics by id number

#47732


for(i in 1:length(lyricstoo$lyricssep)){
  if(stri_count_words(lyricstoo$lyricssep[i]) < 4){
    lyricstoo <- lyricstoo[-i,]
  }
}

# for(i in 1:length(lyricstoo$lyricssep)){
#   if(is.na(stri_count_words(lyricstoo$lyricssep[i])) < 4){
#     lyricstoo <- lyricstoo[-i,]
#   }
# }

#n-gram model setup
lyricstable <- as.vector(lyricstoo$lyricssep) #setting lyrics as a list

lyricstoo$genre <- as.character(lyricstoo$genre)

for(i in 1:length(lyricstoo$genre)){
  if(lyricstoo$genre[i] == "Pop"){
    lyricstoo$genre[i] <- 1
  } else if(lyricstoo$genre[i] == "Metal"){
    lyricstoo$genre[i] <- 2
  } else if(lyricstoo$genre[i] == "Hip-Hop"){
    lyricstoo$genre[i] <-3
  } else if(lyricstoo$genre[i] == "Electronic"){
    lyricstoo$genre[i] <- 4
  } else if(lyricstoo$genre[i] == "Country" ){
    lyricstoo$genre[i] <- 5
  } else if(lyricstoo$genre[i] == "Jazz"){
    lyricstoo$genre[i] <- 6
  } else if(lyricstoo$genre[i] == "Folk"){
    lyricstoo$genre[i] <- 7
  } else if(lyricstoo$genre[i] == "Indie"){
    lyricstoo$genre[i] <- 8
  } else {
    lyricstoo$genre[i] <- 9
  }
}


#run n-gram
#nglyrics <- ngram(lyricstable, n=4)

#check how many n-grams made
#nglyrics

#lyricstoo$genre[1] <- 1

#print out n-grams
#get.ngrams(nglyrics)


#make empty vector to put ngram reuslts
ngtable <- list()
ngtable2 <- list()

for(i in 1:100){
  nglyrics <- ngram(lyricstable[i], n=2)
  ngtemp <- t(get.ngrams(nglyrics))
  ngtable2 <- list.append(ngtable2, ngtemp)
}
# 
# #set list as data frame
ngdata2 <- as.data.frame(t(ngtable2))
ngdata2 <- t(ngdata2)
View(ngdata2)

ngdata3 <- unlist(ngdata2)
ngdata3 <- as.data.frame(ngdata3)

#ngram rWeka style
ngramweka <- NGramTokenizer(lyricstoo$lyricssep)
ngweka <- data.frame()

for(i in 120001:126350){
  nglyricsweka <- NGramTokenizer(lyricstoo$lyricssep[i], Weka_control(min=2,max=2))
  ngtemp <- as.data.frame(paste(unlist(nglyricsweka),collapse = " "))
  ngweka <- rbind(ngweka, ngtemp)
}

#test values
# ngtable[[95]][15]
# ngtable[[1]][2]
# ngtable[[3]][152]
# ngtable[[55]][4]
# ngtable[[100]][45]
# ngtable[[225]][89]
# ngtable[[475]][15]
# ngtable[[725]][43]
# ngtable[[77543]][5]
# ngtable[[121345]][43]

#double brace row, single brace column
#list format

#svm(x, y = NULL, scale = TRUE, type = NULL, kernel =
#"radial", degree = 3, gamma = if (is.vector(x)) 1 else 1 / ncol(x),
#coef0 = 0, cost = 1, nu = 0.5,
#class.weights = NULL, cachesize = 40, tolerance = 0.001, epsilon = 0.1,
#shrinking = TRUE, cross = 0, probability = FALSE, fitted = TRUE,
#..., subset, na.action = na.omit)


x <- lyricstoo$genre
y <- ngramweka


model <- svm(lyricstoo$genre[100] ~ lyricstoo$lyricssep[100], kernel = "radial", type = "C")

# genrelist <- lapply(as.numeric(lyricstoo$genre))
# 
# ngdata <- as.data.frame(ngtable)
# View(ngdata)



sink("ngram.txt")
print(summary(model))
sink()