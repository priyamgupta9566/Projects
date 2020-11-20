import pyttsx3 
import speech_recognition as sr 
import datetime
import wikipedia 
import webbrowser
import os
import calendar
import smtplib
import pandas as pd
import numpy as np

column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)

movie_titles = pd.read_csv("Movie_Id_Titles")
df = pd.merge(df,movie_titles,on='item_id')

#df.head()
df.groupby('title')['rating'].mean().sort_values(ascending=False)#.head()
df.groupby('title')['rating'].count().sort_values(ascending=False)#.head()

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())

moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
ratings.sort_values('num of ratings',ascending=False)#.head(10)

starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']

similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)

corr_starwars.sort_values('Correlation',ascending=False)#.head(10)

corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False)#.head()

corr_liarliar = pd.DataFrame(similar_to_liarliar,columns=['Correlation'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings['num of ratings'])


engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour>=0 and hour<12:
        speak("Good Morning!")

    elif hour>=12 and hour<18:
        speak("Good Afternoon!")   

    else:
        speak("Good Evening!")  

    speak("I am Krypto Sir. Please tell me how may I help you")       

def takeCommand():
    
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print("Recognizing...")    
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}\n")

    except Exception as e:
        # print(e)    
        print("Say that again please...")  
        return "None"
    return query

def tellTime():
    now = datetime.datetime.now()
    meridiem = ''
    if now.hour >=12:
        meridiem = 'p.m'
        hour = now.hour - 12
    else:
        meridiem = 'a.m'
        hour = now.hour    
    speak(now)
    print("The time is sir" + hour + "Hours and" + min + "Minutes")

def tellDay():
    day = datetime.datetime.today().weekday() + 1
    Day_dict = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: "Thursday", 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
    if day in Day_dict.keys():
        day_of_the_week = Day_dict[day]
        print(day_of_the_week)
        speak("The day is " + day_of_the_week)

def tellDate():
    now  = str(datetime.datetime.now())
    my_date = datetime.datetime.today() 
    weekday = calendar.day_name[my_date.weekday()]
    monthNum = now.month
    dayNum = now.day

    month_names = ['January','February','March','April','May','June','July','August','September','October','November','December']

    ordinalNumbers = ['1st','2nd','3rd','4th','5th','6th','7th','8th','9th','10th','11th','12th','13th','14th','15th','16th','17th','18th','19th','20th','21st','22nd','23rd','24th','25th','26th','27th','28th','29th','30th','31st']
    print('Today is: ' + monthNum-1 + ' the ' + ordinalNumbers[dayNum -1])
    speak(ordinalNumbers[dayNum -1] + monthNum-1)

def sendEmail(to, content):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login('monster.02.krypto@gmail.com', 'Krypto@1756')
    server.sendmail('monster.02.krypto@gmail.com', to, content)
    server.close()

def getPerson(text):
    wordList = text.split()
    for i in range(0,len(wordList)):
        if i+3 <= len(wordList) -1 and wordList[i].lower() == 'who' and wordList[i+1].lower() == 'is':
            return wordList[i+2] + ' ' + wordList[i+3]

if __name__ == "__main__":
    wishMe()
    while True:
        query = takeCommand().lower()

        if 'wikipedia' in query:
            speak('Searching Wikipedia...')
            query = query.replace("wikipedia", "")
            results = wikipedia.summary(query, sentences=2)
            speak("According to Wikipedia")
            print(results)
            speak(results)

        elif 'who is' in query:
            person = getPerson(query)
            wiki = wikipedia.summary(person, sentences = 2)
            print(wiki)
            speak(wiki)

        elif 'open youtube' in query:
            webbrowser.open("youtube.com")
            print("You Tube is opened")

        elif 'open google' in query:
            webbrowser.open("google.com")
            print("Google is opened")

        elif 'open stackoverflow' in query:
            webbrowser.open("stackoverflow.com")   


        elif 'play music' in query:
            music_dir = 'G:\\Pokemon\\songs'
            songs = os.listdir(music_dir)
            print(songs)    
            os.startfile(os.path.join(music_dir, songs[0]))

        elif 'the time' in query:
            strTime = datetime.datetime.now().strftime("%H:%M:%S")    
            speak(f"Sir, the time is {strTime}")

        elif 'open code' in query:
            codePath = "C:\\Users\\Priyam Gupta\\Documents\\assistCode.txt"
            os.startfile(codePath)

        elif "which day it is" in query:
            tellDay()

        elif "tell me the time" in query:
            tellTime()

        elif "tell me the date" in query:
            tellDate()

        elif "tell me your name" in query:
            speak("I am Krypto Sir. Please tell me how may I help you")

        elif 'email to Priyam' in query:
            try:
                speak("What should I say?")
                content = takeCommand()
                to = "priyamguptaa13@gmail.com"    
                sendEmail(to, content)
                speak("Email has been sent!")
            except Exception as e:
                print(e)
                speak("Sorry my friend Priyam. I am not able to send this email")    
        
        elif 'recommend a movie' in query:
            try:
                speak("Which is your favourite movie?")
                ans = takeCommand()
                if(ans == "Star Wars"):
                    speak("Here are some recommendations")
                    print(corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head())
                elif(ans == "comedy"):
                    speak("Here are some recommendations")
                    print(corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('Correlation',ascending=False).head())
                else:
                    speak("Sorry Sir I am unable to reccomend")
            except Exception as e:
                print(e)
                speak("Sorry Sir I am unable to reccomend")
                
        elif 'bye' in query:
            speak("Good Bye Sir Have a nice day")
            exit()       