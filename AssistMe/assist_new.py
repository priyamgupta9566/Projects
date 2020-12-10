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
import pyowm


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

    speak("I am AssistMe Sir. Please tell me how may I help you")       

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

    except:
        print("Say that again please...")  
        return "None"
    return query

def tellTime():
    now = datetime.datetime.now()
    meridiem = ''
    if now.hour >=12:
        meridiem = 'p m'
        hour = now.hour - 12
    else:
        meridiem = 'a m'
        hour = now.hour
    if now.minute < 10:
        minute = '0' + str(now.minute)
    else:
        minute = str(now.minute)
    print("Sir, The time is " + str(hour) + ":" + minute + " " + " " + meridiem + " .")
    speak("Sir, The time is " + str(hour) + ":" + minute + " " + " " + meridiem + " .")

def tellDay():
    day = datetime.datetime.today().weekday() + 1
    Day_dict = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: "Thursday", 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
    if day in Day_dict.keys():
        day_of_the_week = Day_dict[day]
        print(day_of_the_week)
        speak("The day is " + day_of_the_week)

def tellDate():
    now  = datetime.datetime.now()
    my_date = datetime.datetime.today() 
    weekday = calendar.day_name[my_date.weekday()]
    monthNum = now.month
    dayNum = now.day

    month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

    ordinalNumbers = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th', '11th', '12th', '13th', '14th', '15th', '16th', '17th', '18th', '19th', '20th', '21st', '22nd', '23rd', '24th', '25th', '26th', '27th', '28th', '29th', '30th', '31st']
    print('Today is: '+weekday  + ' ' + month_names[monthNum-1] + ' the ' + ordinalNumbers[dayNum -1])
    speak('Today is: '+weekday  + ' ' + month_names[monthNum-1] + ' the ' + ordinalNumbers[dayNum -1])

def sendEmail(to, content):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login('Sender Mail ID', 'Password')
    server.sendmail('Sender Mail ID', to, content)
    server.close()

def getPerson(text):
    wordList = text.split()
    for i in range(0,len(wordList)):
        if i + 3 <= len(wordList) - 1 and wordList[i].lower() == 'who' and wordList[i+1].lower() == 'is':
            return wordList[i+2] + ' ' + wordList[i+3]

def getDetails(text):
    wordList = text.split()
    for i in range(0,len(wordList)):
        if i+3 <= len(wordList) -1 and wordList[i].lower() == 'what' and wordList[i+1].lower() == 'is':
            return wordList[i+2] + ' ' + wordList[i+3]


def getWeather():
    try:
        speak("Please tell me your location.")
        ans = str(takeCommand().lower())
        owm = pyowm.OWM('b9312e4c6e525d2065112ae1f7091fce')
        location = owm.weather_at_place('New Delhi,India')
        weather = location.get_weather()
        speak("Sir, according to your location the weather information is ")
        for key,value in weather.items():
            print("The weather information is: ",weather)
        temp = weather.get_temprature('celsius')
        for key,value in temp.items():
            print("The temprature of current location is: ",temp)
        humidity = weather.get_humidity()
        print(humidity)
        for key,value in  weather.items():
            speak(weather)
        for key,value in temp.items():
            speak(temp)
        speak(humidity)
            
    except :
        speak("Sorry Sir, I am unable to forecast weather.")
        
def addNumbers():
    print("How many numbers do you want to add.")
    speak("How many numbers do you want to add.")
    ans = str(takeCommand().lower())
    ans1 = int(ans)
    print("Please tell the numbers to add.")
    speak("Please tell the numbers to add.")
    list1 = []
    for i in range(0,ans1):
        num = str(takeCommand())
        num1 = int(num)
        list1.append(num1)
    total = sum(list1)
    print("The Sum of numbers is: ",total)
    speak("The Sum of numbers is: ",total)

if __name__ == "__main__":
    wishMe()
    while True:
        query = takeCommand().lower()

        if 'search in Wikipedia' in query:
            speak('Searching Wikipedia...')
            query = query.replace("search in Wikipedia", "")
            results = wikipedia.summary(query, sentences=3)
            speak("According to Wikipedia")
            print(results)
            speak(results)

        elif 'who is' in query:
            query = query.replace("who is","")
#            person = getPerson(ans)
            wiki = wikipedia.summary(query, sentences = 2)
            print(wiki)
            speak(wiki)

        elif 'what is' in query:
            query = query.replace("what is", "")
 #           person = getDetails(query)
            wiki = wikipedia.summary(query, sentences = 2)
            print(wiki)
            speak(wiki)
    

        elif 'open youtube' in query:
            webbrowser.open("youtube.com")
            print("You Tube is opened")

        elif 'open google' in query:
            webbrowser.open("google.com")
            print("Google is opened")

        elif 'open gmail' in query:
            webbrowser.open("gmail.com")

        elif 'open netflix' in query:
            webbrowser.open("netflix.com")    

        elif 'play music' in query:
            speak("Would you like me to open Spotify for you?")
            ans = takeCommand().lower()
            if(ans == "yes"):
                path = "G:\\WpSystem\\S-1-5-21-166467239-3098487816-1372498879-1001\\AppData\\Local\\Packages"
                os.startfile(path)
            else:    
                music_dir = 'G:\\Pokemon\\songs'
                songs = os.listdir(music_dir)
                os.startfile(os.path.join(music_dir, songs[0]))

        elif 'what time it is' in query:
            strTime = datetime.datetime.now().strftime("%H:%M:%S")    
            speak(f"Sir, the time is {strTime}")
            print(f"Sir, the time is {strTime}")

        elif 'open code' in query:
            codePath = "C:\\Users\\Documents\\assistCode.txt"
            os.startfile(codePath)

        elif "day" in query:
            tellDay()

        elif "time" in query:
            tellTime()

        elif "date" in query:
            tellDate()

        elif "tell me your name" in query:
            speak("I am AssistMe Sir. I am you personal virtual assistant. Please tell me how may I help you")

        elif "send an email" in query:
            try:
                speak("What should I say?")
                content = takeCommand()
                if "terminate" in query:
                    print("Okay process has been terminated")
                    speak("Okay process has been terminated")
                    break
                to = "Reciever's Mail ID"    
                sendEmail(to, content)
                speak("Email has been sent!")
                print("Email has been sent.")
            except:
                speak("Sorry Sir. I am not able to send this email")    
        
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
                    speak("Sorry Sir I am unable to recommend")
            except:
                speak("Sorry Sir I am unable to recommend")
                
        
        elif 'weather' in query:
            getWeather()

        elif 'add numbers' in query:
            addNumbers()    

        elif 'bye' in query:
            print("Good Bye Sir Have a nice day")
            speak("Good Bye Sir Have a nice day")
            exit()       