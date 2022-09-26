import wave, math, contextlib
import speech_recognition as sr
from moviepy.editor import AudioFileClip
import sys, os
import nltk, operator
from pandas import DataFrame
import matplotlib.pyplot as plt

def converter_video_para_audio(video, audio):
    audioclip = AudioFileClip(video)
    audioclip.write_audiofile(audio)
    with contextlib.closing(wave.open(audio, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration

def converter_audio_para_texto(audio_file, duration):
    total_duration = math.ceil(duration / 60)
    r = sr.Recognizer()
    for i in range(0, total_duration):
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source, offset=i*60, duration=60)
            text = r.recognize_google(audio, language='en-GB', show_all=True)
            f = open(legenda, "a")
            f.write(text['alternative'][0]['transcript'] + ' ')
            f.close()

def processamento_texto_para_df(legenda):
    with open(legenda, 'r', encoding='utf-8') as f:
        texto = f.read()
    print(texto)
    texto_tokens = nltk.word_tokenize(texto)
    text1 = nltk.Text(texto_tokens)
    x = sorted([(freq, token) for token, freq in text1.vocab().items() if len(token) > 5 and freq > 2], key=operator.itemgetter(0), reverse=True)
    df = DataFrame(x, columns=['freq', 'words'])
    return df

def exibicao_em_grafico(df):
    ax= plt.gca()
    colors = ['#a3004c', '#bb3281', '#cd51ae', '#a480cf', '#8e8edb', '#93a9e9', '#98c4f6', '#98c4f6', '#99d6ff']
    df.plot(kind='pie', colors=colors, y = 'freq', ax=ax, startangle=90, shadow=False, autopct='%1.f%%', labels=df['POS Tags'], legend=False, fontsize=11)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    ax.set_xlabel('* Words longer than 3 characters',horizontalalignment='left', verticalalignment='top', labelpad=20)
    plt.ylabel('')
    ax.axis('equal')
    ax.set_title('Most used words in Charles III first speech', fontsize=15, pad=21)
    plt.tight_layout()
    plt.show()

try:
    video = "charlesspeak.mp4"
    audio = "audio.wav"
    legenda = "legenda.txt"

    duration = converter_video_para_audio(video, audio)
    converter_audio_para_texto(audio, duration)
    df = processamento_texto_para_df(legenda)
    df['POS Tags'] = nltk.pos_tag(df['words'])
    exibicao_em_grafico(df)
except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)
