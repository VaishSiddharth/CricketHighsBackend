import time
import cv2
import mss
import numpy
import os
import shutil
from multiprocessing import Process, Queue

import mss
import mss.tools

from os.path import isfile, join

import sounddevice as sd
from scipy.io.wavfile import write
import moviepy.editor as mpe

def make_video():
    pathIn = 'images/'
    pathOut = 'video.avi'
    fps = 30.0
    convert_frames_to_video(pathIn, pathOut, fps)

def convert_frames_to_video(pathIn, pathOut, fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    # for sorting the file names properly
    files.sort(key=lambda x: int(x[5:-4]))

    for i in range(len(files)):
        filename = pathIn + files[i]
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        print(filename)
        # inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
    combine_audio_video()


def combine_audio_video():
    my_clip = mpe.VideoFileClip('video.avi')
    audio_background = mpe.AudioFileClip('output.wav')
    final_clip = my_clip.set_audio(audio_background)
    final_clip.write_videofile("movie.mp4", fps=30)
    cleanup()

def cleanup():
    if os.path.exists('movie.mp4'):
        # str(os.getcwd()) + '/' +
        shutil.rmtree('images')
        os.remove('output.wav')
        os.remove('video.avi')

def record_audio(myrecording):
    sd._terminate()
    sd._initialize()
    fs = 44100  # Sample rate
    seconds = 16  # Duration of recording
    # sd.default.device = 9
    print(type(myrecording))
    myrec = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()
    myrecording.put(myrec)

    # print(type(myrecording))
    # sd.wait()  # Wait until recording is finished
    # print("raudio" + str(myrec))
    # write('output.wav', fs, myrecording)  # Save as WAV file



def grab(queue,myrecording):
    # type: # (Queue) -> None
    rect = {"top": 0, "left": 0, "width": 1920, "height": 1080}

    with mss.mss() as sct:
        sct.compression_level=9
        last_time = time.time()
        x=0
        while x<500:
            if (1 / (time.time() - last_time)) <= 30:
                x=x+1
                last_time = time.time()
                queue.put(sct.grab(rect))
            print("fps: {}".format(1 / (time.time() - last_time)))

    recording=myrecording.get()
    print("grab"+str(recording))
    write('output.wav', 44100, recording)
    # Tell the other worker to stop
    queue.put(None)


def save(queue):
    # type: # (Queue) -> None

    number = 0
    output = "images/file_{}.png"
    to_png = mss.tools.to_png

    while "there are screenshots":
        img = queue.get()
        if img is None:
            break

        print("Saving")
        to_png(img.rgb, img.size, output=output.format(number))
        number += 1



if __name__ == "__main__":
    # The screenshots queue
    # shutil.rmtree('images')
    queue = Queue()
    myrecording=Queue()
    print(type(myrecording))
    if not os.path.exists('images'):
        os.makedirs('images')

    # 2 processes: one for grabing and one for saving PNG files
    p1=Process(target=grab, args=(queue,myrecording))
    p2=Process(target=save, args=(queue,))
    p3 = Process(target=record_audio, args=(myrecording,))
    p1.start()
    p3.start()
    p2.start()

    p1.join()
    p2.join()
    p3.join()

    make_video()
