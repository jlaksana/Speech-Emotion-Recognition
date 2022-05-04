import random
from tkinter import *

emotions = ["Neutral", "Happy", "Anger", "Disgust", "Fear", "Sad"]
emoteColors = {
    "Neutral": "black",
    "Happy": "#ffd966",
    "Anger": "red",
    "Disgust": "#6aa84f",
    "Fear": "#a64d79",
    "Sad": "blue"
}


def onClick():
    print("record button was clicked")
    recordBtn["state"] = DISABLED
    # record audio
    btnText.set("Recording")

    # determine emotion
    btnText.set("Loading...")

    i = random.randint(0, 5)
    newEmotion = emotions[i]
    emotionLabel["text"] = newEmotion
    emotionLabel["fg"] = emoteColors[newEmotion]

    confidence = random.randint(0, 100)
    confidenceLabel["text"] = f"{confidence}% Confidence"

    btnText.set("Record")
    recordBtn["state"] = NORMAL


if __name__ == "__main__":
    root = Tk()
    root.title("Speech Emotion Recognition")

    canvas = Canvas(root, width=800, height=600)
    canvas.grid(columnspan=3, rowspan=7)

    # all titles and labels
    title = Label(root, text="Speech Emotion Recognition AI",
                  font=("Roboto 20 bold"))
    title.grid(row=1, column=1)

    subtitle = Label(root, text="Emotion Detected:", font=("Roboto 14"))
    subtitle.grid(row=4, column=1)

    emotionLabel = Label(root, text="None", font=(
        "Roboto 16 bold"), fg="black")
    emotionLabel.grid(row=5, column=1)

    confidenceLabel = Label(
        root, text="", font=("Roboto 12"), fg="gray")
    confidenceLabel.grid(row=6, column=1)

    # buttons
    btnText = StringVar()
    btnText.set("Record")
    recordBtn = Button(root, textvariable=btnText, command=onClick,
                       fg="white", bg="#3c78d8", font=("Roboto 14 bold"), padx=30, pady=10)
    recordBtn.grid(row=2, column=1)

    root.mainloop()
