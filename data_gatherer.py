# Import required Libraries
from typing import Union, NamedTuple

import tkinter.messagebox
from tkinter import *
from tkinter import filedialog as fd
from PIL import Image, ImageTk
import pandas as pd
import cv2
import mediapipe as mp

# MediaPipe config
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

filename: Union[None, str] = None
df: Union[None, pd.DataFrame] = None
results: Union[None, NamedTuple] = None
count_labels: Union[None, list] = None


def add_data(letter: str) -> None:
    global filename, df, results
    if filename is None or df is None:
        tkinter.messagebox.showerror(
            'DataFrame Error',
            'Error: No DataFrame loaded! Create new file or load an existing one!'
        )
    elif results.multi_hand_landmarks is not None:
        landmarks = []
        for ld in results.multi_hand_landmarks[0].landmark:
            landmarks += [ld.x, ld.y, ld.z]
        for ld in results.multi_hand_landmarks[0].landmark:
            landmarks += [ld.x, ld.y, ld.z]
        landmarks += [results.multi_handedness[0].classification[0].score]
        landmarks += [letter]
        df.loc[len(df)] = landmarks
        print(f'{df=}')
        df.to_csv(filename)
        upadate_text()

def load_file():
    global filename, df
    filename = fd.askopenfilename()
    df = pd.read_csv(filename, index_col=0)
    print(f'{df=}')
    upadate_text()

def upadate_text():
    global count_labels, df

    signs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n',
             'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

    for i, letter in enumerate(signs):
        if letter in df['letter'].value_counts().index:
            count = df['letter'].value_counts()[letter]
        else:
            count = 0
        count_labels[i].config(text=str(count), font=("Arial", 16))

def new_file():
    global filename, df
    ans = tkinter.messagebox.askyesno(
        title='Are you sure?',
        message='Are you sure you want to create new dataset file? It will erase previously '
                'existing file with all its data and the current DataFrame!'
    )
    if ans:
        file = fd.asksaveasfile(
            initialfile='dataset.csv',
            defaultextension='.csv',
            filetypes=[('All Files', '*.*'), ('csv files', '*.csv')], mode='w'
        )
        filename = file.name
        file.close()
        columns = ['landmark_'+str(i)+'.'+a for i in range(21) for a in ['x', 'y', 'z']]
        columns += ['world_landmark_' + str(i) + '.' + a for i in range(21) for a in ['x', 'y', 'z']]
        columns += ['handedness', 'letter']
        df = pd.DataFrame(columns=columns)
        df.to_csv(filename)


def backup_data():
    global df
    if type(df) == pd.DataFrame:
        df.to_csv('backup.csv')
    else:
        tkinter.messagebox.showerror(
            'DataFrame Error',
            'Error: No DataFrame loaded! Create new file or load an existing one!'
        )


def main():
    # Create an instance of TKinter Window or frame
    win = Tk()
    menu = Menu(win)
    win.config(menu=menu)
    file_menu = Menu(menu, tearoff=False)
    menu.add_cascade(label='File', menu=file_menu)
    file_menu.add_command(label='New dataset file', command=new_file)
    file_menu.add_command(label='Load dataset file', command=load_file)
    file_menu.add_command(label='Backup dataset file', command=backup_data)
    file_menu.add_command(label='Exit', command=win.destroy)

    # Set the size of the window
    win.geometry('1920x1080')

    left_label = Label(win)
    left_label.grid(row=0, column=0)

    # Create a Label to capture the Video frames
    video_label = Label(left_label)
    video_label.grid(row=0, column=0)

    cap = cv2.VideoCapture(0)

    # Define signs and create buttons
    button_label = Label(left_label)
    button_label.grid(row=1, column=0)
    signs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n',
             'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']
    photos = {letter: PhotoImage(file=r'./sign/'+letter+'.png') for letter in signs}

    _ = [
       Button(
           button_label, text=letter, image=photos[letter],
           command=lambda letter=letter:
           add_data(letter)).grid(row=int(i/6), column=i % 6) for i, letter in enumerate(signs)
    ]

    global count_labels
    right_label = Label(win)
    right_label.grid(row=0, column=1)
    _ = [Label(right_label, text=letter.upper()+': ', font=("Arial", 16, 'bold')).grid(row=i, column=0)
         for i, letter in enumerate(signs)]
    count_labels = [Label(right_label, text='None', font=("Arial", 16)) for i, letter in enumerate(signs)]
    for i, label in enumerate(count_labels):
        label.grid(row=i, column=1)

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1
    ) as hands:
        # Define function to show frame
        def show_frames():
            global results
            # Get the latest frame and convert into Image
            cv2image = cap.read()[1]
            # To improve performance, optionally mark the image as not writeable to pass
            # by reference.
            cv2image.flags.writeable = False
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
            results = hands.process(cv2image)

            cv2image.flags.writeable = True
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        cv2image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            img = Image.fromarray(cv2image)
            img = img.resize((int(img.size[0]*1080/img.size[1]/2.1), int(1080/2.1)))

            # Convert image to PhotoImage
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
            # Repeat after an interval to capture continuously
            video_label.after(20, show_frames)

        show_frames()
        win.mainloop()


if __name__ == '__main__':
    main()
