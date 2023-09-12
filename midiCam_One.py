import cv2
import numpy as np
from mido import Message, get_output_names, open_output
import logging
import json
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import threading
from queue import Queue

data_queue = Queue()

def update_gui():  # Step 3
    while not data_queue.empty():
        data = data_queue.get()
        # You can update Tkinter GUI here if needed.
        # E.g., update labels, images, etc. based on the data
    root.after(100, update_gui)

class MidiPort:
    def __init__(self, port_name):
        self.port = open_output('monk 1')
        # self.port = open_output(port_name)
        self.initial_note_sent = False  # Flag to track if initial note is sent

    def send_message(self, message):
        if self.initial_note_sent:  # Only send messages after initial note
            self.port.send(message)
        else:
            self.initial_note_sent = True

    def close(self):
        self.port.close()

def log_and_plot(x, y, note, pitch):
    logging.info(f"MIDI note: {note}, pitch: {pitch}")
    plt.scatter(x, 1080 - y, c='r', marker='x')  # Invert y-coordinate by subtracting from 1080
    plt.xlim(0, 1920)  # match the video width
    plt.ylim(0, 1080)  # match the video height
    plt.pause(0.001)

midi_note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def note_number_to_name(note_number):
    octave = note_number // 12 - 1
    note_name = midi_note_names[note_number % 12]
    return f"{note_name}{octave}"

def send_midi(x, y):
    note_number = int((x / 1920) * 127)  # Scale x-coordinate to cover the full MIDI note range
    note_number = min(max(0, note_number), 127)  # Clamp the value to be in the range [0, 127]

    pitch = int((y / 1080) * 4095) - 2048
    note_name = note_number_to_name(note_number)

    print(f"Sending MIDI note: {note_name}, pitch: {pitch}, x: {x}, note_number: {note_number}")

    # outport.send_message(Message('note_on', note=note_number, velocity=64, time=0))
    outport.send_message(Message('control_change', value=note_number))
    # outport.send_message(Message('pitchwheel', pitch=pitch, time=0))

    log_and_plot(x, y, note_name, pitch)

def save_settings():
    history = history_slider.get()
    var_threshold = var_threshold_slider.get()
    detect_shadows = detect_shadows_checkbox_var.get()
    settings = {
        'history': history,
        'var_threshold': var_threshold,
        'detect_shadows': detect_shadows
    }
    with open('slider_settings.json', 'w') as file:
        json.dump(settings, file)

def load_settings():
    try:
        with open('slider_settings.json', 'r') as file:
            settings = json.load(file)
        history_slider.set(settings['history'])
        var_threshold_slider.set(settings['var_threshold'])
        detect_shadows_checkbox_var.set(settings['detect_shadows'])
    except FileNotFoundError:
        pass

def restore_defaults():
    history_slider.set(500)
    var_threshold_slider.set(16)
    detect_shadows_checkbox_var.set(1)

def run_opencv():
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    # Change between movie and webCam input here:

    ###### movie:

    movie_file_path = './hand.mp4'
    cap = cv2.VideoCapture(movie_file_path)

    ###########################

    ###### webcam:
    #cap = cv2.VideoCapture(0)
    ###########################

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        history = history_slider.get()
        var_threshold = var_threshold_slider.get()
        detect_shadows = detect_shadows_checkbox_var.get()

        bg_subtractor.setHistory(history)
        bg_subtractor.setVarThreshold(var_threshold)
        bg_subtractor.setDetectShadows(bool(detect_shadows))

        fg_mask = bg_subtractor.apply(gray)
        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=3)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_area = -1

        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                ci = i

        if max_area != -1:
            cnt = contours[ci]
            hull = cv2.convexHull(cnt)
            cv2.drawContours(frame, [hull], 0, (0, 255, 0), 2)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                data_queue.put((cX, cY))
                cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
                send_midi(cX, cY)

        cv2.imshow('Hand Tracking and Slider Settings', frame)
        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord('q'):
            save_settings()
            break

    outport.close()
    cap.release()
    cv2.destroyAllWindows()

logging.basicConfig(filename='hand_to_midi.log', level=logging.INFO)
output_name = get_output_names()[0]
outport = MidiPort(output_name)

root = tk.Tk()
root.title('Hand Tracking and Slider Settings')

sliders_frame = tk.Frame(root)
sliders_frame.pack(side=tk.LEFT, padx=10)

buttons_frame = tk.Frame(root)
buttons_frame.pack(side=tk.LEFT, padx=10)

history_slider_label = tk.Label(sliders_frame, text='History')
history_slider_label.pack()
history_slider = tk.Scale(sliders_frame, from_=0, to=1000, orient=tk.HORIZONTAL)
history_slider.pack()

var_threshold_slider_label = tk.Label(sliders_frame, text='VarThreshold')
var_threshold_slider_label.pack()
var_threshold_slider = tk.Scale(sliders_frame, from_=0, to=100, orient=tk.HORIZONTAL)
var_threshold_slider.pack()

detect_shadows_checkbox_var = tk.IntVar()
detect_shadows_checkbox = tk.Checkbutton(sliders_frame, text='DetectShadows', variable=detect_shadows_checkbox_var)
detect_shadows_checkbox.pack()

save_button = tk.Button(sliders_frame, text='Save Settings', command=save_settings)
save_button.pack()

restore_defaults_button = tk.Button(sliders_frame, text='Restore Defaults', command=restore_defaults)
restore_defaults_button.pack()

load_settings()
root.after(100, update_gui)

opencv_thread = threading.Thread(target=run_opencv)
opencv_thread.start()

root.mainloop()
