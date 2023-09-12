
import cv2
import numpy as np
from mido import Message, get_output_names, open_output
import logging
import json
import tkinter as tk
from tkinter import ttk
import threading
from queue import Queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import interp1d
from collections import deque
import time
from tkinter import filedialog

# Global variables
width, height = None, None
note_name = None
outport = None
video_mode = True
data_queue = Queue()
video_file_path = "./hand.mp4"  # default path
exit_flag = False

def toggle_mode():
    global video_mode
    video_mode = not video_mode
    mode_label.config(text="Webcam" if not video_mode else "Video")

def start_opencv():
    global outport
    if outport is None:
        print("Please select a MIDI port before starting the video.")
    else:
        opencv_thread = threading.Thread(target=run_opencv)
        opencv_thread.start()

def browse_file():
    global video_file_path
    video_file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
    video_file_var.set(video_file_path)


# Create a new function to start the plotting thread
def start_plotting():
    plotting_thread = threading.Thread(target=update_gui)
    plotting_thread.daemon = True  # Daemonize thread
    plotting_thread.start()
        
# Global variables to keep track of the running list of points for better performance

running_x_data = deque(maxlen=10000)  # Arbitrarily chosen limit
running_y_data = deque(maxlen=10000)  # Arbitrarily chosen limit
note_name = "A"

def update_gui():
    while True:  # Keep running this function
        x_data, y_data = [], []

        # Drain the queue into x_data and y_data
        while not data_queue.empty():
            x, y = data_queue.get()
            if (x, y) != (960, 540):
                x_data.append(x)
                y_data.append(y)

        if x_data and y_data:
            ax.clear()  # Clear the axis once
            
            # Extend the running data
            running_x_data.extend(x_data)
            running_y_data.extend(y_data)

            # Annotate only the last point in the running data
            ax.annotate(note_name, (running_x_data[-1], running_y_data[-1]), textcoords="offset points", xytext=(0,10), ha='center')

            # Plot the data
            ax.scatter(running_x_data, running_y_data, 1, c="green", alpha=0.5, marker='x', label="note")
            ax.scatter(running_x_data[-1], running_y_data[-1], 50, c="red", marker='o')  # Emphasize the last point
            
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
            
            canvas.draw()

        time.sleep(0.001)  # Sleep for a short time to not hog the CPU
        #root.after(10, update_gui)




def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Log to file
    file_handler = logging.FileHandler('hand_to_midi.log')
    logger.addHandler(file_handler)
    
    # Log to console
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

class MidiPort:
    def __init__(self, port_name):
        self.port = open_output(port_name)
        self.initial_note_sent = False

    def send_message(self, message):
        if self.initial_note_sent:
            self.port.send(message)
        else:
            self.initial_note_sent = True

    def close(self):
        self.port.close()

def load_settings():
    global outport, video_file_path # Use the global outport variable
    try:
        with open('slider_settings.json', 'r') as file:
            settings = json.load(file)
        history_slider.set(settings['history'])
        var_threshold_slider.set(settings['var_threshold'])
        detect_shadows_checkbox_var.set(settings['detect_shadows'])
        
        # Check if 'midi_port' is in settings before using it
        if 'midi_port' in settings:
            midi_port_dropdown.set(settings['midi_port'])
            global outport
            outport = MidiPort(settings['midi_port'])
        if 'video_file_path' in settings:
            video_file_path = settings['video_file_path']
            video_file_var.set(video_file_path)
    except FileNotFoundError:
        pass

def save_settings():
    settings = {
        'history': history_slider.get(),
        'var_threshold': var_threshold_slider.get(),
        'detect_shadows': detect_shadows_checkbox_var.get(),
        'midi_port': midi_port_dropdown.get(),
        'video_file_path': video_file_path
    }
    with open('slider_settings.json', 'w') as file:
        json.dump(settings, file)

def restore_defaults():
    history_slider.set(500)
    var_threshold_slider.set(16)
    detect_shadows_checkbox_var.set(1)

midi_note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def note_number_to_name(note_number):
    octave = note_number // 12 - 1
    note_name = midi_note_names[note_number % 12]
    return f"{note_name}{octave}"



def send_midi(x, y, width, height):
    global note_name
    note_number = min(int(x / width * 127), 127)
    pitch = int((y / height) * 4095) - 2048
    note_name = note_number_to_name(note_number)
    
    logging.info(f"Sending MIDI note: {note_name}, pitch: {pitch}")

    # Sending MIDI messages
    # outport.send_message(Message('note_on', note=note_number, velocity=64, time=0))
    # outport.send_message(Message('pitchwheel', pitch=pitch, time=0))
    outport.send_message(Message('control_change', value=note_number))


def update_plot_dimensions(width, height):
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)


def run_opencv():
    global width, height, outport, video_mode, note_name, exit_flag  # Declare global variables


    if outport is None:
        print("Please select a MIDI port before starting the video.")
        return

    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    cap = None

    if video_mode:
        cap = cv2.VideoCapture(video_file_path)
    else:
        cap = cv2.VideoCapture(0)

    if cap is None or not cap.isOpened():
        print("Could not open video source.")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Now it's safe to access cap
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    update_plot_dimensions(width, height)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    fps = int(cap.get(cv2.CAP_PROP_FPS))





    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Frame is empty or stream has ended. Exiting...")
            break
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Continue processing gray frame
        except cv2.error as e:
            print(f"OpenCV error: {e}")
            break

        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord('q') or exit_flag:
            save_settings()
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        history = int(history_slider.get())  # convert to int here
        var_threshold = int(var_threshold_slider.get())  # and here if needed
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
                #print(f"Enqueued: x={cX}, y={cY}")  # Debugging (Step 3)
                cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
                send_midi(cX, cY, width, height)
        if exit_flag:
            break
        cv2.imshow('Hand Tracking and Slider Settings', frame)
        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord('q'):
            save_settings()
            break

    # Properly release resources and close windows
    if cap is not None:
        cap.release()
    outport.close()
    cv2.destroyAllWindows()

    
    exit_flag = False

def worker_thread():
    while not exit_flag:
        worker = threading.Thread(target=worker_thread)
        worker.daemon = True  # This ensures the thread will exit when the main program exits
        worker.start()
        time.sleep(0.001)  # Sleep for a short time to not hog the CPU
        pass

def on_closing():
    global exit_flag
    exit_flag = True
    root.destroy()


if __name__ == "__main__":
    setup_logging()
    logging.basicConfig(filename='hand_to_midi.log', level=logging.INFO)

    exit_flag = False
    data_queue = Queue()

    # Initialize Tkinter window
    root = tk.Tk()
    root.title('Hand Tracking and Slider Settings')



    # Create Frames
    sliders_frame = ttk.LabelFrame(root, text='Settings')
    sliders_frame.grid(row=0, column=0, padx=10, pady=10)
    
    midi_frame = ttk.LabelFrame(root, text='MIDI Port')
    midi_frame.grid(row=0, column=1, padx=10, pady=10)

    # Create MIDI port dropdown
    midi_port_label = ttk.Label(midi_frame, text="Select MIDI Port:")
    midi_port_label.pack(side=tk.TOP, pady=5)

    midi_ports = get_output_names()
    midi_port_var = tk.StringVar()
    midi_port_dropdown = ttk.Combobox(midi_frame, textvariable=midi_port_var)
    midi_port_dropdown['values'] = midi_ports
    midi_port_dropdown.pack(side=tk.TOP, pady=5)

    # Add this code right here to bind the selection event to set_midi_port
    def set_midi_port(event):
        global outport
        selected_port = midi_port_var.get()
        outport = MidiPort(selected_port)

    midi_port_dropdown.bind("<<ComboboxSelected>>", set_midi_port)

    # Create Video file browse button and label
    video_file_var = tk.StringVar()
    video_file_var.set(video_file_path)  # set it to default or loaded path

    video_file_label = ttk.Label(midi_frame, text="Select Video File:")
    video_file_label.pack(side=tk.TOP, pady=5)

    video_file_button = ttk.Button(midi_frame, text='Browse', command=browse_file)
    video_file_button.pack(side=tk.TOP, pady=5)

    video_file_display_label = ttk.Label(midi_frame, textvariable=video_file_var)
    video_file_display_label.pack(side=tk.TOP, pady=5)

    # Add a Start button
    start_button = ttk.Button(root, text='Start', command=start_opencv)
    start_button.grid(row=1, column=0, padx=10, pady=10)

    # Add a toggle button for video/webcam
    toggle_button = ttk.Button(root, text='video/cam', command=toggle_mode)
    toggle_button.grid(row=1, column=1, padx=10, pady=10)

    # Add a label to show current mode
    mode_label = ttk.Label(root, text="Video")
    mode_label.grid(row=1, column=2, padx=10, pady=10)

    # Create Sliders and Buttons
    history_slider_label = ttk.Label(sliders_frame, text='History')
    history_slider_label.pack(side=tk.TOP, pady=5)
    history_slider = ttk.Scale(sliders_frame, from_=0, to=1000, orient=tk.HORIZONTAL)
    history_slider.pack(side=tk.TOP, pady=5)

    var_threshold_slider_label = ttk.Label(sliders_frame, text='VarThreshold')
    var_threshold_slider_label.pack(side=tk.TOP, pady=5)
    var_threshold_slider = ttk.Scale(sliders_frame, from_=0, to=100, orient=tk.HORIZONTAL)
    var_threshold_slider.pack(side=tk.TOP, pady=5)

    detect_shadows_checkbox_var = tk.IntVar()
    detect_shadows_checkbox = ttk.Checkbutton(sliders_frame, text='DetectShadows', variable=detect_shadows_checkbox_var)
    detect_shadows_checkbox.pack(side=tk.TOP, pady=5)

    save_button = ttk.Button(sliders_frame, text='Save Settings', command=save_settings)
    save_button.pack(side=tk.TOP, pady=5)

    restore_defaults_button = ttk.Button(sliders_frame, text='Restore Defaults', command=restore_defaults)
    restore_defaults_button.pack(side=tk.TOP, pady=5)

    load_settings()

    plot_frame = ttk.LabelFrame(root, text='Plotting')
    plot_frame.grid(row=0, column=2, padx=10, pady=10)

    fig, ax = plt.subplots(figsize=(5, 4))
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)





 # schedule update_gui to be called

    root.protocol("WM_DELETE_WINDOW", on_closing)
    start_plotting()
    #root.after(10, update_gui) 
    root.mainloop()  # start Tkinter event loop

