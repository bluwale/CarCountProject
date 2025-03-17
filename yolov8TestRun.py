'''
from flask import Flask, render_template, Response, url_for, request, redirect
import cv2
from flask_socketio import SocketIO
from ultralytics import solutions
from threading import Lock
from datetime import datetime
from random import random


cap = cv2.VideoCapture("Cars in Highway Traffic (FREE STOCK VIDEO).mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points
# region_points = [(20, 400), (1080, 400)]  # For line counting
region_points = [(20, 300), (1080, 300), (1080, 360), (20, 360)]  # For rectangle region counting
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360), (20, 400)]  # For polygon region counting

# Video writer
video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init ObjectCounter
counter = solutions.ObjectCounter(
    show=True,  # Display the output
    region=region_points,  # Pass region points
    model="yolo11n.pt",  # model="yolo11n-obb.pt" for object counting using YOLO11 OBB model.
    # classes=[0, 2],  # If you want to count specific classes i.e person and car with COCO pretrained model.
    # show_in=True,  # Display in counts
    # show_out=True,  # Display out counts
    # line_width=2,  # Adjust the line width for bounding boxes and text display
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = counter(im0)

    # print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

'''


from flask import Flask, render_template, request
import cv2
from flask_socketio import SocketIO
from ultralytics import solutions
from threading import Lock
from datetime import datetime

# Set up video capture and properties
cap = cv2.VideoCapture("Cars in Highway Traffic (FREE STOCK VIDEO).mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                         cv2.CAP_PROP_FRAME_HEIGHT,
                                         cv2.CAP_PROP_FPS))

# Define region points for rectangle region counting
region_points = [(20, 300), (1080, 300), (1080, 360), (20, 360)]

# Video writer for saving the processed output
video_writer = cv2.VideoWriter("object_counting_output.avi",
                               cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize ObjectCounter from ultralytics solutions
counter = solutions.ObjectCounter(
    show=True,           # Display the output if needed
    region=region_points,  # Pass region points
    model="yolo11n.pt",    # Use your YOLO model for object counting
)

def get_current_datetime():
    now = datetime.now()
    return now.strftime("%m/%d/%Y %H:%M:%S")

#backend to make sure the object counting is not active when the user first loads the dashboard.
counting_active = False


# Background task for object counting
def background_thread():

    global counting_active
    while cap.isOpened():
        if counting_active:
            success, im0 = cap.read()
            if not success:
                print("Video frame is empty or processing is complete.")
                break

            # Process the frame using the object counter
            results = counter(im0)
            
            # Extract the object (e.g., car) count from results.
            try:
                car_count = results.count
            except AttributeError:
                car_count = 0


            # Extract in and out counts properly
            cars_in = getattr(results, "in_count", 0)  # Use .in_count attributeq
            cars_out = getattr(results, "out_count", 0)  # Use .out_count attribute



            print(f"Cars IN: {cars_in}, Cars OUT: {cars_out}")

            totalCars = cars_in + (cars_out*-1)

            print(f"Total Cars: {totalCars}")# Display the total count of cars in the region


            video_writer.write(results.plot_im)  # write the processed frame.

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


            # Emit the count and current time to the frontend
            socketio.emit('updateSensorData', {'count': totalCars,"date": get_current_datetime()})
            
            # Write the annotated frame to the output video file
            video_writer.write(results.plot_im)
            
            # Sleep for a short time to simulate real-time processing based on FPS
            socketio.sleep(1.0 / fps)
        else:
            # Sleep for a short time if counting is not active
            socketio.sleep(1.0)

    # Clean up resources after processing is complete
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

# Set up Flask app and Socket.IO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'donsky!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for thread management
thread = None
thread_lock = Lock()

# Route for the main webpage
# This renders the "index.html" file when the user accesses the root URL ("/").
@app.route('/')
def index():
    return render_template('index.html')



# WebSocket event: When a client connects to the server
# This function ensures that the background object counting process starts
# only once, using a thread lock to prevent duplicate threads.
@socketio.on('connect')
def connect():
    global thread
    with thread_lock:
        if thread is None:  # Start the background thread only if it’s not running
            thread = socketio.start_background_task(background_thread)


# WebSocket event: When a client disconnects
# This function simply logs when a client disconnects from the WebSocket server.
@socketio.on('disconnect')
def disconnect():
    print('Client disconnected', request.sid)


# WebSocket event: When the user toggles the counting status
# This function toggles the counting_active variable to start or stop the object counting process. is connected to html start button
@socketio.on('toggle_counting')
def toggle_counting():
    global counting_active
    counting_active = not counting_active
    print(f"Counting active: {counting_active}")
    # Emit the current status back to the client
    socketio.emit('counting_status', {'active': counting_active})



#function will reset the count to 0 when the user clicks the "Reset Count" button in the dashboard.
# This function emits the updated count to the frontend.
@socketio.on('manualRefresh')
def manual_refresh():
    resetCount = 0
    socketio.emit('updateSensorData', {'count': resetCount,
                                           "date": get_current_datetime()})


#Start the Flask-SocketIO server
# This runs the Flask app and WebSocket server to handle real-time communication.
if __name__ == '__main__':
    socketio.run(app)

#TODO: Add the necessary HTML and JavaScript code to create the real-time dashboard.
# This will involve displaying the object count and current time in the dashboard.
# You can use the provided "index.html" file as a starting point for the dashboard layout.
#copilot really clutched up on summarizing TODOs great job!