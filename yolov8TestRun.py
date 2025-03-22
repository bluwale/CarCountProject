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

#global vars to keep track of count offset with refresh button
count_offset = 0
last_totalCars = 0


# Background task for object counting
def background_thread():
    global counting_active, count_offset, last_totalCars

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

            rawTotalCars = cars_in + (cars_out*-1)

            print(f"Total Cars: {rawTotalCars}")# Display the total count of cars in the region

            
            
            last_totalCars = rawTotalCars 

            display_count = rawTotalCars - count_offset
            print(f"Display Count: {display_count}")


            video_writer.write(results.plot_im)  # write the processed frame.

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


            # Emit the count and current time to the frontend
            socketio.emit('updateSensorData', {'count': display_count,"date": get_current_datetime()})
            
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
    global count_offset, last_totalCars
    # Set the current total as the new offset
    count_offset = last_totalCars
    print(f"Count reset! New offset: {count_offset}")
    # Immediately send update to client to show zero
    socketio.emit('updateSensorData', {'count': 0, "date": get_current_datetime()})

#Start the Flask-SocketIO server
# This runs the Flask app and WebSocket server to handle real-time communication.
if __name__ == '__main__':
    socketio.run(app)

#copilot really clutched up on summarizing TODOs great job!

'''

from flask import Flask, render_template, request
import cv2
from flask_socketio import SocketIO
from ultralytics import solutions
from threading import Lock
from datetime import datetime
from queue import Queue
from threading import Lock
import threading


#setup of flask app
app = Flask(__name__)
app.config['Secret_KEY'] = 'donsky!'
socketio = SocketIO(app, cors_allowed_origins="*")

#global vars for managing thread and lock
thread = None
thread_lock = Lock()
counting_active = False


#counters and offesets of both cameras
count_offset1 = 0
count_offset2 = 0
last_totalCars1 = 0
last_totalCars2 = 0

#queue to store thread
countQueue = Queue()

def get_current_datetime():
    now = datetime.now()
    return now.strftime("%m/%d/%Y %H:%M:%S")

class CameraProcessor(threading.Thread):
    def __init__(self, camera_id, video_source, region_points, model = 'best.pt'):
        threading.Thread.__init__(self)

        self.daemon = True  # Thread will close when main program exits
        self.camera_id = camera_id
        self.video_source = video_source
        self.region_points = region_points
        self.model = model
        self.active = False
        self.cap = None
        self.video_writer = None
        self.counter = None
        self.cars_in = 0
        self.cars_out = 0
        self.total_cars = 0
        self.lock = threading.Lock()



    def setup(self):
        #video capture and properties
        self.cap = cv2.VideoCapture(self.video_source)
        assert self.cap.isOpened(), f"Error reading video file for camera {self.camera_id}"

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        # Initialize video writer
        self.video_writer = cv2.VideoWriter(
            f"object_counting_output_{self.camera_id}.avi",
            cv2.VideoWriter_fourcc(*"mp4v"), 
            fps, 
            (w, h)
        )

        # Initialize object counter
        self.counter = solutions.ObjectCounter(
            show=True,
            region=self.region_points,
            model=self.model,
        )

        return fps

    def run(self):
        fps = self.setup()
        frame_delay = 1.0 / fps if fps > 0 else 0.033 # 30 FPS default

        while True:
            if self.active and counting_active:
                if not self.cap.isOpened():
                    print(f"Camera {self.camera_id} is closed. Reopening...")
                    self.setup()
                
                success, frame = self.cap.read()
                if not success:
                    print(f"Failed to read frame from camera {self.camera_id}")
                    # Try to reopen or reset the camera
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.video_source)
                    continue
                
                # Process the frame
                results = self.counter(frame)
                
                # Extract counts
                with self.lock:
                    self.cars_in = getattr(results, "in_count", 0)
                    self.cars_out = getattr(results, "out_count", 0)
                    self.total_cars = self.cars_in + (self.cars_out*-1)
                
                # Write processed frame to video file
                self.video_writer.write(results.plot_im)
                
                # Send count data to the main thread via queue
                countQueue.put({
                    'camera_id': self.camera_id,
                    'cars_in': self.cars_in,
                    'cars_out': self.cars_out,
                    'total_cars': self.total_cars
                })
            
            # Sleep to match the original frame rate
            threading.Event().wait(frame_delay)


    def get_counts(self):
        with self.lock:
            return {
                'cars_in': self.cars_in,
                'cars_out': self.cars_out,
                'total_cars': self.total_cars
            }
    
    def set_active(self, active):
        self.active = active


    
    # Initialize camera processors
camera1 = CameraProcessor(
    camera_id=1,
    video_source="Cars Moving On Road Stock Footage - Free Download.mp4",  # Replace with your first video source
    region_points=[(20, 300), (1080, 300), (1080, 360), (20, 360)]
)

camera2 = CameraProcessor(
    camera_id=2,
    video_source="Cars in Highway Traffic (FREE STOCK VIDEO).mp4",  # Replace with your second video source
    region_points=[(20, 300), (1080, 300), (1080, 360), (20, 360)]
)

# Start camera threads
camera1.start()
camera2.start()

# Background task for aggregating results and updating the UI
def background_thread():
    global counting_active, count_offset1, count_offset2, last_totalCars1, last_totalCars2
    
    # Set cameras active
    camera1.set_active(True)
    camera2.set_active(True)
    
    while True:
        if counting_active:
            # Process all available counts in the queue without blocking
            counts1 = {'total_cars': 0}
            counts2 = {'total_cars': 0}
            
            # Get the latest counts from each camera
            while not countQueue.empty():
                data = countQueue.get()
                if data['camera_id'] == 1:
                    counts1 = data
                elif data['camera_id'] == 2:
                    counts2 = data
            
            # Get the counts directly from the camera objects if queue was empty
            if 'total_cars' not in counts1:
                counts1 = camera1.get_counts()
            if 'total_cars' not in counts2:
                counts2 = camera2.get_counts()
                
            # Update last known counts
            last_totalCars1 = counts1['total_cars']
            last_totalCars2 = counts2['total_cars']
            
            # Calculate display counts (with offsets)
            display_count1 = last_totalCars1 - count_offset1
            display_count2 = last_totalCars2 - count_offset2
            combined_count = display_count1 + display_count2
            
            # Log for debugging
            print(f"Camera 1: {display_count1}, Camera 2: {display_count2}, Combined: {combined_count}")
            
            # Emit data to frontend
            socketio.emit('updateSensorData', {
                'count1': display_count1,
                'count2': display_count2,
                'combined_count': combined_count,
                'date': get_current_datetime()
            })
        
        # Sleep to control update rate (10 updates per second)
        socketio.sleep(0.1)

# Route for the main webpage
@app.route('/')
def index():
    return render_template('index.html')

# WebSocket event: When a client connects
@socketio.on('connect')
def connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)

# WebSocket event: When a client disconnects
@socketio.on('disconnect')
def disconnect():
    print('Client disconnected', request.sid)

# WebSocket event: Toggle counting status
@socketio.on('toggle_counting')
def toggle_counting():
    global counting_active
    counting_active = not counting_active
    print(f"Counting active: {counting_active}")
    socketio.emit('counting_status', {'active': counting_active})

# WebSocket event: Manual refresh (reset counts)
@socketio.on('manualRefresh')
def manual_refresh():
    global count_offset1, count_offset2, last_totalCars1, last_totalCars2
    
    count_offset1 = last_totalCars1
    count_offset2 = last_totalCars2
    
    print(f"Count reset! New offsets: Camera 1: {count_offset1}, Camera 2: {count_offset2}")
    
    socketio.emit('updateSensorData', {
        'count1': 0,
        'count2': 0,
        'combined_count': 0,
        'date': get_current_datetime()
    })

# Start the Flask-SocketIO server
if __name__ == '__main__':
    socketio.run(app)        

