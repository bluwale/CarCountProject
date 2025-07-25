
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

