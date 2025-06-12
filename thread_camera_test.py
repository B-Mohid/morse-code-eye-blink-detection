import cv2
import threading
import time

# Try setting the video capture API preference
# Start with DSHOW, if it fails, try MSMF
cv2.CAP_PROP_API_PREFERENCE = cv2.CAP_DSHOW # First attempt with DSHOW

def camera_thread_function():
    print("Attempting to open camera from a separate thread...")
    cap = cv2.VideoCapture(0)
    time.sleep(0.5) # Give camera a moment to initialize

    if not cap.isOpened():
        print("Error: Camera could NOT be opened from thread.")
        # Try MSMF if DSHOW failed
        cv2.CAP_PROP_API_PREFERENCE = cv2.CAP_MSMF
        cap = cv2.VideoCapture(0)
        time.sleep(0.5)
        if not cap.isOpened():
            print("Error: Camera still could NOT be opened with MSMF from thread.")
            return

    print("Camera opened successfully from thread. Attempting to read frames...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame from thread. Stream might have ended.")
                break
            # Display the frame (this will open a new window)
            cv2.imshow('Threaded Camera Test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting camera thread...")
                break
            time.sleep(0.01) # Small delay to prevent 100% CPU usage
    except Exception as e:
        print(f"An error occurred in camera thread: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows destroyed from thread.")

if __name__ == '__main__':
    print("Main thread starting camera test...")
    camera_thread = threading.Thread(target=camera_thread_function)
    camera_thread.daemon = True # Allow main program to exit even if thread is running
    camera_thread.start()
    print("Camera thread started. Press 'q' in the OpenCV window to quit, or Enter in terminal.")

    # Keep the main thread alive so the daemon camera_thread can run
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Main thread exiting.")

