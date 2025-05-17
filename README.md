## Face Recognition Based Attendance System

This is a web-based attendance system that utilizes face recognition technology to automate the process of tracking attendance. Built with Flask, OpenCV, and a K-Nearest Neighbors (KNN) machine learning model, it identifies registered users and records their attendance. Administrators can easily add new users by capturing images of their faces. The system also features SMS notification upon successful recognition and a web interface to view today's attendance and manage user additions.

## Key Features

* **Automated Attendance:** Records attendance by recognizing the faces of registered users.
* **Face Recognition:** Employs OpenCV for image processing and a KNN model for accurate face identification.
* **User Management:** Administrators can add new users by capturing multiple images of their faces.
* **Attendance Logging:** Maintains a record of attendance in a CSV file, including names and timestamps.
* **SMS Notification:** Sends an SMS notification to the recognized user's registered phone number upon successful attendance marking.
* **Web Interface:** Provides a user-friendly web interface built with Flask to:
    * View today's attendance records.
    * Add new users to the system.

## Technologies Used

* **Backend Framework:** Flask (Python) - For building the web application and handling server-side logic.
* **Image Processing:** OpenCV (Python) - For capturing and processing facial images.
* **Machine Learning:** K-Nearest Neighbors (KNN) - For training a face recognition model.
* **SMS Integration:** [Specify the SMS API or library used, e.g., Twilio, Nexmo] (Python) - For sending SMS notifications.
* **Data Storage:** CSV file - For storing attendance logs.

## How it Works

1.  **User Registration:** Administrators use the web interface to add new users. This involves capturing multiple images of the user's face from different angles and storing these images along with the user's name and phone number. These images are used to train the KNN face recognition model.
2.  **Attendance Marking:** When a user's face is detected by the system (e.g., via a webcam connected to the server), OpenCV processes the captured image.
3.  **Face Recognition:** The processed facial image is then compared against the trained KNN model. If a match is found with a sufficient level of confidence, the user is recognized.
4.  **Attendance Logging:** Upon successful recognition, the user's name and the current timestamp are recorded in the attendance CSV file.
5.  **SMS Notification:** An SMS notification is sent to the recognized user's registered phone number confirming their attendance.
6.  **Web Interface Display:** The web interface displays the attendance records for the current day, fetched from the attendance CSV file.

## Getting Started (For Developers)

1.  **Prerequisites:**
    * Python 3.x installed.
    * pip (Python package installer) installed.
    * Webcam connected to the system running the backend.
    * Account with an SMS API provider (if SMS functionality is to be used).

2.  **Clone the Repository:**
    ```bash
    git clone [YOUR_REPOSITORY_URL]
    cd face-recognition-attendance-system
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure SMS Integration (Optional):**
    * If you intend to use SMS notifications, you will need to sign up for an account with an SMS API provider (e.g., Twilio, Nexmo).
    * Obtain your API credentials (API key, secret, phone number, etc.).
    * Update the relevant configuration settings in your Flask application (e.g., in a configuration file or environment variables).

5.  **Train the Face Recognition Model:**
    * Ensure you have a directory containing images of registered users. The directory structure should ideally have subdirectories for each user, with multiple images of their face inside.
    * Run the script responsible for training the KNN model using these images. This script will typically extract facial features from the images and train the model.

6.  **Run the Flask Application:**
    ```bash
    python app.py
    ```

7.  **Access the Web Interface:** Open your web browser and navigate to the address where the Flask application is running (usually `http://127.0.0.1:5000/`).

## Potential Future Enhancements

* **Real-time Attendance Monitoring:** Display a live feed of recognized faces and attendance marking.
* **Database Integration:** Instead of a CSV file, use a more robust database (e.g., SQLite, MySQL, PostgreSQL) for storing attendance data and user information.
* **User Authentication and Authorization:** Implement user roles and authentication to secure the web interface.
* **Reporting and Analytics:** Generate attendance reports for specific periods.
* **Integration with Existing Systems:** Explore integration with student management systems or HR systems.
* **Improved Face Detection and Recognition Accuracy:** Experiment with more advanced face detection algorithms and potentially other machine learning models.
* **Cloud Deployment:** Deploy the application to a cloud platform for wider accessibility.

## Contributing

[Add information about how others can contribute to your project if you plan to open-source it.]

## Contact

cskoushik koushikcs562@gmail.com

