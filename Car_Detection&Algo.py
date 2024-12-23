# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:52:23 2024

@author: DELL
"""

### ----------------Smart Traffic Management System----------------- ###
#Car and Pedestrian Tracking using OpenCV Module

import cv2
import time
#from random import randrange


def car_tracking(img_file):
    #Our Image
    #img_file = image_path  #It is a simple string storing file name

    #Our pre-trained car classifier
    trained_car_detector = cv2.CascadeClassifier('car_detector.xml') 

    #Create OpenCV Image
    img = cv2.imread(img_file) 

    #Convert Image to grayscale
    grayscaled_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    #Detect cars in image and get it's coordinates
    car_coordinates = trained_car_detector.detectMultiScale(grayscaled_image)
    #print ("Number of Vehicles: ",len(car_coordinates))
    #print("Number of cars detected: ", (car_coordinates / car_coordinates[0]))

    #print(car_coordinates)
    '''
    for coordinate in car_coordinates:
        (x, y, w, h) = coordinate
        cv2.rectangle(img, (x, y), (x+w, y+h),  (0, 0, 256), 3)
    '''
    
    #Display the Image
    #cv2.imshow('Car Detector Window', img) 

    #To display the image for a long period of time
    #cv2.waitKey()

    #To close the windows created by OpenCV
    #cv2.destroyAllWindows()
    return len(car_coordinates)


'''
def car_tracking_realtime():
    webcam1 = cv2.VideoCapture(0)
    webcam2 = cv2.VideoCapture(1)
    webcam3 = cv2.VideoCapture(3)
    webcam4 = cv2.VideoCapture(4)
    trained_car_detector = cv2.CascadeClassifier('car_detector.xml') 


    while True:
        successful_frame_read, frame = webcam.read() #Read each frame from webcam stream
        grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert each frame to grayscale
        car_coordinates = trained_car_detector.detectMultiScale(grayscaled_img) #Get coordinates of faces from each grayscaled frame
        
        for coordinate in face_coordinates:
            #print(coordinate)
            (x, y, w, h) = coordinate
            cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(128,256), randrange(128,256), randrange(128,256)), 2)
        cv2.imshow("Clever Programmer Face Detector", frame)
        key = cv2.waitKey(1)
        
        if key == 81 or key == 113: #Press Q to quit. ASCII values of Q & q is 81 & 113 respec.
            break
'''

def get_user_input():
    """
    Get the vehicle counts for each lane from user input.
    The number of lanes is fixed at 4.
    """
    try:
        num_lanes = 4  # Fixed number of lanes
        print(f"Number of lanes given is: {num_lanes}")
        
        vehicle_counts = {}
        for i in range(num_lanes):
            lane = chr(ord('A') + i)
            count = int(input(f"Enter the number of vehicles in lane {lane} (e.g., 0): "))
            if count < 0:
                raise ValueError(f"Number of vehicles in lane {lane} cannot be negative.")
            vehicle_counts[lane] = count
        
        return vehicle_counts

    except ValueError as e:
        print(f"Invalid input: {e}")
        return None

def display_traffic_lights(lane_priority, vehicle_counts):
    """
    Display the status of the traffic lights for each lane based on priority.
    """
    for lane in lane_priority:
        if lane == lane_priority[0]:  # The first lane in the list gets the green light
            print(f"Lane {lane}: Green Light - GO")
            # Calculate the sleep time based on vehicle count
            if vehicle_counts[lane] <= 15:
                sleep_time = 15
            elif vehicle_counts[lane] >= 60:
                sleep_time = 60
            else:
                sleep_time = vehicle_counts[lane]
            print(f"Green light duration for lane {lane}: {sleep_time} seconds")
            time.sleep(sleep_time)
        else:
            print(f"Lane {lane}: Red Light - STOP")

def ai_traffic_signal(vehicle_counts):
    """
    Run the traffic signal simulation based on user input.
    """
    while True:
        # Get vehicle counts from the user
        #vehicle_counts = get_user_input()
        if vehicle_counts is None:
            continue

        #num_lanes = len(vehicle_counts)

        # Display the vehicle counts
        print(f"Vehicle Counts: {vehicle_counts}")

        # Sort lanes by vehicle count in descending order
        sorted_lanes = sorted(vehicle_counts.keys(), key=lambda lane: vehicle_counts[lane], reverse=True)
        
        # Display the traffic light status for each lane
        for lane in sorted_lanes:
            display_traffic_lights([lane] + [l for l in sorted_lanes if l != lane], vehicle_counts)
            print("\n" + "-"*40 + "\n")



# Run the AI-driven traffic signal simulation
# MAIN Starts here
# Below we are supposed to provide a continuous stream of live traffic footage

number_of_cars_direction1 = car_tracking("Car5.jpeg")
number_of_cars_direction2 = car_tracking("Car4.jpg")
number_of_cars_direction3 = car_tracking("Car2.jpeg")
number_of_cars_direction4 = car_tracking("Car_Image.jpeg")

Array_Vehicle_Count = {1:number_of_cars_direction1, 2 : number_of_cars_direction2, 3:number_of_cars_direction3, 4:number_of_cars_direction4} 

ai_traffic_signal(Array_Vehicle_Count)

    

