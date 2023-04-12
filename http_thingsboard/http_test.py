import requests
import time
import random



url="http(s)://sensorweb.us:9090/api/v1/keith1/telemetry" ###################################




while True:
    data={"HeartRate":random.randint(45,90), "Temperature":random.uniform(90,103)}
    print(data)
    #response=requests.post(url,json=data)
    time.sleep(1)