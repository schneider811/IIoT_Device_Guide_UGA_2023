import requests
import time
import random

heart=random.uniform()

url="http(s)://sensorweb.us:9090/api/v1/9ZhNdW7Wu693dOtVbRWi/telemetry"

data={"HeartRate":"58", "Temperature":"98.6"}


while True:
    response=requests.post(url,json=data)
    time.sleep(5)