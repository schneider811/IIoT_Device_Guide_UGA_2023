This is a quick guide to how to use this repository to build a raspberry pi image and have it run a service/script on RPi startup. There are several methods you can use to run a program on startup on a raspberry pi. Unfortunately, with this implementation of building the image automatically through github actions the only method I was able to get to work as we wanted it to was through the SYSTEMD method. (rc.local, bashrc, init.d and crontab are all other methods that either did not work as we wanted it to or did not work at all entirely)

This is a fairly simple process. In this folder there is a gpio_test.py program, this will be our service we want to run on start up. This program just blinks an LED hooked up to GPIO pin 23 on a RPi. The light is in serial with a 330 ohm resistor to ensure we do not burn out the LED. Once the circuit is set up you can also try this example. In the below diagram the blue wire is connected to ground and the red wire is connected to whatever GPIO pen you specify (23 in the provided code). 

<p float="left">
  <img src="/GPIO_diagram.png" width="100" />
</p>

The next step to understanding how to run this is to look at the gpio_test.service file. This file is the format required to create a service on a raspberry pi. This files defines a new service called My Script Service and defines that this script should run after the multi-user environment is available (on start-up once everything is fully booted and one can SSH into the device). The 'ExecStart' parameter is what tells the RPi what program to run once the startup conditions have been met. Since it is a python file you must specify your python path - this is the same as when you write 'python' on your computer before executing a python program from the command line. Then you simply write the full path to the program you wish to have execute on startup.

Fortunately thanks to github actions all of this can be implemented in your RPi operating system before you even boot to it! If you go to the file located at \.github\workflows\startup_service.yml . This github actions yml file is doing the few necessary steps to copy the gpio_test.service file to the systemd folder, changing the permissions of the python script so it can execute and enabling the service, below are the commands the script is running on the RPi. 

```yaml
        commands: |
          cd IIoT_Device_Guide_UGA_2023/
          sudo cp /home/pi/IIoT_Device_Guide_UGA_2023/startup/gpio_test.service /lib/systemd/system/gpio_test.service
          sudo chmod 644 /lib/systemd/system/gpio_test.service
          sudo systemctl enable gpio_test.service
```

All you have to do is download the image that action creates, load it into your raspberry pi and it'll begin running once you boot it up!