#Integration between the workflow and Thingsboard.

In order to get this workflow to run there are only a few lines of code that need to be changed. If you are trying to get the workflow to run on the sensorweb server you only need to edit line 44 of all_together.py before you run the workflow.

If you are trying to build an image and connect to a different thingsboard server lines 33, 40, 42 and 44 will need to be changed with the applicable information for your server.

This all_together.yml workflow works by creating a service on the Pi image that runes all_together_initialize.service. That will create a crontab using the contents of file all_together_init.sh. Once that crontab is created the device will constantly check if the built file is running and if the file stopped running for whatever reason will restart the file. 

Once again to see all of this in action please watch the suplemental Youtube videos at: 

https://youtu.be/oG-rT_H-thw

https://youtu.be/2cG1O7QNw_Q
