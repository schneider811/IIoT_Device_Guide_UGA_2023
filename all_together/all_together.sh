chmod +x /home/pi/IIoT_Device_Guide_UGA_2023/all_together/dist/all_together
crontab -l | { cat; echo "* * * * * /home/pi/IIoT_Device_Guide_UGA_2023/all_together/dist/all_together 2>&1"; } | crontab -
crontab -l | { cat; echo "@reboot /home/pi/IIoT_Device_Guide_UGA_2023/all_together/dist/all_together 2>&1"; } | crontab -