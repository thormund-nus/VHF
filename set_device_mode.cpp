#include <stdio.h>
#include <string>
#include <cstring>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

/*
    ./set_device_mode - Program to set configuration mode for VHF processor **without root**
    Made by Thormund Tay

    Compilation and setup instructions:
    1. g++ -o set_device_mode -Os set_device_mode.cpp
    2. sudo chown root set_device_mode
    3. sudo chmod u+s set_device_mode
*/
int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Invalid arguments specified!\n\n");
        printf("%s <device path> <mode>\n", argv[0]);
        printf("Example:\n\t%s /sys/bus/usb/devices/1-6.1/bConfigurationValue 1\n", argv[0]);
        return -1;
    }

    char header[] = "/sys/bus/usb/devices/";
    std::string devicePath = argv[1];
    strncmp(devicePath.c_str(), header, sizeof(header));

    char mode = argv[2][0];
    if (mode != '1' && mode != '2') {
        printf("Invalid mode specified!\nValid modes are: 1 or 2\n");
        return -1;
    }

    char cmd[100];
    setuid(0);
    sprintf(cmd, "echo '%c' > %s", mode, devicePath.c_str());
    // printf("%s\n", cmd); // Debug use
    system(cmd);
    return 0;
}