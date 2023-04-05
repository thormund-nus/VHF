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

void print_help(char* prg) {
    printf("Invalid arguments specified!\n\n");
    printf("%s <command> <device path> <mode>\n", prg);
    printf("command: set reload\n");
    printf("Example:\n");
    printf("\t%s set /sys/bus/usb/devices/1-6.1/bConfigurationValue 1\n", prg);
    printf("\t%s reload\n", prg);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_help(argv[0]);
        return -1;
    }

    char header[] = "/sys/bus/usb/devices/";
    std::string command = argv[1];
    if (command == "set") {
        if (argc < 4) {
            print_help(argv[0]);
            return -1;
        }
        
        std::string devicePath = argv[2];
        strncmp(devicePath.c_str(), header, sizeof(header));

        char mode = argv[3][0];
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
    else if (command == "reload") {
        setuid(0);
        system("udevadm control --reload-rules");
        system("udevadm trigger");
        return 0;
    }
    else {
        printf("Invalid command specified!\ncommand must be one of: set, reload\n");
        return -1;
    }
}