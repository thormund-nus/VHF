.PHONY: init
init:
	g++ set_device_mode.cpp -O3 -o set_device_mode
	sudo chown root:root set_device_mode
	sudo chmod +s set_device_mode
	ln -sf /dev/usbhybrid0 vhf_board.softlink
	ln -sf /home/qitlab/programs/usbhybrid/apps/teststream teststream.exec
	mkdir Log
	mkdir Data
