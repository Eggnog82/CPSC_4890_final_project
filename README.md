robot_ip=192.168.1.216

Cmds to run:
1. python scripts/gello_get_offset.py \
    --start-joints 0 -0.4 0 0.62 0.005 1.04 0.009 \
    --joint-signs 1 1 1 1 1 1 1 \
    --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT9MG69Y-if00-port0

2. python experiments/launch_nodes.py --robot xarm --robot_ip 192.168.1.216
3. python experiments/run_env.py --agent=gello --use-save-interface --data_dir "/home/ericli/Desktop/CPSC_4890_final_project/demos/with_img_2" --wrist_camera_port 5000 --base_camera_port 5001 --robot_port 6001