robot_ip=192.168.1.216

Cmds to run:
1. python scripts/gello_get_offset.py \
    --start-joints -0.005235987755982988 -0.778416846389471 -0.010471975511965976 0.36826447217080355 0.006981317007977318 1.1763519158441782 0.012217304763960306 \
    --joint-signs 1 1 1 1 1 1 1 \
    --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT9MG69Y-if00-port0



2. python experiments/launch_nodes.py --robot xarm --robot_ip 192.168.1.216
3. python experiments/run_env.py --agent=gello --use-save-interface --data_dir "/home/ericli/Desktop/CPSC_4890_final_project/demos/with_img_2" --wrist_camera_port 5000 --base_camera_port 5001 --robot_port 6001

# Data conversion
1. python demo_to_npy_from_ulas.py ./demos/with_img_4/gello

# Training
1. python -m scripts.bc --mode train --data asset/bc_train.npz
2. python -m scripts.bc --mode inference --data asset/bc_train.npz --ip 192.168.1.216

# Results

1. 9 demos, no image data input
- Epoch 1000 | Train MSE (norm): 0.423259 | Test MSE (norm): 26.354415