# for plane seg
cd point_seg/
python3 -m point_seg.seg

# for start usb cam 
cd publish_image/
python3 retrieve_image_usb.py

# for colorization:
cd colorization/
python3 colorize_pointcloud.py 