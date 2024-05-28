import cv2

# Set the video file name, fourcc code, fps, and frame size
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 24  # Frames per second
frame_size = None  # Frame size will be determined from the first image

# Create VideoWriter objects
output_video_back = None
output_video_fore = None

# Loop through the images and write to the video
for i in range(0, 998):
    # for background
    img1 = cv2.imread(f"D:\\ML Assignment 1 (Bg subs)\\frames back\\back{i}.jpg")
    # for foreground
    img2 = cv2.imread(f"D:\\ML Assignment 1 (Bg subs)\\frames fore\\fore{i}.jpg")
    
    # Determine frame size from the first image
    if frame_size is None:
        height, width, _ = img1.shape
        frame_size = (width, height)
        output_video_back = cv2.VideoWriter('output_video_back.avi', fourcc, fps, frame_size)
        output_video_fore = cv2.VideoWriter('output_video_fore.avi', fourcc, fps, frame_size)

    # Write frames to videos
    output_video_back.write(img1)
    output_video_fore.write(img2)

# Release the VideoWriter objects
output_video_back.release()
output_video_fore.release()
