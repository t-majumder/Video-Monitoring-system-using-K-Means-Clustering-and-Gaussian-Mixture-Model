#Name: Tamal Majumder
#Enrollment No: 2023PHS7226
#Email: 987tamal@gmail.com

import numpy as np
import cv2

#__________________________________________________________________________________________
#__________________________________________________________________________________________

#____________________K - MEANS____________________#

#Function for K means which yields the Mean, varience and Responsibility

def K_means(frame, gaussians):

    rows, cols, channels = frame.shape
    points = rows * cols
    r = np.zeros((points, gaussians, channels))

    # Initialization of mean values
    u = [[00, 00, 00], [130, 130, 130], [230, 230, 230]]
    b = [[00, 00, 00], [130, 130, 130], [230, 230, 230]]
    itr = 0

    # Computing until the means converges
    while (1):
        # clustering each pixel of the image to its nearest mean
        for z in range(0, channels):
            for k in range(0, rows):
                for i in range(0, cols):
                    a = frame[k][i][z]
                    min = (a - u[0][z]) ** 2
                    r[i + cols * k][0][z] = 0
                    id = 0
                    for j in range(1, gaussians):
                        c = (a - u[j][z]) ** 2
                        if c < min:
                            min = c
                            id = j
                        r[i + cols * k][j][z] = 0
                    r[i + cols * k][id][z] = 1
        p = np.zeros((gaussians, channels))

        # Gives the mean of the new clusters
        for z in range(0, channels):
            for j in range(0, gaussians):
                p[j][z] = 1
                for k in range(0, rows):
                    for i in range(0, cols):
                        u[j][z] = u[j][z] + frame[k][i][z] * r[i + cols * k][j][z]
                        p[j][z] = p[j][z] + r[i + cols * k][j][z]
                u[j][z] = u[j][z] / p[j][z]

        # Checking the cluster mean convergence
        sum = 0
        for z in range(0, channels):
            for j in range(0, gaussians):
                sum = sum + (b[j][z] - u[j][z]) ** 2
                b[j][z] = u[j][z]
        if sum < 100:
            break
        itr += 1

    # Calculate the Variances of the new clusters around the means
    si = np.zeros((gaussians))
    for j in range(0, gaussians):
        p[j][0] = 0
        for k in range(0, rows):
            for i in range(0, cols):
                si[j] = si[j] + (frame[k][i][0] - u[j][0]) ** 2 * r[i + cols * k][j][0]
                p[j][0] = p[j][0] + r[i + cols * k][j][0]

        si[j] = si[j] / p[j][0]

    return u, r, si

#__________________________________________________________________________________________
#__________________________________________________________________________________________

#______________________Gaussian Mixture Model______________________#

#Function for the Gaussian Mixture model which will yield the foreground and background frames

def GMM(frame, sigma, pi, mean, background, foreground, gaussians, alpha, rho):
    # get the dimensions of the frame
    rows, cols, channels = frame.shape
    rat = [0 for i in range(gaussians)]

    # For each pixel in the frame compute new mean and variance
    for z in range(0, channels):
        for k in range(0, rows):
            for i in range(0, cols):
                flag = 0
                temp = 0
                for j in range(0, gaussians):
                    if abs(frame[k][i][z] - mean[i + cols * k][j][z]) < (2.5 * (sigma[i + cols * k][j]) ** (1 / 2.0)):
                        mean[i + cols * k][j][z] = (1 - rho) * mean[i + cols * k][j][z] + rho * frame[k][i][z]
                        sigma[i + cols * k][j] = (1 - rho) * sigma[i + cols * k][j] + rho * (frame[k][i][z] - mean[i + cols * k][j][z]) ** 2
                        pi[i + cols * k][j][z] = (1 - alpha) * pi[i + cols * k][j][z] + alpha
                        flag = 1
                    else:
                        pi[i + cols * k][j][z] = (1 - alpha) * pi[i + cols * k][j][z]

                    temp = temp + pi[i + cols * k][j][z]

                # Normalizing weights and find the corresponding pi/sig values
                for j in range(0, gaussians):
                    pi[i + cols * k][j][z] = pi[i + cols * k][j][z] / temp
                    rat[j] = pi[i + cols * k][j][z] / sigma[i + cols * k][j]

                # Arranging the mean, variance, and weights in decreasing order as per the ratio pi/sig
                for j in range(0, gaussians):
                    swapped = False
                    for x in range(0, gaussians - j - 1):
                        if rat[x] < rat[x + 1]:
                            rat[x], rat[x + 1] = rat[x + 1], rat[x]
                            pi[i + cols * k][x][z], pi[i + cols * k][x + 1][z]= pi[i + cols * k][x + 1][z], pi[i + cols * k][x][z]
                            mean[i + cols * k][x][z], mean[i + cols * k][x + 1][z] = mean[i + cols * k][x + 1][z], mean[i + cols * k][x][z]
                            sigma[i + cols * k][x], sigma[i + cols * k][x + 1]= sigma[i + cols * k][x + 1], sigma[i + cols * k][x]
                            swapped = True
                    if swapped == False:
                        break

                # If the current pixel does not belong to any gaussian then updating the one with least weightage 
                if flag == 0:
                    mean[i + cols * k][gaussians - 1][z] = frame[k][i][z]
                    sigma[i + cols * k][gaussians - 1] = 10000

                #background or foreground pixel check
                b = 0
                B = 0
                for j in range(0, gaussians):
                    b = b + pi[i + cols * k][j][z]
                    if b > 0.9:
                        B = j
                        break

                # Update the value of foreground and background pixel
                for j in range(0, B + 1):
                    if flag == 0 or abs(frame[k][i][z] - mean[i + cols * k][j][z]) > (2.5 * (sigma[i + cols * k][j]) ** (1 / 2.0)):
                        foreground[k][i][z] = frame[k][i][z]
                        background[k][i][z] = mean[i + cols * k][j][z]
                        break
                    else:
                        foreground[k][i][z] = 255
                        background[k][i][z] = frame[k][i][z]
                        
    # Making the background pixels of the foreground frame white
    for z in range(0, channels):
        for k in range(0, rows):
            for i in range(0, cols):
                if foreground[k][i][z] == 255:
                    foreground[k][i][0] = foreground[k][i][1] = foreground[k][i][2] = 255
    foreground = cv2.medianBlur(foreground, 3)
    return background, foreground


#__________________________________________________________________________________________
#__________________________________________________________________________________________

##______________________THE MAIN PROGRAM______________________##

if __name__ == "__main__":

    # ::  P A R A M E T E R S :: 
    gaussians = 3
    alpha = 0.03
    ro = 0.1

    # Read the input video
    cap = cv2.VideoCapture('D:\\ML Assignment 1 (Bg subs)\\umcp.mpg')

    if not cap.isOpened():
        print("Error opening the video file")
    else:
        print("Video imported successfully")

    # Taking 1st frame of the video
    ret, frame = cap.read()
    rows, cols, channels = frame.shape
    points = rows * cols

    # Applying kmeans on the first frame
    u, r, si = K_means(frame, gaussians)

    # calculate the parameters for the gaussians
    sig = np.zeros((points, gaussians))
    pi = np.zeros((points, gaussians, channels))
    mean = np.zeros((points, gaussians, channels))
    back = np.zeros((rows, cols, channels), dtype=np.uint8)
    fore = np.zeros((rows, cols, channels), dtype=np.uint8)

    for z in range(0, channels):
        for j in range(0, gaussians):
            for k in range(0, rows):
                for i in range(0, cols):
                    mean[i + cols * k][j][z] = u[j][z]
                    sig[i + cols * k][j] = si[j]
                    pi[i + cols * k][j][z] = (1 / gaussians) * (1 - alpha) + alpha * r[i + cols * k][j][z]

    count = 0

    # Updation of gaussian mixture model for each pixel in the new frame
    while ret:
        # take each frame from the video
        ret, frame = cap.read()

        # Performing of background and foreground detection on the frame using GMM and save it in back and fore
        back, fore = GMM(frame, sig, pi, mean, back, fore, gaussians, alpha, ro)

        # save the foreground and background frames
        cv2.imwrite(r"D:\ML Assignment 1 (Bg subs)\frames fore\fore%d.jpg" % count, fore)
        cv2.imwrite(r"D:\ML Assignment 1 (Bg subs)\frames back\back%d.jpg" % count, back)

        count += 1

    cap.release()
    cv2.destroyAllWindows()
#__________________________________________________________________________________________
#__________________________________________________________________________________________
    
#_______________________MAKING THE VIDEO_______________________#

fourcc = cv2.VideoWriter_fourcc(*'XVID') #video codac format
fps = 24  # Framerate
frame_size = None  

# VideoWriter objects
output_video_back = None
output_video_fore = None

# Loop through the images and write to the video
for i in range(0, 998):
    # for background
    img1 = cv2.imread(f"D:\\ML Assignment 1 (Bg subs)\\frames back\\back{i}.jpg")
    # for foreground
    img2 = cv2.imread(f"D:\\ML Assignment 1 (Bg subs)\\frames fore\\fore{i}.jpg")
    
    # Determinea frame size from the first image
    if frame_size is None:
        height, width, _ = img1.shape
        frame_size = (width, height)
        output_video_back = cv2.VideoWriter('output_video_back.avi', fourcc, fps, frame_size)
        output_video_fore = cv2.VideoWriter('output_video_fore.avi', fourcc, fps, frame_size)

    # Writing frames to videos
    output_video_back.write(img1)
    output_video_fore.write(img2)

# Release the VideoWriter objects
output_video_back.release()
output_video_fore.release()

