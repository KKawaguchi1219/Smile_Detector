import cv2

GST_STR = 'nvarguscamerasrc \
    ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=(fraction)30/1 \
    ! nvvidconv ! video/x-raw, width=(int)800, height=(int)600, format=(string)BGRx \
    ! videoconvert \
    ! appsink drop=true sync=false'
WINDOW_NAME = 'Camera'

def smile(img, rect):
    smile_man=cv2.imread("./smile_face.png", cv2.IMREAD_UNCHANGED)

    x1, y1, x2, y2 = rect
    w = x2 - x1
    h = y2 - y1
    print(f"smile!")
    img_face = cv2.resize(smile_man, (w, h))

    # permeabilize
    alpha_channel=smile_man[:,:,3]/255.0
    img_face=img_face[:,:,:3]
    alpha_channel_resized=cv2.resize(alpha_channel, (w, h))
    alpha_channel_resized=alpha_channel_resized[:,:,None]

    img[y1:y2, x1:x2]=img[y1:y2, x1:x2]*(1-alpha_channel_resized)+img_face*alpha_channel_resized
    #img2 = img.copy()
    #img2[y1:y2, x1:x2] = img_face
    return img
 

def main():
    cap = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER)
    face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    smile_cascade = cv2.CascadeClassifier("./haarcascade_smile.xml")


    if not cap.isOpened():
        print("カメラが正常ではありません")
        exit()

    while True:
        ret, frame = cap.read()
        if ret != True:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        print(f"face:{faces}")
        for (x,y,w,h) in faces:
            text_size=w/200

            cv2.rectangle(frame,(x,y),(x+w, y+h),(255, 0, 0),2) # blue
            cv2.putText(frame, text="face", org=(x, y+h), fontScale=text_size, fontFace=cv2.FONT_HERSHEY_DUPLEX, color=(0, 0, 255), thickness=1)
            roi_gray = gray[y:y+h, x:x+w] #Gray画像から，顔領域を切り出す．
            smiles= smile_cascade.detectMultiScale(roi_gray,scaleFactor= 1.2, minNeighbors=10, minSize=(20, 20))#笑顔識別
            if len(smiles) >0 :
                frame=smile(frame, [x,y,x+w,y+h])
                #for(sx,sy,sw,sh) in smiles:
                    #cv2.circle(frame,(int(x+sx+sw/2),int(y+sy+sh/2)),int(sw/2),(0, 0, 255),2)#red
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(10)
        if key == 27: # ESC 
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()