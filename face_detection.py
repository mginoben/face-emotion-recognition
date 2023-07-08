import sys
import numpy as np
import cv2 as cv
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
IMG_SIZE = 100

def preprocess(img, output_size=IMG_SIZE, blur_ksize=5):
    
    # resize
    img = cv.resize(img, (output_size, output_size))
    
    # blur
    blur_img = cv.GaussianBlur(img, (blur_ksize,blur_ksize), 0)
    
    # contrast
    contrast = 10
    contrast_img = np.int16(blur_img)
    contrast_img = contrast_img * (contrast/127+1) - contrast
    contrast_img = np.clip(contrast_img, 0, 255)
    contrast_img = np.uint8(contrast_img)
    
    # apply sobel derivatives
    grad_x = cv.Sobel(contrast_img, cv.CV_64F, 1, 0)
    grad_y = cv.Sobel(contrast_img, cv.CV_64F, 0, 1)

    # add square and take squareroot 
    grad = np.sqrt(grad_x**2 + grad_y**2)

    # normalize to range 0 to 255 and clip negatives
    output = (grad * 255 / grad.max()).astype(np.uint8)
    
    return output

def build_model():

    num_classes = 5
    model = Sequential(name='Proposed_Model')

    model.add(
        Conv2D(
            filters=32,
            kernel_size=(5,5),
            input_shape=(IMG_SIZE, IMG_SIZE, 1),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
        )
    )
    model.add(BatchNormalization())

    model.add(
        Conv2D(
            filters=32,
            kernel_size=(5,5),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal'
        )
    )
    model.add(BatchNormalization())

    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal'
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
        )
    )
    model.add(BatchNormalization())
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.4))

    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
        )
    )
    model.add(BatchNormalization())

    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3,3),
            activation='elu',
            padding='same',
            kernel_initializer='he_normal',
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.6))

    model.add(Flatten())
        
    model.add(
        Dense(
            128,
            activation='elu',
            kernel_initializer='he_normal',
        )
    )
    model.add(BatchNormalization())
    
    model.add(Dropout(0.25))
    
    model.add(
        Dense(
            num_classes,
            activation='softmax',
        )
    )
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0005),
        metrics=['accuracy']
    )
    
    return model

def run():

    emotions = ["angry", "happy", "neutral", "sad", "surprise"]

    cap = cv.VideoCapture(0)

    model = build_model()
    model.load_weights('proposed_model.h5')

    while True:
        # Capture frame by frame
        _, frame = cap.read()

        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect face
        detected_faces = haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=9)

        # If face detected
        if len(detected_faces) > 0:
            for (x, y, w, h) in detected_faces:
                
                # Cropped face
                offset = 5
                face_image = gray_frame[y - offset:y + h + offset, x - offset:x + w + offset]

                # Predict
                preprocessed_img = preprocess(face_image, IMG_SIZE)

                cv.imshow('Features', cv.resize(preprocessed_img, (400, 400)))

                reshaped_img = np.array(preprocessed_img).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
                rescaled_img = reshaped_img/255.0
                predicted = model.predict(rescaled_img)[0].argmax()
                print(emotions[predicted])

                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
                cv.putText(frame, emotions[predicted], (x, y - 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

        # Display the frame
        cv.imshow('Frame', frame)

        if cv.waitKey(20) & 0xFF == ord('q'):
            break


    # If match detected, release the capture
    cap.release()
    cv.destroyAllWindows()
    sys.exit()

run()
