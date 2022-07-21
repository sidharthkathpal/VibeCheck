import streamlit as st
import pandas as pd
import hashlib
import cv2
from fer import FER
import facial_detection as fd
import facial_verification as fv
import altair as alt
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
from numpy import dot
from numpy.linalg import norm
import csv
from datetime import datetime

csv_columns = ['username', 'day', 'angry','disgust','fear', 'happy', 'sad', 'surprise', 'neutral']
csv_file = "user_data.csv"
user_data_dict = []
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False


# DB Management
import sqlite3

conn = sqlite3.connect('data.db')
c = conn.cursor()

# DB  Functions


def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username, password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)', (username, password))
    conn.commit()

def login_user(username, password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?', (username, password))
    data = c.fetchall()
    return data
    # global userstable
    # if userstable[username][0] == password:
    #     return username, password


def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data


class sentiment_analysis():
    def __init__(self, img):
        self.img = img

    def execute(self):
        emotion_detector = FER(mtcnn=True)
        result = emotion_detector.detect_emotions(self.img)
        bounding_box = result[0]["box"]
        emotions = result[0]["emotions"]
        cv2.rectangle(self.img, (
            bounding_box[0], bounding_box[1]), (
                          bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (0, 155, 255), 2, )
        emotion_name, score = emotion_detector.top_emotion(self.img)
        for index, (emotion_name, score) in enumerate(emotions.items()):
            color = (211, 211, 211) if score < 0.01 else (255, 0, 0)
            emotion_score = "{}: {}".format(emotion_name, "{:.2f}".format(score))

            cv2.putText(self.img, emotion_score,
                        (bounding_box[0], bounding_box[1] + bounding_box[3] + 30 + index * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA, )
        return emotions


class main_func():
    def __init__(self) -> None:
        self.fd_obj = fd.Detection()
        self.fv_obj = fv.Verification()
        self.user_table = {}
    def compute_similarity(self, cropped_img_vector, profile_img_vector):
        cos_sim = dot(cropped_img_vector, profile_img_vector) / (norm(cropped_img_vector) * norm(profile_img_vector))
        return cos_sim

    def execute(self):

        st.title("VibeCheckStation")

        menu = ["Home", "Login", "SignUp"]
        choice = st.sidebar.selectbox("Menu", menu)

        # emotions_over_the_week = {username:{}}

        if choice == "Home":
            st.subheader("Who wants a vibe check?")
            picture = st.camera_input("First, take a picture...")

            if picture:
                with open('test.jpg', 'wb') as file:
                    file.write(picture.getbuffer())
                    values = self.fd_obj.detect("test.jpg")
                    #st.text(values.shape)

                    #temp_vector_values = self.fv_obj.verify(values)
                    for i in range(values.shape[0]):
                        cv2.imwrite("test.jpg", values[i])
                        #temp_vector = temp_vector_values[i]
                        #print(temp_vector)
                        img = cv2.imread("test.jpg")
                        #self.func(temp_vector)
                        senti_obj = sentiment_analysis(img)
                        emotions = senti_obj.execute()
                        st.image(senti_obj.img, channels="RGB")
                        st.text(emotions)

        elif choice == "Login":
            st.subheader("Welcome to your customised Vibe tracker")

            username = st.sidebar.text_input("User Name")
            password = st.sidebar.text_input("Password", type='password')

            # print(picture)
            if st.sidebar.checkbox("Login"):
                # create_usertable() ????
                hashed_pswd = make_hashes(password)

                result = login_user(username, check_hashes(password, hashed_pswd))
                if result:

                    st.success("Hello {}".format(username))

                    task = st.selectbox("Task", ["Click Photo", "Weekly Overview", "Happiness vs Sadness Tracker"])
                    df = pd.read_csv("user.csv")
                    if task == "Click Photo":
                        st.subheader("Click Your Picture")
                        picture = st.camera_input("First, take a picture...")

                        with open("user_table.pickle", "rb") as file:
                            loaded_dict = pickle.load(file)

                        profile_picture_vec = loaded_dict[username][1]

                        if picture:
                            with open('test.jpg', 'wb') as file:
                                file.write(picture.getbuffer())
                                values = self.fd_obj.detect("test.jpg")
                                temp_vector_values = self.fv_obj.verify(values)
                                # temp_vector_values = temp_vector_values.reshape(224, 224, 3)
                                arr4d = np.expand_dims(profile_picture_vec, 0)
                                profile_values = self.fv_obj.verify(arr4d)
                                similarity_values = []
                                all_img_emotions = []
                                for i in range(values.shape[0]):
                                    # cv2.imwrite("test.jpg", values[i])
                                    temp_vector = temp_vector_values[i]
                                    # st.text(temp_vector_values[i].shape())
                                    # st.text(profile_values.shape())
                                    img = cv2.imread("test.jpg")
                                    similarity_values.append(self.compute_similarity(temp_vector, profile_values.T))
                                    senti_obj = sentiment_analysis(img)
                                    emotions = senti_obj.execute()
                                    all_img_emotions.append(emotions)
                                    #st.image(senti_obj.img, channels="RGB")
                                    #st.text(emotions)
                                # st.text(similarity_values)
                                max_similarity_idx = similarity_values.index(max(similarity_values))
                                user_emotion = all_img_emotions[max_similarity_idx]
                                st.header("login user emotion")
                                for i in range(2, len(csv_columns)):
                                    string =  "- " + csv_columns[i] + " " + str(user_emotion[csv_columns[i]])
                                    st.subheader(string)
                                user_emotion['day'] = datetime.today().strftime('%A')
                                user_emotion['username'] = username
                                user_data_dict.append(user_emotion)

                                try:
                                    with open(csv_file, 'w') as csvfile:
                                        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                                        writer.writeheader()
                                        for data in user_data_dict:
                                            writer.writerow(data)
                                except IOError:
                                    print("I/O error")


                    elif task == "Weekly Overview":
                        st.subheader("Weekly Overview")
                        # happy_df = df.loc[df["Emotions"] == "happy"]
                        for i in df.keys()[1:]:
                            chart = alt.Chart(df).mark_bar().encode(
                                x='Emotions',
                                y=alt.Y(i, axis=alt.Axis(title='Values')),
                                color=alt.Color("Emotions", scale=alt.Scale(scheme='spectral'))
                            ).properties(
                                width=300,
                                height=400,
                                title='Emotions vs ' + i
                            ).configure_view(
                                strokeWidth=0, width=0
                            )
                            st.write(chart)
                    elif task == "Happiness vs Sadness Tracker":
                        st.subheader("Happiness vs Sadness Tracker")
                        happysad_df = df.T
                        header_row = happysad_df.iloc[0]
                        happysad_df_2 = pd.DataFrame(happysad_df.values[1:], columns=header_row)
                        counter = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                        happysad_df_2["Day"] = counter
                        chart = alt.Chart(happysad_df_2).mark_line().encode(
                            x='Day',
                            y=alt.Y("happy")
                        ).properties(
                            width=300,
                            height=400,
                            title='Happiness Coefficent through the Week'
                        ).configure_view(
                            strokeWidth=0, width=0
                        )
                        st.write(chart)
                        chart_1 = alt.Chart(happysad_df_2).mark_line().encode(
                            x='Day',
                            y=alt.Y("sad")
                        ).properties(
                            width=300,
                            height=400,
                            title='Sadness Coefficent through the Week'
                        ).configure_view(
                            strokeWidth=0, width=0
                        )
                        st.write(chart_1)
                else:
                    st.warning("Incorrect Username/Password")
        elif choice == "SignUp":
            st.subheader("Create New Account")
            new_user = st.text_input("Username")
            new_password = st.text_input("Password", type='password')
            profile_picture = st.file_uploader("Upload Picture")
            # profile = cv2.imread(profile_picture)
            # profile = cv2.cvtColor(profile, cv2.COLOR_BGR2RGB)
            if profile_picture is not None:
                image = Image.open(profile_picture).convert('RGB')
                profile_picture_vector = np.array(image)  # reshape (194, 188, 4) or any other shape to (224, 224, 3)
                profile_picture_vector = tf.image.resize(profile_picture_vector, size=(224, 224))
                profile_picture_vector = profile_picture_vector.numpy()
                st.text(profile_picture_vector)
            if st.button("Signup"):
                create_usertable()
                add_userdata(new_user, make_hashes(new_password))
                self.user_table[new_user] = [new_password, profile_picture_vector]
                with open("user_table.pickle", "wb") as file:
                    pickle.dump(self.user_table, file, pickle.HIGHEST_PROTOCOL)
                st.success("You have successfully created a valid Account")
                st.info("Go to Login Menu to login")


if __name__ == '__main__':
    main_obj = main_func()
    main_obj.execute()
