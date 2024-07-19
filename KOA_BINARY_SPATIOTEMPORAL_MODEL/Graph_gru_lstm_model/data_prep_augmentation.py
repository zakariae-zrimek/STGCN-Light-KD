import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Indices des parties du corps
index_nose = 0
index_left_eye_inner = 3
index_left_eye = 6
index_left_eye_outer = 9
index_right_eye_inner = 12
index_right_eye = 15
index_right_eye_outer = 18
index_left_ear = 21
index_right_ear = 24
index_mouth_left = 27
index_mouth_right = 30
index_left_shoulder = 33
index_right_shoulder = 36
index_left_elbow = 39
index_right_elbow = 42
index_left_wrist = 45
index_right_wrist = 48
index_left_pinky = 51
index_right_pinky = 54
index_left_index = 57
index_right_index = 60
index_left_thumb = 63
index_right_thumb = 66
index_left_hip = 69
index_right_hip = 72
index_left_knee = 75
index_right_knee = 78
index_left_ankle = 81
index_right_ankle = 84
index_left_heel = 87
index_right_heel = 90
index_left_foot = 93
index_right_foot = 96


class Data_Loader:
    def __init__(self, dir):
        self.num_repitation = 5
        self.num_channel = 3
        self.dir = dir
        self.body_part = self.body_parts()
        self.dataset = []
        self.sequence_length = []
        self.num_timestep = 70
        self.new_label = []
        self.train_x, self.train_y = self.import_dataset()
        self.batch_size = self.train_y.shape[0]
        self.num_joints = len(self.body_part)
        self.scaler = StandardScaler()  # You can switch to MinMaxScaler if needed
        self.scaled_x, self.scaled_y = self.preprocessing()

    def body_parts(self):
        body_parts = [
            index_nose,
            index_left_eye_inner,
            index_left_eye,
            index_left_eye_outer,
            index_right_eye_inner,
            index_right_eye,
            index_right_eye_outer,
            index_left_ear,
            index_right_ear,
            index_mouth_left,
            index_mouth_right,
            index_left_shoulder,
            index_right_shoulder,
            index_left_elbow,
            index_right_elbow,
            index_left_wrist,
            index_right_wrist,
            index_left_pinky,
            index_right_pinky,
            index_left_index,
            index_right_index,
            index_left_thumb,
            index_right_thumb,
            index_left_hip,
            index_right_hip,
            index_left_knee,
            index_right_knee,
            index_left_ankle,
            index_right_ankle,
            index_left_heel,
            index_right_heel,
            index_left_foot,
            index_right_foot,
        ]
        return body_parts

    def import_dataset(self):
        train_x = np.load(f"./{self.dir}/train_70.npy", allow_pickle=True)
        train_y = pd.read_csv(f"./{self.dir}/labels_70_bin.csv").values
        print(train_x.shape, train_y.shape)
        return train_x, train_y

    def augment_data(self, X, y):
        augmented_X = []
        augmented_y = []

        for i in range(len(X)):
            augmented_X.append(X[i])
            augmented_y.append(y[i])
            augmented_X.append(self.random_translate(X[i]))
            augmented_y.append(y[i])
            augmented_X.append(self.random_rotate(X[i]))
            augmented_y.append(y[i])
            augmented_X.append(self.random_noise(X[i]))
            augmented_y.append(y[i])

        return np.array(augmented_X), np.array(augmented_y)

    def random_translate(self, sequence):
        # Translation aléatoire des séquences de squelettes
        max_translation = 3
        translated_sequence = sequence + np.random.uniform(
            -max_translation, max_translation, sequence.shape
        )
        return translated_sequence

    def random_rotate(self, sequence):
        # Rotation aléatoire des séquences de squelettes
        max_angle = np.pi / 12
        angle = np.random.uniform(-max_angle, max_angle)
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )

        # Appliquer la rotation uniquement sur les coordonnées x et y
        rotated_sequence = np.copy(sequence)
        for t in range(sequence.shape[0]):
            for j in range(sequence.shape[1]):
                xy = sequence[t, j, :2]  # Extraire les coordonnées x et y
                rotated_xy = np.dot(xy, rotation_matrix)  # Appliquer la rotation
                rotated_sequence[t, j, :2] = (
                    rotated_xy  # Mettre à jour les coordonnées x et y
                )

        return rotated_sequence

    def random_noise(self, sequence):
        # Ajout de bruit gaussien
        noise = np.random.normal(0, 0.1, sequence.shape)
        noisy_sequence = sequence + noise
        return noisy_sequence

    def preprocessing(self):
        print('x')
        X_train = np.reshape(
            self.train_x,
            (
                self.train_x.shape[0] * self.train_x.shape[1],
                self.train_x.shape[2] * self.train_x.shape[3],
            ),
        )
        X_train = self.scaler.fit_transform(X_train)
        X_train = np.reshape(
            X_train,
            (
                self.train_x.shape[0],
                self.train_x.shape[1],
                self.train_x.shape[2],
                self.train_x.shape[3],
            ),
        )

        # Train-test split before applying SMOTE
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, self.train_y, test_size=0.2, random_state=42, stratify=self.train_y
        )

        # Augmentation des données
        X_train, y_train = self.augment_data(X_train, y_train)
        print("augmented data")

        # SMOTE for data augmentation
        smote = SMOTE(sampling_strategy="auto", random_state=42)
        X_train_smot = np.reshape(X_train, (X_train.shape[0], -1))
        X_train_smot, y_train = smote.fit_resample(X_train_smot, y_train)
        X_train = np.reshape(
            X_train_smot,
            (
                X_train_smot.shape[0],
                self.train_x.shape[1],
                self.train_x.shape[2],
                self.train_x.shape[3],
            ),
        )

        print("After SMOTE and augmentation:", X_train.shape, y_train.shape)
        return X_train, y_train
