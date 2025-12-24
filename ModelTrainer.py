import tensorflow as tf
from ModelCreator import ModelCreator
from keras.regularizers import l2
from keras.optimizers import Adam
from KeypointDatasetProcessor import KeypointDatasetProcessor
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from keras.callbacks import EarlyStopping
import numpy as np

#evaluation
from sklearn import metrics
import matplotlib.pyplot as plt


class ModelTrainer:
    def __init__(self, model_creator: ModelCreator, keypt_processor: KeypointDatasetProcessor, label_id_dic):
        self.model_creator = model_creator
        self.x = keypt_processor.x
        self.y = keypt_processor.y
        self.subjects = keypt_processor.subjects

        self.label_id_dic = label_id_dic

    def train_base(self, train_idx, test_idx):
        # New Subject ID as test each time
        current_test_subject = self.subjects[test_idx[0]]
        print(f"Training fold. Leaving out Subject ID: {current_test_subject}")

        # Split the data
        x_train, x_test = self.x[train_idx], self.x[test_idx]
        y_train, y_test = self.y[train_idx], self.y[test_idx]

        # create new model each time
        model = self.model_creator.create_conv2d_lstm_model()

        hp_learning_rate = 0.0001  # Hyperparameter for learning rate

        optimizer = Adam(learning_rate=hp_learning_rate)

        # Compile the model
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

        es = EarlyStopping(
            monitor="val_loss",
            min_delta=0.01,
            patience=8,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
            start_from_epoch=0,
        )

        model.fit([x_train], y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1, callbacks=[es])

        self.evaluate_model(model, x_test, y_test)

    def train_model_loso(self):
        logo = LeaveOneGroupOut()
        for train_idx, test_idx in logo.split(self.x, self.y, groups=self.subjects):
            self.train_base(train_idx, test_idx)

    def train_model_gkf(self):
        group_kfold = GroupKFold(n_splits=2, shuffle=True)
        for i, (train_idx, test_idx) in enumerate(group_kfold.split(self.x, self.y, groups=self.subjects)):
            self.train_base(train_idx, test_idx)

    def evaluate_model(self, model, x, y_true):
        y_pred_probs = model.predict(x)
        y_pred = np.argmax(y_pred_probs, axis=-1)

        labels_ids = sorted(self.label_id_dic.values())  # ergibt [0, 1, 2]
        labels_names = [name for name, i in sorted(self.label_id_dic.items(), key=lambda x: x[1])]

        cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_ids)

        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_names)

        # 5. Plotten
        fig, ax = plt.subplots(figsize=(8, 6))
        cm_display.plot(ax=ax, cmap=plt.cm.Blues)
        plt.title("Confusion Matrix (LOSO Fold)")
        plt.show()

        print(metrics.classification_report(y_true, y_pred, target_names=labels_names, zero_division=0))
