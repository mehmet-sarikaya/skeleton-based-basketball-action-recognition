import tensorflow as tf
from ModelCreator import ModelCreator
from keras.regularizers import l2
from keras.optimizers import Adam
from KeypointDatasetProcessor import KeypointDatasetProcessor
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, KFold, StratifiedGroupKFold, train_test_split
from keras.callbacks import EarlyStopping
import numpy as np

#evaluation
from sklearn import metrics
import matplotlib.pyplot as plt
import xgboost as xgb

class ModelTrainer:
    def __init__(self, model_creator: ModelCreator, keypt_processor: KeypointDatasetProcessor, label_id_dic,
                 random_state=42):
        self.model_creator = model_creator
        self.random_state = random_state
        self.x = keypt_processor.x
        self.y = keypt_processor.y
        self.subjects = keypt_processor.subjects

        self.label_id_dic = label_id_dic

    def train_base(self, train_idx, test_idx):
        # Split the data
        x_train, x_test = self.x[train_idx], self.x[test_idx]
        y_train, y_test = self.y[train_idx], self.y[test_idx]

        # create new model each time
        model = self.model_creator.create_model()

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

    def train_model_gkf(self, n_splits):
        group_kfold = GroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        for i, (train_idx, test_idx) in enumerate(group_kfold.split(self.x, self.y, groups=self.subjects)):
            self.train_base(train_idx, test_idx)

    def train_model_sgkf(self, n_splits):
        group_kfold = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        for i, (train_idx, test_idx) in enumerate(group_kfold.split(self.x, self.y, groups=self.subjects)):
            self.train_base(train_idx, test_idx)

    def train_model_classic_kfold(self, n_splits=2):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        for i, (train_idx, test_idx) in enumerate(kf.split(self.x, self.y)):
            print(f"\n--- Klassischer Fold {i + 1} (Subjekte gemischt) ---")
            self.train_base(train_idx, test_idx)

    def train_model_sklearn_simple(self, test_size=0.2):
        """Klassischer Split: Mischt alle Fenster zufällig."""
        indices = np.arange(len(self.x))

        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            shuffle=True,
            random_state=self.random_state
        )

        print(f"\n--- Sklearn Simple Split ({1 - test_size:.0%}/{test_size:.0%}) ---")
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

    def train_xgb(self, train_idx, test_idx):
        x_train, x_test = self.x[train_idx], self.x[test_idx]
        y_train, y_test = self.y[train_idx], self.y[test_idx]

        n_samples_train = x_train.shape[0]
        n_samples_test = x_test.shape[0]

        x_train_flat = x_train.reshape(n_samples_train, -1)
        x_test_flat = x_test.reshape(n_samples_test, -1)

        clf = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=self.random_state,
            objective='multi:softprob',  # Für Multi-Class Classification
            use_label_encoder=False,
            eval_metric='mlogloss'
        )

        print("Starte XGBoost Training...")
        clf.fit(x_train_flat, y_train, eval_set=[(x_test_flat, y_test)], verbose=False)

        self.evaluate_xgb(clf, x_test_flat, y_test)

    def evaluate_xgb(self, model, x_flat, y_true):
        y_pred = model.predict(x_flat)

        labels_ids = sorted(self.label_id_dic.values())
        labels_names = [name for name, i in sorted(self.label_id_dic.items(), key=lambda x: x[1])]

        cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_ids)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_names)

        fig, ax = plt.subplots(figsize=(8, 6))
        cm_display.plot(ax=ax, cmap=plt.cm.Greens)
        plt.title("Confusion Matrix (XGBoost)")
        plt.show()

        print(metrics.classification_report(y_true, y_pred, target_names=labels_names, zero_division=0))

    def train_model_xgb_gkf(self, n_splits):
        group_kfold = GroupKFold(n_splits)
        for i, (train_idx, test_idx) in enumerate(group_kfold.split(self.x, self.y, groups=self.subjects)):
            print(f"\n--- XGBoost Fold {i + 1} ---")
            self.train_xgb(train_idx, test_idx)
