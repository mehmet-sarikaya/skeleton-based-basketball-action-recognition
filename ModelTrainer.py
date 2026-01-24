from sklearn.utils import class_weight

from ModelCreator import ModelCreator
from keras.optimizers import Adam
from KeypointDatasetProcessor import KeypointDatasetProcessor
from sklearn.model_selection import (LeaveOneGroupOut, GroupKFold, KFold, StratifiedGroupKFold, train_test_split,
                                     GroupShuffleSplit)
from keras.callbacks import EarlyStopping, Callback, TensorBoard, ReduceLROnPlateau
from keras.utils import to_categorical
import numpy as np
import uuid
import os
import datetime

#evaluation
from sklearn import metrics
import matplotlib.pyplot as plt
import xgboost as xgb

class ModelTrainer:
    def __init__(self, model_creator: ModelCreator, keypt_processor: KeypointDatasetProcessor, label_id_dic,
                 batch_size, lr, patience, num_classes, random_state=42):
        self.model_creator = model_creator
        self.random_state = random_state
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience

        self.x = keypt_processor.x
        self.y = keypt_processor.y
        self.video_id = keypt_processor.video_id
        self.is_original_data = keypt_processor.is_original_data

        self.num_classes = num_classes

        self.label_id_dic = label_id_dic

    def train_base(self, train_idx, test_idx, merge_classes=False):
        self.check_label_distribution()

        if merge_classes:
            self.apply_new_grouping()

        x_train = self.x[train_idx]
        y_train = to_categorical(self.y[train_idx], num_classes=self.num_classes)

        x_test = self.x[test_idx][self.is_original_data[test_idx]]
        y_test = to_categorical(self.y[test_idx][self.is_original_data[test_idx]], num_classes=self.num_classes)

        # create new model each time
        model = self.model_creator.create_model()

        optimizer = Adam(learning_rate=self.lr)

        # Compile the model
        model.compile(optimizer=optimizer, loss='categorical_crossentropy')

        es = EarlyStopping(
            monitor="val_loss",
            min_delta=0.01,
            patience=self.patience,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
            start_from_epoch=0,
        )
        best_val_loss = BestValLossCallback()

        log_run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = "logs"

        tboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,  # Records weights distribution every epoch
            write_graph=True,  # Visualizes the model architecture
            update_freq='epoch'  # How often to write logs
        )

        reduce_lr = ReduceLROnPlateau(patience=int(self.patience*0.35), factor=0.7, min_lr=self.lr/10, verbose=1)

        x_t, x_val, y_t, y_val = train_test_split(
            x_train, y_train,
            test_size=0.2,
            random_state=self.random_state,
            shuffle=True,
            stratify=np.argmax(y_train, axis=1)  # Sorgt für gleiche Klassenverteilung
        )


        class_weight = self.compute_class_weights(y_t)

        model.fit(x_t,
                  y_t,
                  epochs=1000,
                  batch_size=self.batch_size,
                  validation_data=(x_val, y_val),
                  class_weight=class_weight,
                  verbose=1,
                  callbacks=[es, best_val_loss, tboard, reduce_lr])

        self.evaluate_model(model, x_test, y_test)

        model_id = str(uuid.uuid4())[:8]
        model_name = f"bball_gesture_pose_{model_id}.keras"
        model.save(f"models/{model_name}")
        print(f"saved model at {model_name}")

    def train_model_loso(self):
        logo = LeaveOneGroupOut()
        for train_idx, test_idx in logo.split(self.x, self.y, groups=self.video_id):
            self.train_base(train_idx, test_idx)

    def train_model_gkf(self, n_splits):
        group_kfold = GroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        for i, (train_idx, test_idx) in enumerate(group_kfold.split(self.x, self.y, groups=self.video_id)):
            self.train_base(train_idx, test_idx)

    def train_model_sgkf(self, n_splits):
        group_kfold = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        for i, (train_idx, test_idx) in enumerate(group_kfold.split(self.x, self.y, groups=self.video_id)):
            self.train_base(train_idx, test_idx)

    def train_model_classic_kfold(self, n_splits=2):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        for i, (train_idx, test_idx) in enumerate(kf.split(self.x, self.y)):
            print(f"\n--- Klassischer Fold {i + 1} (Subjekte gemischt) ---")
            self.train_base(train_idx, test_idx)

    def train_model_sklearn_simple(self, test_size=0.2):
        """Klassischer Split: Mischt alle Fenster zufällig."""
        self.check_label_distribution()
        # self.balance_data(max_samples_per_class=2000)

        indices = np.arange(len(self.x))

        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            shuffle=True,
            random_state=self.random_state
        )

        print(f"\n--- Sklearn Simple Split ({1 - test_size:.0%}/{test_size:.0%}) ---")
        self.train_base(train_idx, test_idx)

    def train_model_group_shuffle_split(self, n_splits=1, test_size=0.2):
        """Klassischer Split: Mischt alle Fenster zufällig."""
        self.check_label_distribution()
        # self.balance_data(max_samples_per_class=2000)

        gss = GroupShuffleSplit(n_splits=1, random_state=self.random_state)

        for train_idx, test_idx in gss.split(X=self.x, y=self.y, groups=self.video_id):
            print(f"\n--- Group Shuffle Split ({1 - test_size:.0%}/{test_size:.0%}) ---")
            self.train_base(train_idx, test_idx)

    def evaluate_model(self, model, x, y_true):
        y_pred_probs = model.predict(x)
        y_pred = np.argmax(y_pred_probs, axis=-1)

        y_true = np.asarray(y_true)
        y_true = np.argmax(y_true, axis=-1)
        y_pred = np.asarray(y_pred)

        # label_id_dic is {id: name}
        all_label_ids = sorted(self.label_id_dic.keys())

        # Keep only labels that actually occur in this fold (true OR predicted)
        present_ids = sorted(set(np.unique(y_true)).union(set(np.unique(y_pred))))
        labels_ids = [i for i in all_label_ids if i in present_ids]
        labels_names = [self.label_id_dic[i] for i in labels_ids]

        if len(labels_ids) == 0:
            print("⚠️ No labels present in y_true/y_pred for this fold. Skipping confusion matrix.")
            return

        cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_ids)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_names)

        fig, ax = plt.subplots(figsize=(8, 6))
        cm_display.plot(ax=ax, cmap=plt.cm.Blues, values_format="d")
        plt.title("Confusion Matrix (LOSO Fold)")
        plt.show()

        print(metrics.classification_report(
            y_true,
            y_pred,
            labels=labels_ids,  # IMPORTANT: match filtered ids
            target_names=labels_names,  # aligned names
            zero_division=0
        ))

    def compute_class_weights(self, y_train):
        y_to_integers = np.argmax(y_train,axis=1)
        classes = np.unique(y_to_integers)
        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_to_integers
        )
        return dict(zip(classes, weights))

    def apply_new_grouping(self):
        id_to_new_id = {
            0: 1,  # block -> Defensive Stance
            1: 2,  # pass -> Passing
            2: 0,  # run -> Active Movement
            3: 0,  # dribble -> Active Movement
            4: 3,  # shoot -> Shooting
            5: 0,  # ball in hand -> Active Movement
            6: 1,  # defense -> Defensive Stance
            7: 0,  # pick -> Active Movement
            8: 0,  # no_action -> Active Movement
            9: 0  # walk -> Active Movement
        }

        self.y = np.array([id_to_new_id[val] for val in self.y])

        self.label_id_dic = {
            0: "Active Movement",
            1: "Defensive Stance",
            2: "Passing",
            3: "Shooting"
        }

    def check_label_distribution(self):
        """Gibt die Anzahl der Samples pro Klasse aus."""
        # IDs und deren Häufigkeit zählen
        unique, counts = np.unique(self.y, return_counts=True)
        counts_dict = dict(zip(unique, counts))

        print(f"{'ID':<5} | {'Label Name':<20} | {'Anzahl':<10} | {'Anteil':<10}")
        print("-" * 55)

        total = len(self.y)
        for label_id in sorted(self.label_id_dic.keys()):
            count = counts_dict.get(label_id, 0)
            name = self.label_id_dic[label_id]
            percentage = (count / total) * 100
            print(f"{label_id:<5} | {name:<20} | {count:<10} | {percentage:>6.2f}%")

        print("-" * 55)
        print(f"Gesamtanzahl Samples: {total}")

    def balance_data(self, max_samples_per_class=5000):
        """
        Balanciert die Daten in self.x, self.y, self.video_id
        und self.is_original_data durch Downsampling.
        """
        unique_classes = np.unique(self.y)
        indices_to_keep = []

        for c in unique_classes:
            class_indices = np.where(self.y == c)[0]
            if len(class_indices) > max_samples_per_class:
                # Zufällige Auswahl ohne Zurücklegen
                keep = np.random.choice(class_indices, max_samples_per_class, replace=False)
                indices_to_keep.extend(keep)
            else:
                # Wenn Klasse kleiner als das Limit ist, behalte alle Samples
                indices_to_keep.extend(class_indices)

        # Indices in ein Numpy-Array umwandeln und mischen
        indices_to_keep = np.array(indices_to_keep)
        np.random.seed(self.random_state)  # Für Reproduzierbarkeit
        np.random.shuffle(indices_to_keep)

        # Alle Datenfelder synchron mit den neuen Indices überschreiben
        self.x = self.x[indices_to_keep]
        self.y = self.y[indices_to_keep]
        self.video_id = self.video_id[indices_to_keep]

        # Hier ist die wichtige Zeile für deine Original-Daten-Maske
        if hasattr(self, 'is_original_data'):
            self.is_original_data = self.is_original_data[indices_to_keep]

        print(f"✅ Data balanced. New total samples: {len(self.y)}")

        # Verteilung zur Kontrolle ausgeben
        self.check_label_distribution()


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

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # label_id_dic is {id: name}
        all_label_ids = sorted(self.label_id_dic.keys())

        # Keep only labels that actually occur in this fold (true OR predicted)
        present_ids = sorted(set(np.unique(y_true)).union(set(np.unique(y_pred))))
        labels_ids = [i for i in all_label_ids if i in present_ids]
        labels_names = [self.label_id_dic[i] for i in labels_ids]

        if len(labels_ids) == 0:
            print("⚠️ No labels present in y_true/y_pred for this fold. Skipping confusion matrix.")
            return

        cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_ids)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_names)

        fig, ax = plt.subplots(figsize=(8, 6))
        cm_display.plot(ax=ax, cmap=plt.cm.Greens)
        plt.title("Confusion Matrix (XGBoost)")
        plt.show()

        print(metrics.classification_report(y_true, y_pred, target_names=labels_names, zero_division=0))

    def train_model_xgb_gkf(self, n_splits):
        group_kfold = GroupKFold(n_splits)
        for i, (train_idx, test_idx) in enumerate(group_kfold.split(self.x, self.y, groups=self.video_id)):
            print(f"\n--- XGBoost Fold {i + 1} ---")
            self.train_xgb(train_idx, test_idx)

    def train_model_xgb_simple_sklearn(self, test_size=0.2):
        """Klassischer Split: Mischt alle Fenster zufällig."""
        self.check_label_distribution()
        # self.balance_data(max_samples_per_class=2000)

        indices = np.arange(len(self.x))

        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            shuffle=True,
            random_state=self.random_state
        )

        print(f"\n--- Sklearn Simple Split ({1 - test_size:.0%}/{test_size:.0%}) ---")
        self.train_xgb(train_idx, test_idx)


class BestValLossCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        val_losses = self.model.history.history.get("val_loss", None)
        if val_losses is None:
            print("No validation loss recorded.")
            return
        best_epoch = int(np.argmin(val_losses)) + 1
        best_val_loss = float(np.min(val_losses))
        print(f"\nBest val_loss = {best_val_loss:.6f} at epoch {best_epoch}")
