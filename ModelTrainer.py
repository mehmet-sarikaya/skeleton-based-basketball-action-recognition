import traceback

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
import datetime
import os
import tensorflow as tf
import pandas as pd

#evaluation
from sklearn import metrics
import matplotlib.pyplot as plt
import xgboost as xgb
import platform

class ModelTrainer:
    def __init__(self, model_creator: ModelCreator, keypt_processor: KeypointDatasetProcessor, label_id_dic,
                 batch_size, lr, patience, num_classes, max_epochs, random_state=42):
        self.model_creator = model_creator
        self.random_state = random_state
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience

        self.x = keypt_processor.x
        self.y = keypt_processor.y
        self.video_id = keypt_processor.video_id
        self.is_original_data = keypt_processor.is_original_data

        self.num_classes = num_classes

        self.label_id_dic = label_id_dic

        self.fold_reports = []
        self.session_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def train_base(self, train_idx, test_idx, merge_classes=False, remove_aug_data_val_test=True):
        if merge_classes:
            self.apply_new_grouping()

        x_train = self.x[train_idx]
        y_train = to_categorical(self.y[train_idx], num_classes=self.num_classes)

        # Test erst später bauen (nach optionaler Bereinigung über Indizes)
        test_idx_clean = test_idx
        if remove_aug_data_val_test:
            test_idx_clean = test_idx_clean[self.is_original_data[test_idx_clean]]

        # create new model each time
        model = self.model_creator.create_model()

        optimizer = Adam(learning_rate=self.lr)
        metrics_list = ["accuracy"]

        # Compile the model

        if self.model_creator.model_name == "gcn_paper":
            model.compile(optimizer=optimizer,
                          loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                          metrics=metrics_list)
        else:
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics_list)

        groups_train = self.video_id[train_idx]
        gss_val = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=self.random_state)
        idx_t, idx_val = next(gss_val.split(x_train, y_train, groups=groups_train))

        # globale Indizes (wichtig für is_original_data)
        train_idx_t_global = train_idx[idx_t]
        train_idx_val_global = train_idx[idx_val]

        # Val bereinigen: nur Originaldaten
        if remove_aug_data_val_test:
            train_idx_val_global = train_idx_val_global[self.is_original_data[train_idx_val_global]]

        # Jetzt final arrays bauen
        x_t = self.x[train_idx_t_global]
        y_t = to_categorical(self.y[train_idx_t_global], num_classes=self.num_classes)

        x_val = self.x[train_idx_val_global]
        y_val = to_categorical(self.y[train_idx_val_global], num_classes=self.num_classes)

        # Test final bauen (optional bereinigt)
        x_test = self.x[test_idx_clean]
        y_test = to_categorical(self.y[test_idx_clean], num_classes=self.num_classes)

        # Callbacks and class weight ########################################################################
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
        reduce_lr = ReduceLROnPlateau(patience=int(self.patience * 0.4), factor=0.5, min_lr=self.lr / 10, verbose=1,
                                      min_delta=0.01)
        class_weights = self.compute_class_weights(y_t)

        # End of Callbacks and Weights ##########################################

        self.check_label_distribution(y_t, "Training")
        self.check_label_distribution(y_val, "Validation")
        self.check_label_distribution(y_test, "Test")

        if self.model_creator.model_name == "gcn_paper":
            x_t = to_stgcn_input(x_t)
            x_val = to_stgcn_input(x_val)
            x_test = to_stgcn_input(x_test)

        print("x_t:", x_t.shape, x_t.dtype)
        print("y_t:", y_t.shape, y_t.dtype)
        print("unique y_t (first 20):", np.unique(y_t[:20]) if y_t.ndim == 1 else "one-hot?")

        model.fit(x_t,
                  y_t,
                  epochs=self.max_epochs,
                  batch_size=self.batch_size,
                  validation_data=(x_val, y_val),
                  class_weight=class_weights,
                  verbose=1,
                  callbacks=[es, best_val_loss, tboard, reduce_lr])

        model_id = str(uuid.uuid4())[:8]
        fold_id = f"bball_gesture_pose_{model_id}"

        self.evaluate_model(model, x_test, y_test, fold_id)

        model_architecture_name = getattr(self.model_creator, "model_name", "unknown_model")
        model_save_path_curr = os.path.join("evaluations", model_architecture_name, self.session_id, fold_id)

        self.save_model(model, model_save_path_curr, fold_id, extension=".keras")

    def save_model(self, model, path, fold_id, extension, only_weights=False):
        try:
            model_save_path = os.path.join(path, fold_id + extension)
            if only_weights:
                model_save_path = os.path.join(path, fold_id + ".weights.h5")
                model.save_weights(model_save_path)
            else:
                model.save(model_save_path)
            print(f"saved model at {model_save_path}")
        except (NotImplementedError, ValueError, TypeError) as e:
            if only_weights:
                print("Could not save model due to Error")
                traceback.print_exc()
                print(e)
            else:
                print(f"Could not save as {extension}.")
                print(e)
                traceback.print_exc()
                print("Try saving as .weights.h5")
                self.save_model(model, path, fold_id, ".weights.h5", only_weights=True)

    def train_model_loso(self):
        logo = LeaveOneGroupOut()
        for train_idx, test_idx in logo.split(self.x, self.y, groups=self.video_id):
            self.train_base(train_idx, test_idx)

    def train_model_gkf(self, n_splits):
        group_kfold = GroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        for i, (train_idx, test_idx) in enumerate(group_kfold.split(self.x, self.y, groups=self.video_id)):
            self.train_base(train_idx, test_idx)

    def train_model_sgkf(self, n_splits, balance_data=True):
        session_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        session_name = f"Session_{session_id}"

        print("Balancing Data? : ", balance_data)
        self.check_label_distribution()
        if balance_data:
            self.balance_data(max_samples_per_class=12000)
        self.check_label_distribution()
        group_kfold = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        for i, (train_idx, test_idx) in enumerate(group_kfold.split(self.x, self.y, groups=self.video_id)):
            print(f"\n------------------------ Fold {i + 1} ----------------------------")
            self.train_base(train_idx, test_idx)

        self.save_cv_final_results(n_splits=n_splits)

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
            stratify=self.y,
            random_state=self.random_state
        )

        print(f"\n--- Sklearn Simple Split ({1 - test_size:.0%}/{test_size:.0%}) ---")
        self.train_base(train_idx, test_idx)

    def train_model_group_shuffle_split(self, n_splits=1, test_size=0.2, balance_data=False):
        """Klassischer Split: Mischt alle Fenster zufällig."""
        self.check_label_distribution()
        if balance_data:
            self.balance_data(max_samples_per_class=3500)
        self.check_label_distribution()

        gss = GroupShuffleSplit(n_splits=1, random_state=self.random_state, test_size=test_size)

        for train_idx, test_idx in gss.split(X=self.x, y=self.y, groups=self.video_id):
            print(f"\n--- Group Shuffle Split ({1 - test_size:.0%}/{test_size:.0%}) ---")
            self.train_base(train_idx, test_idx)

    def evaluate_model(self, model, x, y_true, save_name):
        if self.model_creator.model_name == "gcn_paper":
            # ST-GCN gibt LOGITS zurück
            logits = model.predict(x, verbose=0)  # (N, num_classes)
            y_pred_probs = tf.nn.softmax(logits, axis=-1).numpy()
            y_pred = np.argmax(y_pred_probs, axis=-1)

        else:
            # Alle anderen Modelle geben Softmax-Wahrscheinlichkeiten zurück
            y_pred_probs = model.predict(x, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=-1)

        y_true = np.asarray(y_true)
        y_true = np.argmax(y_true, axis=-1)
        y_pred = np.asarray(y_pred)

        all_label_ids = sorted(self.label_id_dic.keys())

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
        os_name = platform.system()
        if os_name == "Windows":
            plt.show()
        elif os_name == "Linux":
            print("Not showing plot since you're on BwUniCluster 3.0 probably")
        elif os_name == "Darwin":
            print("You are on macOS?! How?")

        report_str = metrics.classification_report(
            y_true,
            y_pred,
            labels=labels_ids,
            target_names=labels_names,
            zero_division=0
        )
        print(report_str)

        self.save_evaluation_results(
            cm=cm,
            labels_names=labels_names,
            y_true=y_true,
            y_pred=y_pred,
            save_name=save_name
        )

    def save_evaluation_results(
            self,
            cm,
            labels_names,
            y_true,
            y_pred,
            save_name
    ):
        model_dir_name = getattr(self.model_creator, "model_name", "unknown_model")
        out_dir = os.path.join("evaluations", model_dir_name, self.session_id, str(save_name))
        os.makedirs(out_dir, exist_ok=True)

        # --- Confusion matrix plot speichern ---
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_names)
        disp.plot(ax=ax, cmap=plt.cm.Blues, values_format="d")
        ax.set_title(f"Confusion Matrix ({save_name})")
        fig.tight_layout()

        cm_path = os.path.join(out_dir, "confusion_matrix.png")
        fig.savefig(cm_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        # --- Confusion matrix als Array speichern ---
        np.save(os.path.join(out_dir, "confusion_matrix.npy"), cm)

        # --- Confusion matrix als CSV speichern ---
        cm_csv_path = os.path.join(out_dir, "confusion_matrix.csv")
        df_cm = pd.DataFrame(cm, index=labels_names, columns=labels_names)
        df_cm.to_csv(cm_csv_path, sep=";", decimal=",")

        # --- Classification report ---
        report_dict = metrics.classification_report(
            y_true,
            y_pred,
            target_names=labels_names,
            zero_division=0,
            output_dict=True
        )

        self.fold_reports.append(report_dict)

        # TXT (wie bisher, nur schöner formatiert)
        report_str = metrics.classification_report(
            y_true,
            y_pred,
            target_names=labels_names,
            zero_division=0
        )

        report_path = os.path.join(out_dir, "classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_str)

        # CSV
        df_report = pd.DataFrame(report_dict).T
        report_csv_path = os.path.join(out_dir, "classification_report.csv")
        df_report.to_csv(report_csv_path, sep=";", decimal=",")

        print(f"💾 Evaluation saved to: {out_dir}")

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

        self.num_classes = 4
        self.model_creator.num_classes = 4

    def check_label_distribution(self, y=None, title="Alle Daten"):
        """
        Benötigt:
        - y (Optional): NumPy-Array mit Labels. Falls None, wird self.y verwendet.
        - self.label_id_dic: Dictionary {id: name}.
        - title (Optional): Ein String für die Kopfzeile.
        """
        if title:
            print(f"\n{title} Labels:")

        y_to_process = y if y is not None else self.y

        y_working = np.array(y_to_process)

        if y_working.ndim > 1 and y_working.shape[1] > 1:
            y_labels = np.argmax(y_working, axis=1)
        else:
            y_labels = y_working

        unique, counts = np.unique(y_labels, return_counts=True)
        counts_dict = dict(zip(unique, counts))

        print(f"{'ID':<5} | {'Label Name':<20} | {'Anzahl':<10} | {'Anteil':<10}")
        print("-" * 55)

        total = len(y_labels)
        for label_id in sorted(self.label_id_dic.keys()):
            count = counts_dict.get(label_id, 0)
            name = self.label_id_dic[label_id]
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"{label_id:<5} | {name:<20} | {count:<10} | {percentage:>6.2f}%")

        print("-" * 55)
        print(f"Gesamtanzahl Samples: {total}\n")

    def balance_data(self, max_samples_per_class=5000):
        """
        Balanciert die Daten in self.x, self.y, self.video_id
        und self.is_original_data durch Downsampling.
        """
        unique_classes = np.unique(self.y)
        indices_to_keep = []

        np.random.seed(self.random_state)

        for c in unique_classes:
            class_indices = np.where(self.y == c)[0]

            # Split: Original vs Augmented (Augmented soll zuerst rausfliegen)
            orig_idx = class_indices[self.is_original_data[class_indices]]
            aug_idx = class_indices[~self.is_original_data[class_indices]]

            if len(class_indices) > max_samples_per_class:
                n_keep = max_samples_per_class

                # Erst Augmented löschen: Originals bevorzugt behalten
                if len(orig_idx) >= n_keep:
                    keep = np.random.choice(orig_idx, n_keep, replace=False)
                else:
                    n_missing = n_keep - len(orig_idx)
                    keep_aug = np.random.choice(aug_idx, n_missing, replace=False)
                    keep = np.concatenate([orig_idx, keep_aug])

                indices_to_keep.extend(keep.tolist())
            else:
                indices_to_keep.extend(class_indices.tolist())

        indices_to_keep = np.array(indices_to_keep)
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

    def save_cv_final_results(self, n_splits):
        if not self.fold_reports:
            return

        model_dir_name = getattr(self.model_creator, "model_name", "unknown_model")
        # Hauptordner für diese CV-Session
        out_dir = os.path.join("evaluations", model_dir_name, self.session_id, "final_cv_summary")
        os.makedirs(out_dir, exist_ok=True)

        # Alle Metriken sammeln (F1, Precision, Recall)
        metrics_to_average = ['precision', 'recall', 'f1-score']
        final_summary = {}

        sample_report = self.fold_reports[0]
        target_keys = [k for k in sample_report.keys() if k != 'accuracy']

        for key in target_keys:
            final_summary[key] = {}
            for m in metrics_to_average:
                scores = [r[key][m] for r in self.fold_reports if key in r]
                final_summary[key][m] = np.mean(scores)
                final_summary[key][f"{m}_std"] = np.std(scores)  # Standardabweichung für Thesis!

        # Als DataFrame speichern (CSV)
        df_final = pd.DataFrame(final_summary).T
        df_final.to_csv(os.path.join(out_dir, "final_cv_metrics_mean.csv"), sep=";", decimal=",")

        # Schöne Textdatei für schnellen Überblick
        summary_path = os.path.join(out_dir, "summary_report.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"FINAL CV RESULTS ({n_splits} Folds)\n")
            f.write("=" * 40 + "\n")
            f.write(f"Overall Macro F1: {final_summary['macro avg']['f1-score']:.4f}\n")
            f.write("-" * 40 + "\n")
            for label in self.label_id_dic.values():
                if label in final_summary:
                    f1 = final_summary[label]['f1-score']
                    std = final_summary[label]['f1-score_std']
                    f.write(f"{label:15}: {f1:.4f} (+/- {std:.4f})\n")

        print(f"📊 Final CV Summary saved to: {out_dir}")

def to_stgcn_input(x):
    """
    x: (N, T, V, C) = (batch, 16, 12, 3)
    returns: (N, C, T, V, 1)
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)   # (N,T,V,C)
    x = tf.transpose(x, [0, 3, 1, 2])               # (N,C,T,V)
    x = tf.expand_dims(x, axis=-1)                  # (N,C,T,V,1)
    return x

class BestValLossCallback(Callback):
    def __init__(self):
        super().__init__()
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
        print("\n \n")

    def on_epoch_end(self, epoch, logs=None):
        current_epoch_to_disp = epoch + 1
        print("\n")
        print("[BestValLossCallback]")
        print(f"Epoch {current_epoch_to_disp}")

        # Nimm den aktuellen val_loss aus den logs dieser Epoche
        current_val_loss = logs.get("val_loss")

        if current_val_loss is None:
            return

        epoch_diff = current_epoch_to_disp - self.best_epoch

        # Vergleiche mit dem bisherigen Bestwert
        if current_val_loss < self.best_val_loss:
            last_val_loss = self.best_val_loss
            epoch_last_best_val_loss = self.best_epoch

            self.best_val_loss = current_val_loss
            self.best_epoch = current_epoch_to_disp

            print(f"New best val_loss after {epoch_diff} epochs!")
            print(f"Old val_loss : {last_val_loss:.4f} in epoch {epoch_last_best_val_loss}")
            print(f"New val_loss : {current_val_loss:.4f} in epoch {current_epoch_to_disp}")
            print(f"Difference in loss: {(current_val_loss - last_val_loss):.4f}")
        else:
            print(f"No improvement since {epoch_diff} epochs")
            print(f"Best so far:      {self.best_val_loss:.4f} in epoch {self.best_epoch}")
            print(f"Current val_loss: {current_val_loss:.4f} in epoch {current_epoch_to_disp}")
