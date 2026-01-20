from sklearn.utils import class_weight

from ModelCreator import ModelCreator
from keras.optimizers import Adam
from KeypointDatasetProcessor import KeypointDatasetProcessor
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, KFold, StratifiedGroupKFold, train_test_split
from keras.callbacks import EarlyStopping
import numpy as np
import uuid

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
        self.is_original_data = keypt_processor.is_original_data

        self.label_id_dic = label_id_dic

    def train_base(self, train_idx, test_idx, merge_classes=True):
        if merge_classes:
            self.apply_new_grouping()
        x_train = self.x[train_idx]
        y_train = self.y[train_idx]

        x_test = self.x[test_idx][self.is_original_data[test_idx]]
        y_test = self.y[test_idx][self.is_original_data[test_idx]]

        # create new model each time
        model = self.model_creator.create_model()

        hp_learning_rate = 0.00005  # Hyperparameter for learning rate

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
        class_weight = self.compute_class_weights(y_train)

        model.fit([x_train],
                  y_train,
                  epochs=100,
                  batch_size=128,
                  validation_split=0.2,
                  verbose=1,
                  class_weight=class_weight,
                  callbacks=[es])

        self.evaluate_model(model, x_test, y_test)

        model_id = str(uuid.uuid4())[:8]
        model_name = f"bball_gesture_pose_{model_id}.keras"
        model.save(f"models/{model_name}")
        print(f"saved model at {model_name}")

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
        classes = np.unique(y_train)
        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=y_train
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
