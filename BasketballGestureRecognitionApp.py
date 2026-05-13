import traceback
from InquirerPy import inquirer
from KeypointDatasetProcessor import KeypointDatasetProcessor
from ModelCreator import ModelCreator
from ModelTrainer import ModelTrainer
from KeypointDatasetCreator import KeypointDatasetCreator
from ModelTester import ModelTester


class BasketballGestureRecognitionApp:
    def __init__(self, model_name, n_splits, fps, win_len_sec, stride_len_sec, num_classes, path_to_videos,
                 include_conf, augment_data, batch_size=64, lr=0.0001, patience=20, max_epochs=100):
        self.fps = fps
        self.win_len_sec = win_len_sec
        self.stride_len_sec = stride_len_sec
        self.num_classes = num_classes
        self.path_to_videos = path_to_videos
        self.keypt_dataset_creator = KeypointDatasetCreator(target_fps=self.fps)

        self.n_splits = n_splits
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.max_epochs = max_epochs
        self.random_state = 25
        self.include_conf = include_conf

        self.label_id_dic = {
            0: "block", 1: "pass", 2: "run", 3: "dribble", 4: "shoot",
            5: "ball in hand", 6: "defense", 7: "pick", 8: "no_action", 9: "walk"
        }

        self.augment_data = augment_data
        self.keypt_processor = KeypointDatasetProcessor(
            label_id_dic=self.label_id_dic,
            fps=self.fps,
            random_state=self.random_state
        )

        self.model_name = model_name
        self.model_creator = ModelCreator(
            model_name=self.model_name,
            win_len_sec=self.win_len_sec,
            fps=self.fps,
            num_classes=self.num_classes,
            include_conf=include_conf
        )
        self.model_trainer = None
        self.model_tester = ModelTester(app=self)

    def create_dataset_from_videos(self):
        print(f"\n[INFO] Starte Keypoint-Extraktion aus: {self.path_to_videos}")
        self.keypt_dataset_creator.create_keypoints_dataset(self.path_to_videos, filter_videos=False)

    def load_dataset_and_train(self, strategy):
        print(f"\n[INFO] Lade Datensatz und starte Training mit Strategie: {strategy}...")
        self.keypt_processor.load_dataset(".", include_conf=self.include_conf, augment_data=self.augment_data)

        self.model_trainer = ModelTrainer(
            model_creator=self.model_creator,
            keypt_processor=self.keypt_processor,
            label_id_dic=self.label_id_dic,
            random_state=self.random_state,
            batch_size=self.batch_size,
            lr=self.lr,
            patience=self.patience,
            num_classes=self.num_classes,
            max_epochs=self.max_epochs
        )

        # Mapping der Strategien auf die entsprechenden Methoden im ModelTrainer
        if strategy == "xgb_gkf":
            self.model_trainer.train_model_xgb_gkf(n_splits=self.n_splits)
        elif strategy == "sgkf":
            self.model_trainer.train_model_sgkf(n_splits=self.n_splits)
        elif strategy == "gkf":
            self.model_trainer.train_model_gkf(n_splits=self.n_splits)
        elif strategy == "classic_kfold":
            self.model_trainer.train_model_classic_kfold(n_splits=self.n_splits)
        elif strategy == "sklearn_simple":
            self.model_trainer.train_model_sklearn_simple(test_size=0.2)
        elif strategy == "group_shuffle_split":
            self.model_trainer.train_model_group_shuffle_split(test_size=0.1, n_splits=1)
        elif strategy == "xgb_simple":
            self.model_trainer.train_model_xgb_simple_sklearn(test_size=0.2)

    def train_all_models_cv(self):
        models_to_train = ["gcn_paper", "tcn", "cnn_lstm", "cnn_bilstm"]
        self.keypt_processor.load_dataset(".", include_conf=self.include_conf, augment_data=self.augment_data)

        for m_name in models_to_train:
            print(f"\n### STARTING CROSS-VALIDATION FOR: {m_name} ###")
            self.model_creator.model_name = m_name
            self.model_trainer = ModelTrainer(
                model_creator=self.model_creator,
                keypt_processor=self.keypt_processor,
                label_id_dic=self.label_id_dic,
                random_state=self.random_state,
                batch_size=self.batch_size,
                lr=self.lr,
                patience=self.patience,
                num_classes=self.num_classes,
                max_epochs=self.max_epochs
            )
            try:
                self.model_trainer.train_model_sgkf(n_splits=self.n_splits)
            except Exception as e:
                print(f"Error during CV for {m_name}: {e}")

    def test_model_on_camera(self, model_path, camera_id_or_url=0):
        self.model_tester.test_model_on_camera(model_path=model_path, camera_id_or_url=camera_id_or_url)

    def test_model_on_video(self, model_path, video_path):
        self.model_tester.test_model_on_video(video_path=video_path, model_path=model_path, save_video=False)


def main():
    mode = inquirer.select(
        message="Was möchtest du tun?",
        choices=[
            {"name": "1. Keypoint-Datensatz aus Videos erstellen", "value": "create_ds"},
            {"name": "2. Modell trainieren (Strategie wählen)", "value": "train"},
            {"name": "3. Cross-Validation (Alle Modelle)", "value": "train_cv"},
            {"name": "4. Modell an Video testen", "value": "test_video"},
            {"name": "5. Live-Kamera Inferenz", "value": "test_cam"},
            {"name": "Beenden", "value": "exit"},
        ],
    ).execute()

    if mode == "exit":
        return

    # Strategie-Auswahl, wenn Modus 2 (Train) gewählt wurde
    train_strategy = None
    if mode == "train":
        train_strategy = inquirer.select(
            message="Wähle die Trainings-Strategie:",
            choices=[
                {"name": "Group Shuffle Split (Standard)", "value": "group_shuffle_split"},
                {"name": "Stratified Group K-Fold (SGKF)", "value": "sgkf"},
                {"name": "Group K-Fold (GKF)", "value": "gkf"},
                {"name": "Classic K-Fold", "value": "classic_kfold"},
                {"name": "Sklearn Simple Split", "value": "sklearn_simple"},
                {"name": "XGBoost mit Group K-Fold", "value": "xgb_gkf"},
                {"name": "XGBoost Simple Sklearn", "value": "xgb_simple"},
            ]
        ).execute()

    model_type = "tcn"
    if mode != "create_ds":
        model_type = inquirer.select(
            message="Modell-Architektur:",
            choices=["tcn", "gcn_paper", "cnn_lstm", "cnn_bilstm"],
            default="tcn"
        ).execute()

    app = BasketballGestureRecognitionApp(
        model_name=model_type,
        n_splits=5,
        fps=10,
        win_len_sec=1.6,
        stride_len_sec=1.6,
        num_classes=10,
        batch_size=32,
        lr=0.003,
        patience=8,
        max_epochs=100,
        include_conf=False,
        augment_data=True,
        path_to_videos="space_jam/examples"
    )

    if mode == "create_ds":
        app.create_dataset_from_videos()
    elif mode == "train":
        app.load_dataset_and_train(strategy=train_strategy)
    elif mode == "train_cv":
        app.train_all_models_cv()
    elif mode == "test_video":
        v_path = inquirer.text(message="Video Pfad:", default="videos/1080p_Mehmet_demo_video.mov").execute()
        m_path = inquirer.text(message="Modell Pfad:", default="models/bball_gesture_pose_7d35d102.keras").execute()
        app.test_model_on_video(m_path, v_path)
    elif mode == "test_cam":
        cam_url = inquirer.text(message="Kamera/Stream:", default="http://192.168.178.54:8080/video").execute()
        m_path = inquirer.text(message="Modell Pfad:", default="models/bball_gesture_pose_7d35d102.keras").execute()
        cam_source = int(cam_url) if cam_url.isdigit() else cam_url
        app.test_model_on_camera(m_path, cam_source)


if __name__ == "__main__":
    main()