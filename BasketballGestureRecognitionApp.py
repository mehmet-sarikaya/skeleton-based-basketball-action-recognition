import traceback

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

        self.label_id_dic_old = {
            "dribbling": 0,
            "passing": 1,
            "throw": 2
        }

        self.label_id_dic = {0: "block",
                             1: "pass",
                             2: "run",
                             3: "dribble",
                             4: "shoot",
                             5: "ball in hand",
                             6: "defense",
                             7: "pick",
                             8: "no_action",
                             9: "walk"
                             }
        # 10: discard

        self.augment_data = augment_data
        self.keypt_processor = KeypointDatasetProcessor(label_id_dic=self.label_id_dic, fps=self.fps,
                                                        random_state=self.random_state)

        self.model_name = model_name
        self.model_creator = ModelCreator(
            model_name=self.model_name,
            win_len_sec=self.win_len_sec,
            fps=self.fps,
            num_classes=self.num_classes,
            include_conf=include_conf)
        self.model_trainer = None
        self.model_tester = ModelTester(
            app=self
        )

    def create_dataset_from_videos(self):
        self.keypt_dataset_creator.create_keypoints_dataset(self.path_to_videos, filter_videos=False)

    def load_dataset_and_train(self):
        self.keypt_processor.load_dataset(".", include_conf=self.include_conf, augment_data=self.augment_data)
        # self.keypt_processor.print_data_after_preparation()
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
        # self.model_trainer.train_model_xgb_gkf(n_splits=self.n_splits)
        # self.model_trainer.train_model_sgkf(n_splits=self.n_splits)
        # self.model_trainer.train_model_gkf(n_splits=self.n_splits)
        # self.model_trainer.train_model_classic_kfold(n_splits=self.n_splits)

        # self.model_trainer.train_model_sklearn_simple(test_size=0.2)

        self.model_trainer.train_model_group_shuffle_split(test_size=0.1, n_splits=1)

        # self.model_trainer.train_model_xgb_simple_sklearn(test_size=0.2)

    def train_all_models_cv(self):
        # Liste deiner Modell-Architekturen
        models_to_train = ["gcn_paper", "tcn", "cnn_lstm", "cnn_bilstm"]

        # 1. Daten einmalig laden
        self.keypt_processor.load_dataset(".", include_conf=self.include_conf, augment_data=self.augment_data)

        for model_name in models_to_train:
            print("\n" + "#" * 60)
            print(f"### STARTING CROSS-VALIDATION FOR: {model_name} ###")
            print("#" * 60 + "\n")

            self.model_creator.model_name = model_name

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
                print(f"\nFinished all folds for {model_name}.")
            except Exception as e:
                print(f"\nError during CV for {model_name}: {e}")
                traceback.print_exc()

        print("\n" + "=" * 60)
        print("COMPLETED CROSS-VALIDATION FOR ALL MODELS.")
        print("=" * 60)

    def test_model_on_camera(self, model_path, camera_id_or_url=0):
        self.model_tester.test_model_on_camera(model_path=model_path, camera_id_or_url=camera_id_or_url)

    def test_model_on_video(self, model_path, video_path):

        self.model_tester.test_model_on_video(
            video_path=video_path,
            model_path=model_path,
            save_video=False
        )


if __name__ == "__main__":
    #TODO: batch_size und lr hinzufügen und bei test videos oben bei der funktion auch videlink in aufruf
    app = BasketballGestureRecognitionApp(
        model_name="tcn",
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
        path_to_videos="space_jam/examples")

    # app.create_dataset_from_videos()
    app.load_dataset_and_train()

    videos = ["videos/1080p_Mehmet_demo_video.mov", "videos/1v1.mov"]

    video_path = videos[0]

    # Modell Testen auf einem Video
    # app.test_model_on_video("models/bball_gesture_pose_7d35d102.keras",  # bisher bestes Modell
    #                         video_path=video_path)

    # Modell Testen über Normal Kamera USB (funktioniert in WSL nicht, lieber über Stream unten)
    # app.test_model_on_camera(model_path="models/bball_gesture_pose_7d35d102.keras",
    #                          camera_id_or_url=0)  # bisher bestes Modell

    Camera_stream_url = "http://134.103.169.216:8080/video"

    # Modell Testen über Kamera Stream
    # app.test_model_on_camera(model_path="models/bball_gesture_pose_7d35d102.keras",
    #                          camera_id_or_url=Camera_stream_url)  # bisher bestes Modell

    # TCN MODELL testen auf Kamera Stream
    # tcn_path = "evaluations/tcn/20260129-123200/bball_gesture_pose_3feb2c81/bball_gesture_pose_3feb2c81.keras"
    # app.test_model_on_camera(tcn_path, camera_id_or_url=Camera_stream_url)

    # Ein GCN Modell das auf fast allen Daten traininiert wurde

    model_path = "evaluations/gcn/20260129-022130/bball_gesture_pose_a0049b48/bball_gesture_pose_a0049b48.keras"
    # app.test_model_on_video(model_path=model_path, video_path="video_path")

    # Group K Fold auf mehreren Modellen auf einmal
    # app.train_all_models_cv()
