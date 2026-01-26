from KeypointDatasetProcessor import KeypointDatasetProcessor
from ModelCreator import ModelCreator
from ModelTrainer import ModelTrainer
from KeypointDatasetCreator import KeypointDatasetCreator
from ModelTester import ModelTester

class BasketballGestureRecognitionApp:
    def __init__(self, model_name, n_splits, fps, win_len_sec, stride_len_sec, num_classes, path_to_videos,
                 include_conf, batch_size=64, lr=0.0001, patience=20):
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

        self.keypt_processor = KeypointDatasetProcessor(label_id_dic=self.label_id_dic, fps=self.fps)
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
        self.keypt_processor.load_dataset(".", include_conf=self.include_conf)
        # self.keypt_processor.print_data_after_preparation()
        self.model_trainer = ModelTrainer(
            model_creator=self.model_creator,
            keypt_processor=self.keypt_processor,
            label_id_dic=self.label_id_dic,
            random_state=25,
            batch_size=self.batch_size,
            lr=self.lr,
            patience=self.patience,
            num_classes=self.num_classes
        )
        # self.model_trainer.train_model_xgb_gkf(n_splits=self.n_splits)
        # self.model_trainer.train_model_sgkf(n_splits=self.n_splits)
        # self.model_trainer.train_model_gkf(n_splits=self.n_splits)
        # self.model_trainer.train_model_classic_kfold(n_splits=self.n_splits)

        # self.model_trainer.train_model_sklearn_simple(test_size=0.2)

        self.model_trainer.train_model_group_shuffle_split(test_size=0.1, n_splits=1)

        # self.model_trainer.train_model_xgb_simple_sklearn(test_size=0.2)

    def test_model(self, model_path):
        self.model_tester.test_model_on_video(
            video_path="videos/1080p_Mehmet_demo_video.mov",
            model_path=model_path)


if __name__ == "__main__":
    #TODO: batch_size und lr hinzufügen und bei test videos oben bei der funktion auch videlink in aufruf
    app = BasketballGestureRecognitionApp(
        model_name="gcn",
        n_splits=2,
        fps=10,
        win_len_sec=1.6,
        stride_len_sec=1.6,
        num_classes=10,
        batch_size=64,
        lr=0.0001,
        patience=20,
        include_conf=False,
        path_to_videos="space_jam/examples")
    # app.create_dataset_from_videos()
    app.load_dataset_and_train()
    # app.test_model("models/bball_gesture_pose_8ac559eb.keras")
