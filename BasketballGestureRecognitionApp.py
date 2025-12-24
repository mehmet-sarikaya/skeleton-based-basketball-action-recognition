from KeypointDatasetProcessor import KeypointDatasetProcessor
from ModelCreator import ModelCreator
from ModelTrainer import ModelTrainer
from KeypointDatasetCreator import KeypointDatasetCreator

class BasketballGestureRecognitionApp:
    def __init__(self, fps, win_len_sec, num_classes, path_to_videos):
        self.fps = fps
        self.win_len_sec = win_len_sec
        self.num_classes = num_classes
        self.path_to_videos = path_to_videos
        self.keypt_dataset_creator = KeypointDatasetCreator()

        self.label_id_dic = {
            "dribbling": 0,
            "passing": 1,
            "throw": 2
        }

        self.keypt_processor = KeypointDatasetProcessor(label_id_dic=self.label_id_dic,fps=self.fps)
        self.model_creator = ModelCreator(duration_sec=3, fps=self.fps, num_classes=self.num_classes)
        self.model_trainer = None

    def create_dataset_from_videos(self):
        self.keypt_dataset_creator.create_keypoints_dataset(self.path_to_videos)

    def load_dataset_and_train(self):
        self.keypt_processor.load_all_keypoints_data_in_dir(self.path_to_videos)
        self.keypt_processor.prepare_data_for_training(win_len_sec=self.win_len_sec)
        self.keypt_processor.print_data_after_preparation()
        self.model_trainer = ModelTrainer(
            model_creator=self.model_creator,
            keypt_processor=self.keypt_processor,
            label_id_dic = self.label_id_dic
        )
        self.model_trainer.train_model_gkf()


if __name__ == "__main__":
    app = BasketballGestureRecognitionApp(fps=10, win_len_sec=3, num_classes=3, path_to_videos="videos")
    app.load_dataset_and_train()
