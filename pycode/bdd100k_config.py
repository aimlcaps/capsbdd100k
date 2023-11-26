
class BDDDatasetConfig:
    def __init__(self,image_source_root,label_source_root,image_save_root,label_save_root,img_size):
        self.IMAGE_SOURCE_ROOT = image_source_root
        self.LABEL_SOURCE_ROOT = label_source_root
        self.IMAGE_SAVE_ROOT = image_save_root
        self.LABEL_SAVE_ROOT = label_save_root
        self.IMAGE_SOURCE_PATH_TRAIN= self.IMAGE_SOURCE_ROOT + '/train'
        self.LABEL_SOURCE_PATH_TRAIN= self.LABEL_SOURCE_ROOT + '/train'
        self.IMAGE_SAVE_PATH_TRAIN= self.IMAGE_SAVE_ROOT + '/train'
        self.LABEL_SAVE_PATH_TRAIN= self.LABEL_SAVE_ROOT + '/train'

        self.IMAGE_SOURCE_PATH_VAL= self.IMAGE_SOURCE_ROOT + '/val'
        self.LABEL_SOURCE_PATH_VAL= self.LABEL_SOURCE_ROOT + '/val'
        self.IMAGE_SAVE_PATH_VAL= self.IMAGE_SAVE_ROOT + '/val'
        self.LABEL_SAVE_PATH_VAL= self.LABEL_SAVE_ROOT + '/val'

        self.IMAGE_SOURCE_PATH_TEST= self.IMAGE_SOURCE_ROOT + 'test'
        self.LABEL_SOURCE_PATH_TEST= self.LABEL_SOURCE_ROOT + 'test'
        self.IMAGE_SAVE_PATH_TEST= self.IMAGE_SAVE_ROOT + 'test'
        self.LABEL_SAVE_PATH_TEST= self.LABEL_SAVE_ROOT + 'test'

        self.IMAGE_SIZE = img_size

        self.class_labels = {'car': 0,
                        'bus': 1,
                        'person': 2,
                        'bike': 3,
                        'truck': 4,
                        'motor': 5,
                        'train': 6,
                        'rider': 7,
                        'traffic sign': 8,
                        'traffic light': 9}

    IMAGE_SOURCE_ROOT= r'S:\IIITH\Capstone\datasets\bdd100k\images\100k'
    LABEL_SOURCE_ROOT= r'S:\IIITH\Capstone\datasets\bdd100k_labels_release\bdd100k\labels'
    IMAGE_SAVE_ROOT= r'S:\IIITH\Capstone\datasets\bdd100k\images\100k'
    LABEL_SAVE_ROOT= r'S:\IIITH\Capstone\datasets\bdd100k\labels\100k'

    IMAGE_SOURCE_PATH_TRAIN= IMAGE_SOURCE_ROOT + '/train'
    LABEL_SOURCE_PATH_TRAIN= LABEL_SOURCE_ROOT + '/train'
    IMAGE_SAVE_PATH_TRAIN= IMAGE_SAVE_ROOT + '/train'
    LABEL_SAVE_PATH_TRAIN= LABEL_SAVE_ROOT + '/train'

    IMAGE_SOURCE_PATH_VAL= IMAGE_SOURCE_ROOT + '/val'
    LABEL_SOURCE_PATH_VAL= LABEL_SOURCE_ROOT + '/val'
    IMAGE_SAVE_PATH_VAL= IMAGE_SAVE_ROOT + '/val'
    LABEL_SAVE_PATH_VAL= LABEL_SAVE_ROOT + '/val'

    IMAGE_SOURCE_PATH_TEST= IMAGE_SOURCE_ROOT + 'test'
    LABEL_SOURCE_PATH_TEST= LABEL_SOURCE_ROOT + 'test'
    IMAGE_SAVE_PATH_TEST= IMAGE_SAVE_ROOT + 'test'
    LABEL_SAVE_PATH_TEST= LABEL_SAVE_ROOT + 'test'

    IMAGE_SIZE = (1280, 720)

    class_labels = {'car': 0,
                'bus': 1,
                'person': 2,
                'bike': 3,
                'truck': 4,
                'motor': 5,
                'train': 6,
                'rider': 7,
                'traffic sign': 8,
                'traffic light': 9}