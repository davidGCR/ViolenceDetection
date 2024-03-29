import os
PATH_HOCKEY_FRAMES_VIOLENCE = '/media/david/datos/Violence DATA/HockeyFights/frames/violence'
PATH_HOCKEY_FRAMES_NON_VIOLENCE = '/media/david/datos/Violence DATA/HockeyFights/frames/nonviolence'
PATH_VIOLENTFLOWS_FRAMES = '/media/david/datos/Violence DATA/violentflows/movies Frames'
PATH_CHECKPOINTS_DI = 'checkpoints/di'
PATH_CHECKPOINTS_MASK = 'checkpoints/masked'
PATH_LEARNING_CURVES_DI = 'learningCurves/di'
PATH_LEARNING_CURVES_MASK = 'learningCurves/masked'
PATH_SALIENCY_MODELS = 'SALIENCY/Models'
PATH_BLACK_BOX_MODELS = 'BlackBoxModels'
LABEL_PRODUCTION_MODEL = '-PRODUCT'
PATH_SALIENCY_DATASET = 'DATA/saliency'
PATH_UCFCRIME2LOCAL_VIDEOS = '/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local/videos'
PATH_UCFCRIME2LOCAL_FRAMES = '/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local/frames'
PATH_UCFCRIME2LOCAL_README = '/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local/readme'
PATH_UCFCRIME2LOCAL_BBOX_ANNOTATIONS = '/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local/readme/Txt annotations'
PATH_UCFCRIME2LOCAL_FRAMES_REDUCED = '/media/david/datos/Violence DATA/AnomalyCRIME/UCFCrime2Local/frames_reduced'

ANOMALY_PATH_CHECKPOINTS = 'ANOMALYCRIME/checkpoints'
ANOMALY_PATH_LEARNING_CURVES = 'ANOMALYCRIME/learning_curves'
ANOMALY_PATH_TRAIN_SPLIT = os.path.join(PATH_UCFCRIME2LOCAL_README, 'Train_split_AD.txt')
ANOMALY_PATH_TEST_SPLIT = os.path.join(PATH_UCFCRIME2LOCAL_README, 'Test_split_AD.txt')
ANOMALY_PATH_SALIENCY_MODELS = 'SALIENCY/Models/anomaly'
ANOMALY_PATH_BLACK_BOX_MODELS = 'SALIENCY/BlackBoxModels/anomaly'

OPERATION_TRAINING = 'training'
OPERATION_TRAINING_FINAL = 'trainingFinal'
OPERATION_TESTING = 'testing'

JOIN_TEMP_MAX_POOL = 'tempMaxPool'
JOIN_CONCATENATE = 'cat'

# frame, flac, xmin, ymin, xmax, ymax
IDX_FRAME_NAME = 0
IDX_FLAC = 1
IDX_XMIN = 2
IDX_YMIN = 3
IDX_XMAX = 4
IDX_YMAX = 5

RED = 'r'
GREEN = 'g'
CYAN = 'c'
PIL_WHITE = (255, 255, 255)
PIL_RED = (255, 0, 0)