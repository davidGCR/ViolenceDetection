
# import sys
# sys.path.insert(1, '/media/david/datos/PAPERS-SOURCE_CODE/MyCode')
import ANOMALYCRIME.anomaly_dataset as anomaly_dataset
# import anomaly_dataset
import util
import random
import os
import constants
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

def initialize_final_only_test_anomaly_dataset(path_dataset, train_videos_path, test_videos_path, dataset_source, batch_size, num_workers, numDiPerVideos, transforms_t, maxNumFramesOnVideo, videoSegmentLength, positionSegment, shuffle):
     test_names, test_labels, test_num_frames, test_bbox_files = anomaly_dataset.only_anomaly_test_videos(test_videos_path, path_dataset)
     test_labels = anomaly_dataset.labels_2_binary(test_labels)
     util.print_balance(test_labels, 'test')
     image_datasets = {
          "test": anomaly_dataset.AnomalyDataset( dataset=test_names, labels=test_labels, numFrames=test_num_frames, bbox_files=test_bbox_files, spatial_transform=transforms_t["test"], source=dataset_source,
               nDynamicImages=numDiPerVideos, maxNumFramesOnVideo=maxNumFramesOnVideo, videoSegmentLength=videoSegmentLength, positionSegment=positionSegment)
     }
     dataloaders_dict = {
          "test": torch.utils.data.DataLoader( image_datasets["test"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
     }
     return dataloaders_dict, test_names

def initialize_final_anomaly_dataset(path_dataset, train_videos_path, test_videos_path, dataset_source, batch_size, num_workers, numDiPerVideos, transforms_t, maxNumFramesOnVideo, videoSegmentLength, positionSegment, shuffle):
     train_names, train_labels, train_num_frames, test_names, test_labels, test_num_frames = anomaly_dataset.train_test_videos(train_videos_path, test_videos_path, path_dataset)
     combined = list(zip(train_names, train_labels, train_num_frames))
     random.shuffle(combined)

     train_names[:], train_labels[:], train_num_frames = zip(*combined)
     train_labels = anomaly_dataset.labels_2_binary(train_labels)
     test_labels = anomaly_dataset.labels_2_binary(test_labels)
     util.print_balance(train_labels, 'train')
     util.print_balance(test_labels, 'test')
     image_datasets = {

          "train": anomaly_dataset.AnomalyDataset( dataset=train_names, labels=train_labels, numFrames=train_num_frames, spatial_transform=transforms_t["train"], source=dataset_source,
               nDynamicImages=numDiPerVideos, maxNumFramesOnVideo=maxNumFramesOnVideo, videoSegmentLength=videoSegmentLength, positionSegment=positionSegment),
          "test": anomaly_dataset.AnomalyDataset( dataset=test_names, labels=test_labels, numFrames=test_num_frames, spatial_transform=transforms_t["test"], source=dataset_source,
               nDynamicImages=numDiPerVideos, maxNumFramesOnVideo=maxNumFramesOnVideo, videoSegmentLength=videoSegmentLength, positionSegment=positionSegment)
     }
     dataloaders_dict = {
          "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers),
          "test": torch.utils.data.DataLoader( image_datasets["test"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
     }
     return dataloaders_dict, test_names

def initialize_train_val_anomaly_dataset(path_dataset, train_videos_path, test_videos_path, dataset_source, batch_size, num_workers, numDiPerVideos, transforms, maxNumFramesOnVideo, videoSegmentLength, positionSegment, shuffle):
     train_names, train_labels, train_num_frames, test_names, test_labels, test_num_frames = anomaly_dataset.train_test_videos(train_videos_path, test_videos_path, path_dataset)
     # print(train_multilabels)
     combined = list(zip(train_names, train_num_frames))
     #     print(combined)
     #     print('='*100)
     combined_train_names, combined_val_names, train_labels, val_labels = train_test_split(combined, train_labels, stratify=train_labels, test_size=0.30)
     train_names, train_num_frames = zip(*combined_train_names)
     val_names, val_num_frames = zip(*combined_val_names)
     train_labels = anomaly_dataset.labels_2_binary(train_labels)
     val_labels = anomaly_dataset.labels_2_binary(val_labels)
     test_labels = anomaly_dataset.labels_2_binary(test_labels)
     util.print_balance(train_labels, 'train')
     util.print_balance(val_labels, 'val')
     util.print_balance(test_labels, 'test')
     # combined = list(zip(train_names, train_labels, train_num_frames))
     # random.shuffle(combined)
     # train_names[:], train_labels[:], train_num_frames = zip(*combined)
     #     combined = list(zip(test_names, test_labels, test_num_frames))
     #     random.shuffle(combined)
     #     test_names[:], test_labels[:], test_num_frames = zip(*combined)
     
     # print(len(train_names), len(train_labels), len(train_num_frames), len(test_names), len(test_labels), len(test_num_frames))
     
     
     image_datasets = {
          "train": anomaly_dataset.AnomalyDataset( dataset=train_names, labels=train_labels, numFrames=train_num_frames, spatial_transform=transforms["train"], source=dataset_source,
               nDynamicImages=numDiPerVideos, maxNumFramesOnVideo=maxNumFramesOnVideo, videoSegmentLength=videoSegmentLength, positionSegment=positionSegment),
          "val": anomaly_dataset.AnomalyDataset( dataset=val_names, labels=val_labels, numFrames=val_num_frames, spatial_transform=transforms["val"], source=dataset_source,
               nDynamicImages=numDiPerVideos, maxNumFramesOnVideo=maxNumFramesOnVideo, videoSegmentLength=videoSegmentLength, positionSegment=positionSegment),
          "test": anomaly_dataset.AnomalyDataset( dataset=test_names, labels=test_labels, numFrames=test_num_frames, spatial_transform=transforms["test"], source=dataset_source,
               nDynamicImages=numDiPerVideos, maxNumFramesOnVideo=maxNumFramesOnVideo, videoSegmentLength=videoSegmentLength, positionSegment=positionSegment)
     }
     dataloaders_dict = {
          "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers),
          "val": torch.utils.data.DataLoader( image_datasets["val"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers),
          "test": torch.utils.data.DataLoader( image_datasets["test"], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
     }
     return dataloaders_dict, test_names