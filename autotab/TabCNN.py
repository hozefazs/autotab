""" A CNN to classify 6 fret-string positions
    at the frame level during guitar performance
"""

from __future__ import print_function
import os
from os.path import join, dirname
from autotab.DataGenerator import DataGenerator
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, Lambda
from tensorflow.keras import backend as K
from tensorflow.python.keras.engine.sequential import clear_previously_created_nodes
import pandas as pd
import numpy as np
import datetime
from autotab.Metrics import *
from google.cloud import storage

from autotab.param import BUCKET_NAME, DATA_BUCKET_FOLDER, GCP, LOCAL_DATA

is_gcp = GCP  # boolean to know whether to use GCP buckets


class TabCNN:
    def __init__(
        self,
        batch_size=128,
        epochs=8,
        con_win_size=9,
        spec_repr="c",
        data_path=LOCAL_DATA,
        # data_path=f"gs://{BUCKET_NAME}/{DATA_BUCKET_FOLDER}/",
        id_file="id.csv",
        save_path="saved/",
    ):
        """The cross validation model

        Args:
            batch_size (int, optional): batch size. Defaults to 128.
            epochs (int, optional): number of times to run all batches. Defaults to 8.
            con_win_size (int, optional):  sliding window (I belive 9 because kernel is 3x3) Defaults to 9.
            spec_repr (str, optional): type of transform. Defaults to "c".
            data_path (str, optional): [description]. Defaults to "".
            id_file (str, optional): [description]. Defaults to "id.csv".
            save_path (str, optional): [description]. Defaults to "saved/".
        """
        # save_path=f"gs://{BUCKET_NAME}/saved/"):
        self.batch_size = batch_size
        self.epochs = epochs
        self.con_win_size = con_win_size
        self.spec_repr = spec_repr  #

        # initialize data path
        if is_gcp:
            print("Using Google cloud buckets")
            print(f"BUCKET_NAME {BUCKET_NAME}")
            print(f"DATA_BUCKET_FOLDER {DATA_BUCKET_FOLDER}")
            self.data_path = f"gs://{BUCKET_NAME}/{DATA_BUCKET_FOLDER}/"
        else:
            self.data_path = data_path
        print(self.data_path, flush=True)
        self.id_file = id_file
        if is_gcp:
            self.save_path = f"gs://{BUCKET_NAME}/saved/"
        else:
            self.save_path = save_path
        print(self.save_path, flush=True)

        # load all ids in the dataset
        self.load_IDs()

        #setup save folder and log file
        self.save_folder = (
            self.save_path + self.spec_repr + " " +
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "/")
        if not is_gcp:
            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder)
        else:
            self.check_gcp_save_folder()

        self.log_file = self.save_folder + "log.txt"

        # setup all the metrics required
        self.metrics = {}
        self.metrics["pp"] = []
        self.metrics["pr"] = []
        self.metrics["pf"] = []
        self.metrics["tp"] = []
        self.metrics["tr"] = []
        self.metrics["tf"] = []
        self.metrics["tdr"] = []
        self.metrics["data"] = [
            "g0", "g1", "g2", "g3", "g4", "g5", "mean", "std dev"
        ]

        # based on the type of repr, choose the input shape
        if self.spec_repr == "c":
            self.input_shape = (192, self.con_win_size, 1)
        elif self.spec_repr == "m":
            self.input_shape = (128, self.con_win_size, 1)
        elif self.spec_repr == "cm":
            self.input_shape = (320, self.con_win_size, 1)
        elif self.spec_repr == "s":
            self.input_shape = (1025, self.con_win_size, 1)

        # these probably won't ever change
        self.num_classes = 21  # 21 frets of the guitar
        self.num_strings = 6  # 6 strings on the guitar

    def check_gcp_save_folder(self):
        pass

    def load_IDs(self):
        """Loads the id.csv files with the name of the music files to fit/predict upon        """
        csv_file = self.data_path + self.id_file
        print(f"getting ids from {csv_file}", flush=True)
        self.list_IDs = list(pd.read_csv(csv_file, header=None)[0])

    def partition_data(self, data_split):
        """Function to create the training and validation split
            Note : our dataset has 6 set of files starting from 00 upto 05
            Args:
            data_split (int): int between 0 and 5  to determine validation set"""
        self.data_split = data_split
        self.partition = {}
        self.partition["training"] = []
        self.partition["validation"] = []
        for ID in self.list_IDs:  # for all lines in the id.csv file
            guitarist = int(
                ID.split("_")
                [0])  #getting the partition index (first 2 digits) of the file
            if guitarist == data_split:  # if the partition index equal the validation index
                self.partition["validation"].append(ID)  #add to validation set
            else:
                self.partition["training"].append(
                    ID)  # else add to training set

        self.training_generator = DataGenerator(  # training set generator
            self.partition["training"],
            data_path=self.data_path,
            batch_size=self.batch_size,
            shuffle=True,
            spec_repr=self.spec_repr,
            con_win_size=self.con_win_size,
        )

        self.validation_generator = DataGenerator(  # validation set generator
            self.partition["validation"],
            data_path=self.data_path,
            batch_size=len(self.partition["validation"]),
            shuffle=False,
            spec_repr=self.spec_repr,
            con_win_size=self.con_win_size,
        )
        # make a folder in saved mentioning this validation split
        self.split_folder = self.save_folder + str(self.data_split) + "/"
        if not is_gcp:
            if not os.path.exists(self.split_folder):
                os.makedirs(self.split_folder)
        else:
            self.check_gcp_split_folder()

    def check_gcp_split_folder(self):
        pass

    def log_model(self):
        if not is_gcp:
            logFileName = self.log_file  # if not gcp use originla filename
        else:
            logFileName = "tmp_log.txt"  # if gcp use a temp file

        with open(logFileName, "w") as fh:  # write to the file
            fh.write("\nbatch_size: " + str(self.batch_size))
            fh.write("\nepochs: " + str(self.epochs))
            fh.write("\nspec_repr: " + str(self.spec_repr))
            fh.write("\ndata_path: " + str(self.data_path))
            fh.write("\ncon_win_size: " + str(self.con_win_size))
            fh.write("\nid_file: " + str(self.id_file) + "\n")
            self.model.summary(print_fn=lambda x: fh.write(x + "\n"))

        if is_gcp:  # gcp needs the file created locally to then upload
            logBlob = storage.Client().bucket(BUCKET_NAME).blob(self.log_file)
            logBlob.upload_from_filename(  # this is using the tmp file
                logFileName)  # upload the local file made

    def softmax_by_string(self, t):
        """The activation function for the last layer of the model
            Args:
            t ([type]): the tensor that will be passed in by the model

        Returns:
            tensor: tensor determining 0,1 for each string
        """
        sh = K.shape(
            t)  # shape of the tensor passed by the model while training
        string_sm = []
        for i in range(self.num_strings):  # for all strings
            #provides a dimension in the tensor for each string
            string_sm.append(K.expand_dims(K.softmax(t[:, i, :]), axis=1))
        # returns concatenated tensor to fill in for each string on/off (1,0)
        return K.concatenate(string_sm, axis=1)

    def catcross_by_string(self, target, output):
        """Categorical Cross Entropy for each string

            Args:
            target ([type]): Will be provided by model during training
            output ([type]): Will be provided by model during training

            Returns:
            float: total categorical cross entropy loss summed over all 6 strings"""
        loss = 0
        for i in range(self.num_strings):  #for all strings
            # add their individual cross entropy losses
            loss += K.categorical_crossentropy(target[:, i, :], output[:,
                                                                       i, :])
        return loss

    def avg_acc(self, y_true, y_pred):
        """the accuracy calculation for this training

            Args:
            y_true ([type]): y_true provided by the model
            y_pred ([type]): Y_pred predicted by the model

            Returns:
            float: accuracy of the prediction"""
        return K.mean(  # the mean of
            # the scores everytime a prediction is true over all strings
            K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)))

    def build_model(self):
        model = Sequential()
        model.add(
            Conv2D(32,
                   kernel_size=(3, 3),
                   activation="relu",
                   input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes * self.num_strings))  # no activation
        model.add(Reshape((self.num_strings, self.num_classes)))
        model.add(Activation(self.softmax_by_string)
                  )  #adds an activation function at the end for the 6 strings

        model.compile(loss=self.catcross_by_string,
                      optimizer=Adadelta(),
                      metrics=[self.avg_acc])

        self.model = model
        return model

    def train(self):
        """training the model
        """
        self.model.fit_generator(
            generator=self.training_generator,
            validation_data=None,
            epochs=self.epochs,
            verbose=1,
            use_multiprocessing=True,
            workers=9,
        )

    def save_weights(self):
        """Save the model/"weights"
        """
        if not is_gcp:
            save_file_name = (self.split_folder + "weights.h5"
                              )  # if not gcp use original filename
        else:
            save_file_name = "tmp_save.h5"  # if gcp use a temp file
        self.model.save_weights(save_file_name)

        if is_gcp:  # gcp needs the file created locally to then upload
            save_blob = (
                storage.Client().bucket(BUCKET_NAME).blob(self.split_folder +
                                                          "weights.h5"))
            save_blob.upload_from_filename(  # this is using the tmp file
                save_file_name)  # upload the local file made

    def test(self):
        """do a prediction on the first item in the validation set
        """
        self.X_test, self.y_gt = self.validation_generator[0]
        self.y_pred = self.model.predict(self.X_test)

    def save_predictions(self):
        """and then save that prediction
        """
        if not is_gcp:
            save_file_name = (self.split_folder + "predictions.npz"
                              )  # if not gcp use original filename
        else:
            save_file_name = "tmp_save.npz"  # if gcp use a temp file
        np.savez(save_file_name, y_pred=self.y_pred, y_gt=self.y_gt)

        if is_gcp:  # gcp needs the file created locally to then upload
            save_blob = (
                storage.Client().bucket(BUCKET_NAME).blob(self.split_folder +
                                                          "predictions.npz"))
            save_blob.upload_from_filename(  # this is using the tmp file
                save_file_name)  # upload the local file made

    def evaluate(self):
        """do an evaluation on the prediction
        """
        self.metrics["pp"].append(pitch_precision(self.y_pred, self.y_gt))
        self.metrics["pr"].append(pitch_recall(self.y_pred, self.y_gt))
        self.metrics["pf"].append(pitch_f_measure(self.y_pred, self.y_gt))
        self.metrics["tp"].append(tab_precision(self.y_pred, self.y_gt))
        self.metrics["tr"].append(tab_recall(self.y_pred, self.y_gt))
        self.metrics["tf"].append(tab_f_measure(self.y_pred, self.y_gt))
        self.metrics["tdr"].append(tab_disamb(self.y_pred, self.y_gt))

    def save_results_csv(self):
        """Save all the metrics to results.csv
        """
        output = {}
        for key in self.metrics.keys():
            if key != "data":
                vals = self.metrics[key]
                mean = np.mean(vals)
                spec_repr = "c"
                std = np.std(vals)
                output[key] = vals + [mean, std]
        output["data"] = self.metrics["data"]
        df = pd.DataFrame.from_dict(output)
        df.to_csv(self.save_folder + "results.csv")


if __name__ == "__main__":
    ##################################
    ########### EXPERIMENT ###########
    ##################################
    tabcnn = TabCNN()
    # tabcnn.clear_previously_created_nodes()

    print("logging model...")
    tabcnn.build_model()
    tabcnn.log_model()

    # Note : our dataset has 6 set of files starting from 00 upto 05
    for fold in range(6):  # The 'fold' is used as the validation set
        print("\nfold " + str(fold))  #printing the fold number
        tabcnn.partition_data(
            fold)  # partitionig of data, with the fold used as validation
        print("building model...")
        tabcnn.build_model()  #building the model to do the training
        print("training...")
        tabcnn.train()
        tabcnn.save_weights()
        print("testing...")
        tabcnn.test()
        tabcnn.save_predictions()
        print("evaluation...")
        tabcnn.evaluate()
    print("saving results...")
    tabcnn.save_results_csv()
