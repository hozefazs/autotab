import os
import numpy as np
import jams
from scipy.io import wavfile
import sys
import librosa
from tensorflow.keras.utils import to_categorical


class TabDataReprGen:
    def __init__(self, mode="c"):
        # file path to the GuitarSet dataset
        path = "data/GuitarSet/"
        self.path_audio = path + "audio/audio_mic/"
        self.path_anno = path + "annotation/"

        # labeling parameters
        self.string_midi_pitches = [40, 45, 50, 55, 59, 64]
        self.highest_fret = 19
        self.num_classes = self.highest_fret + 2  # for open/closed

        # prepresentation and its labels storage
        self.output = {}

        # preprocessing modes
        #
        # c = cqt
        # m = melspec
        # cm = cqt + melspec
        # s = stft
        #
        self.preproc_mode = mode    # Preprocessing mode for the wav file data
        self.downsample = True      # Select to lower sample rate of data
        self.normalize = True       # Select to normalize data
        self.sr_downs = 22050       # Lowered sample rate

        # CQT parameters
        self.cqt_n_bins = 192           # Number of bins for the constant-Q transform "c"
        self.cqt_bins_per_octave = 24   # Number of bins per octave

        # STFT parameters
        self.n_fft = 2048       # Length of the FFT window
        self.hop_length = 512   # Number of samples between successive frames

        # save file path
        self.save_path = "data/spec_repr/" + self.preproc_mode + "/"

    def load_rep_and_labels_from_raw_file(self, filename):
        """
        Loads wav and jams files, reads wav file and creates sample rate [int]
        and data [np.array].
        Constructs, cleans, and categorizes labels and stores them in output dict
        Returns the number of frames
        """
        file_audio = self.path_audio + filename + "_mic.wav"    # wav file
        file_anno = self.path_anno + filename + ".jams"         # jams file
        jam = jams.load(file_anno)                              # loads jams file
        self.sr_original, data = wavfile.read(file_audio)       # creates sample rate [int] and data from wav file
        self.sr_curr = self.sr_original

        # preprocess audio, store in output dict
        self.output["repr"] = np.swapaxes(self.preprocess_audio(data), 0, 1)
        print(self.output['repr'].shape)

        # construct labels
        frame_indices = range(len(self.output["repr"]))  # Counts the frames
        times = librosa.frames_to_time( # Converts frame counts to time (seconds)
            frame_indices,
            sr=self.sr_curr,            # Sample rate
            hop_length=self.hop_length  # Number of samples between successive frames
            )

        # loop over all strings and sample annotations
        labels = []
        for string_num in range(6):
            anno = jam.annotations["note_midi"][string_num]
            string_label_samples = anno.to_samples(times)
            # replace midi pitch values with fret numbers
            for i in frame_indices:
                if string_label_samples[i] == []:
                    string_label_samples[i] = -1
                else:
                    string_label_samples[i] = int(
                        round(string_label_samples[i][0]) -
                        self.string_midi_pitches[string_num])
            labels.append([string_label_samples])

        labels = np.array(labels)       # Creates np.array out of labels list
        # remove the extra dimension
        labels = np.squeeze(labels)
        labels = np.swapaxes(labels, 0, 1)

        # clean labels
        labels = self.clean_labels(labels) # Returns an array of the labels with
        # the correct string numbering and categorized according to the number of classes defined

        # store and return
        self.output["labels"] = labels # Stores the cleaned labels in output dict
        return len(labels)

    def correct_numbering(self, n):
        """
        Adds +1 to correct the string number
        """
        n += 1
        if n < 0 or n > self.highest_fret:
            n = 0
        return n

    def categorical(self, label):
        """
        Categorizes the label in the number of classes defined
        (highest_fret (19) + 2  # for open/closed)
        """
        return to_categorical(label, self.num_classes)

    def clean_label(self, label):
        """
        Takes the label, corrects the string numbering and categorizes the label
        using to_categorical.
        Returns categorized and clean label
        """
        label = [self.correct_numbering(n) for n in label]
        return self.categorical(label)

    def clean_labels(self, labels):
        """
        Returns an array of all the cleaned labels with the correct string numbering
        and categorized according to the number of classes defined
        """
        return np.array([self.clean_label(label) for label in labels])

    def preprocess_audio(self, data):
        """
        Preprocesses data depending on mode selected using librosa.
        It converts data to float, then it normalizes it and resamples it
        to a lower sample rate. Then, preprocesses it and returns the processed data
            Args:
                data ([np.array]): [data created by wavfile.read]
            Returns:
                [np.ndarrray[shape=(n_bins, t)]]: [preprocessed data array]
        """
        data = data.astype(float)
        if self.normalize:
            data = librosa.util.normalize(data)
        if self.downsample:
            data = librosa.resample(data, self.sr_original, self.sr_downs)
            self.sr_curr = self.sr_downs
        if self.preproc_mode == "c":
            data = np.abs(
                librosa.cqt(data,     # Computes the constant-Q transform of an audio signal
                            hop_length=self.hop_length,
                            sr=self.sr_curr,        # data sample rate
                            n_bins=self.cqt_n_bins,
                            bins_per_octave=self.cqt_bins_per_octave))
        elif self.preproc_mode == "m":
            data = librosa.feature.melspectrogram(y=data,
                                                  sr=self.sr_curr,
                                                  n_fft=self.n_fft,
                                                  hop_length=self.hop_length)
        elif self.preproc_mode == "cm":
            cqt = np.abs(
                librosa.cqt(data,
                            hop_length=self.hop_length,
                            sr=self.sr_curr,
                            n_bins=self.cqt_n_bins,
                            bins_per_octave=self.cqt_bins_per_octave))
            mel = librosa.feature.melspectrogram(y=data,
                                                 sr=self.sr_curr,
                                                 n_fft=self.n_fft,
                                                 hop_length=self.hop_length)
            data = np.concatenate((cqt, mel), axis=0)
        elif self.preproc_mode == "s":
            data = np.abs(
                librosa.stft(data,
                             n_fft=self.n_fft,
                             hop_length=self.hop_length))
        else:
            print("invalid representation mode.")

        return data

    def save_data(self, filename):
        """
        Saves the generated data output dictionary into an npz file
        """
        np.savez(filename, **self.output)

    def get_nth_filename(self, n):
        """
        Sorts the jams files in the directory, looks for the nth one,
        removes the .jams extension and returns only the filename
            Returns:
                [str]: [filename]
        """
        filenames = np.sort(np.array(os.listdir(self.path_anno)))
        filenames = list(filter(lambda x: x[-5:] == ".jams", filenames))
        print(filenames[n])
        return filenames[n][:-5]

    def load_and_save_repr_nth_file(self, n):
        """
        Gets the filename, preprocesses it, and gets the number of frames.
        Saves the file as an npz
        """

        filename = self.get_nth_filename(n)     # Gets only filename with no .jams extension
        print(filename)
        num_frames = self.load_rep_and_labels_from_raw_file(filename)
        print("done: " + filename + ", " + str(num_frames) + " frames")
        save_path = self.save_path
        if not os.path.exists(save_path):               # Creates saving path if it does not exist
            os.makedirs(save_path)
        self.save_data(save_path + filename + ".npz")   # Saves generated output dictionary in an npz file

    def load_rep_from_raw_file(self, filename):
        """Function to generate x_new data from a wave file
            to feed into the prediction model
                Args:
                filename (string): location of the file to be pre-processed
                Returns:
                [numpy.ndArray]: a numpy array of shape
                num frames x 192 x 9 (con_win) x 1
            """
        self.con_win_size = 9
        self.half_win = self.con_win_size // 2
        file_audio = filename
        self.sr_original, data = wavfile.read(file_audio)
        self.sr_curr = self.sr_original
        # preprocess audio, store in output dict
        repr = np.swapaxes(self.preprocess_audio(data), 0, 1)
        full_x = np.pad(
            repr,
            [
                (self.half_win, self.half_win
                 ),  # full x is the entire song padded with halfwin*2 frames
                (0, 0)
            ],
            mode='constant')
        x_new = []
        for frame_idx in range(0,
                               len(repr)):  # for all frames in the experiment
            sample_x = full_x[frame_idx:frame_idx + self.con_win_size]
            x_new.append(np.expand_dims(np.swapaxes(sample_x, 0, 1), -1))
        x_new = np.array(x_new, dtype='float32')
        return x_new

def main(args):
    """
    Gets the index of the file (m) and the preprocessing mode (n)
    Does the processing and saves it as an npz
    """
    n = args[0]
    m = args[1]
    gen = TabDataReprGen(mode=m)
    gen.load_and_save_repr_nth_file(n)


if __name__ == "__main__":
    #for index in range(361):
    #    main([index, 'c'])
    main([0,"c"])
