import autotab.TabCNN as TabCNN
import pandas as pd
from autotab.param import LOCAL_MODEL
from autotab.TabDataReprGen import TabDataReprGen
"""Script to predict the tabs from a model"""
"""
#### PSEUDOCODE
1. load the created model into a model object
2. load the wave file to run prediction upon
3. preprocess the wave file to get repr
4. labels = model.predict(repr)
5. make labels into Dataframe and print
6. squeeze the frames into fret changes
    i.e. keep the frame if fret changed OR
         keep the frame if no fret changes for 3 frames i.e 0.6 secs
         else remove the frame
7. display the squeezed labels dataframe
8. visualize the squeezed labels into a tablature format
9. Save the tablature to a file

"""


def make_empty_tab():
    tab_dict = {
        'E': [],
        'A': [],
        'D': [],
        'G': [],
        'B': [],
        'e': [],
    }
    tablature = pd.DataFrame.from_dict(tab_dict)
    tablature = tablature.T
    return tablature


def make_full_tab(labels, num_frames=99):
    tablature = make_empty_tab()
    # loop over all frames to add for all frame (lets test for num_frames frames)
    for frame_idx in range(0, num_frames):
        frame = pd.DataFrame(
            labels[frame_idx])  # loading frame frame_idx into a dataframe
        # now set (Frame, 0 Fret, where value < 0.9) set as 0
        # frame[0].replace(to_replace=list(frame[0][frame[0] < 0.9]),
        #                  value=0,
        #                  inplace=True)
        #now use idmax to check best fret for every string
        fret = list(frame.idxmax(axis='columns'))
        #add fret to tablature
        tablature[frame_idx] = fret
    return tablature - 1  #decreasing by 1 so that 0 fret becomes -1 and 1 fret becomes 0


def make_squeezed_tab(tablature, n=9):
    """Make displayable squeezed tablature
        Args:
        tablature : the full tablature made from labels
        n : the frame squeeze window. Currently using same size as con-win
    """
    #now do it for all n frames
    squeezed_Tab = make_empty_tab()
    for batch_idx in range(0, tablature.shape[1], n):
        # for every frame_batch
        frame_batch = tablature.loc[:, batch_idx:batch_idx + n - 1]
        squeezed_Tab[batch_idx] = frame_batch.mode(axis='columns')[0]
    return squeezed_Tab.astype(int)


def make_dynamic_tab(full_tab, n=5):
    """Makes a dynamic tab from a full tab, squeezing frames which are same
        for n frames and retaining fret changes between frames. All other frets are removed

        Args:
        full_tab (pandas.DataFrame): a dataframe with the full tabs
        n (int, optional): the number of frames with equal frets to squeeze. Defaults to 5 (which means 0.2sec*5=1sec)
        Note : fret changes beteen frmes will be retained

        Returns:
        (pandas.DataFrame): The dynamically squeezed DataFrame
    """
    dynamic_tab = make_empty_tab()  # make the empty dynamic_tab
    col_read = 0  # set the red pointer to 0
    col_write = 0  # set the write pointer to 0
    total_frames = full_tab.shape[
        1]  # this is the maximum number of frames in the full tab
    while col_read < total_frames - 1:  #keep doing until you reach the last frame
        changed = False  # set changed as false
        max_col = col_read + n  # this is the window from current col + n cols ahead 5/0
        while ((not changed) and (col_read < max_col)
               and (col_read < total_frames - 1)
               ):  # only if not changed and full window not checked
            if not full_tab[col_read].equals(full_tab[col_read + 1]):
                # if current column doesnt equal next column, meaning change
                dynamic_tab[col_write] = full_tab[
                    col_read]  # write this col to the dynamic tab
                col_write = col_write + 1  # advance the write counter
                changed = True  # set changed to true
            col_read = col_read + 1  # advance the read counter, either ways
        if not changed:  # this means the entire window was checked but no change
            dynamic_tab[col_write] = full_tab[
                col_read - 1]  # so write the last entry of this window
            col_write = col_write + 1  # advance the write counter
    # below code is commented, as it sometimes causes problems with the last frame, need to debug, but still works fine without this section
    # #need to take care of the last column
    # if not dynamic_tab[col_write].equals(full_tab[col_read]):#if last written doesnt equal to last col in full_tab
    #     dynamic_tab[col_write + 1] = full_tab[col_read] #write this column
    return dynamic_tab


def str_row(row):
    """
    Takes a row of the dataframe and returns it as a processed string
    """
    row_list = row.values.tolist()[0]
    row_list = [str(item) for item in row_list]
    joined = ''.join(row_list)
    joined = joined.replace("-1", "-")  # Replaces -1 with -
    joined = '|'.join([joined[i:i + 16] for i in range(0, len(joined), 16)])
    return (joined)  # Returns long string


def print_tabs(tabs):
    """
    Takes a tabs dataframe and prints it as a formatted tablature
    """
    tab_list = ['e', 'B', 'G', 'D', 'A', 'E']
    len_long_row = len(str_row(tabs.loc[['e']]))
    num_div = 4  # Number of divisions per line
    num_lines = int(len_long_row / 17 // num_div +
                    1)  # Number of lines for the tab
    for line in range(0, num_lines):  # Iterates over lines
        for index in tab_list:  # Iterates over each index
            row = tabs.loc[[f'{index}']]
            long_row = str_row(row)
            print(f"{index}|{long_row[num_div*17*line:num_div*17*(line+1)]}")
        print()


def web_tabs(tabs):
    """
  Takes a tabs dataframe and returns a multilinear string to feed the app
  """
    tab_list = ['e', 'B', 'G', 'D', 'A', 'E']
    len_long_row = len(str_row(tabs.loc[['e']]))
    num_div = 4  # Number of divisions per line
    num_lines = int(len_long_row / 17 // num_div +
                    1)  # Number of lines for the tab
    line_list = []
    for line in range(0, num_lines):  # Iterates over lines
        for index in tab_list:  # Iterates over each index
            row = tabs.loc[[f'{index}']]
            long_row = str_row(row)
            line_list.append(
                f"{index}|{long_row[num_div*17*line:num_div*17*(line+1)]}")
        line_list.append("\n")
    multi_str = '\n'.join([line for line in line_list
                           ])  # joins all lines into a multistring
    return multi_str


def load_model_and_weights():
    my_tabcnn = TabCNN.TabCNN()
    model = my_tabcnn.build_model()
    model.load_weights(LOCAL_MODEL)
    return model


def load_x_new(filename):
    """function to load the filename to be processed into an x_new ready to be predicted
        Args:
        filename (str): the file path of the wav file to be processed
        Returns:
        [numpy.ndArray]: a numpy array of shape
        num frames x 192 x 9 (con_win) x 1
    """
    genrep = TabDataReprGen()
    x_new = genrep.load_rep_from_raw_file(filename)
    return x_new


if __name__ == "__main__":
    ##################################
    ########### LOAD MODEL & PREDICT ###########
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
