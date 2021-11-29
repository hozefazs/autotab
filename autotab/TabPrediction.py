import autotab.TabCNN as TabCNN
import pandas as pd
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
        frame[0].replace(to_replace=list(frame[0][frame[0] < 0.9]),
                         value=0,
                         inplace=True)
        #now use idmax to check best fret for every string
        fret = list(frame.idxmax(axis='columns'))
        #add fret to tablature
        tablature[frame_idx] = fret
    return tablature


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
