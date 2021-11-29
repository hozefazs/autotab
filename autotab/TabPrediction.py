import autotab.TabCNN as TabCNN
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


def full_tab(y_pred):
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
    for frame_idx in range(0, len(y_pred)):
        this_frame = pd.DataFrame(y_pred[frame_idx])
        tablature[str(frame_idx)] = this_frame.idxmax(axis='columns')[0]
    return tablature


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
