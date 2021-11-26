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
