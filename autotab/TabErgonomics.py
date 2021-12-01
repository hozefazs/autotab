import numpy as np
import itertools

pitch_matrix = [
    [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57],
    [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62],
    [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67],
    [55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72],
    [59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76],
    [64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81]
]


def string_distance(curr, prev):
    if curr == prev:  #if no string changed
        return 0  # zero distance
    if curr == -1 or curr == 0 or prev == -1 or prev == 0:
        # this means that a string is moving from/to a non-played or non-fretted position
        return 1  # this is the smallest change
    return abs(curr - prev)


def frame_distance(curr, prev):
    frame_changes = list(map(string_distance, curr, prev))
    total_change = np.sum(frame_changes)
    return total_change


def frame_to_midi(frame):
    midi = []
    for i in range(0, len(frame)):  # for each string
        if frame[i] == -1:
            midi.append(-1)
        else:
            midi.append(pitch_matrix[i][frame[i]])
    return np.array(midi)


def combination_to_frames(combinations):
    result = []
    for combo in combinations:
        frame = {0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1}
        frame.update(dict(combo))
        result.append(list(frame.values()))
    return result


def get_all_combinations(frame):
    #convert frame to midi notes
    midi = frame_to_midi(frame)
    midi_notes = []
    for note in np.unique(midi):
        note_frets = []
        if note != -1:
            for idx in np.where(midi == note)[0]:
                note_frets.append((idx, frame[idx]))
            midi_notes.append(note_frets)
    combinations = list(itertools.product(*midi_notes))
    return combination_to_frames(combinations)


def best_frame(curr, prev):
    options = get_all_combinations(curr)
    prev_frames = np.tile(prev, (len(options), 1))
    distance_matrix = np.array(list(map(frame_distance, options, prev_frames)))
    best_options_idx = list(
        np.where(distance_matrix == distance_matrix.min())[0])
    best_options = []
    for idx in best_options_idx:
        best_options.append(options[idx])
    variances = list(map(np.var, best_options))
    lowest_var_option = best_options[np.argmin(variances)]
    return lowest_var_option


if __name__ == "__main__":
    frame_2 = [10, -1, 0, 7, 3, 3]
    frame_1 = [3, 5, 2, 7, 0, -1]
    print(best_frame(frame_2, frame_1))
