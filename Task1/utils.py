import csv
import numpy as np


def load_matrix(path):
    """
    Loads a substitution matrix that includes match, mismatch, and gap penalties from a CSV file

    Parameters:
        - path (str): filepath to the substitution matrix in CSV format

    Returns:
        - matrix_ (dict): substitution matrix in a form of a nested dictionary where each key is a nucleotide,
          and each value is a dictionary mapping other nucleotides to their scores
    """
    matrix_ = {}
    with open(path, 'r') as file:
        r = csv.reader(file)
        nucleotides1 = [header.strip() for header in next(r)[1:]]
        for row in r:
            nucleotide = row[0].strip()
            values = list(map(int, [x.strip() for x in row[1:]]))
            matrix_[nucleotide] = dict(zip(nucleotides1, values))
    return matrix_


def print_and_save_results(filename, alignments, title):
    """
    Prints and saves n optimal alignments to an output file

    Parameters:
        - filename (str): name of the output file
        - alignments (list): alignment sequences and their scores
        - title (str): string indicating the type of alignment ("Global", "Local").
    """
    with open(filename, 'w') as file:  # Changed to 'a' mode to append to the file
        for i, (a1, a2, score) in enumerate(alignments, 1):
            print(f"{title} alignment no. {i}:")
            print(f"{a1}")
            print(f"{a2}")
            print(f"Score: {score}\n")

            file.write(f"{title} alignment no. {i}:\n")
            file.write(f"{a1}\n")
            file.write(f"{a2}\n")
            file.write(f"Score: {score}\n\n")


def fill_matrices(sequence1, sequence2, substitution_matrix, GP, global_alignment=True):
    """
    Fills the scoring and direction matrices

    Parameters:
        - sequence1 (str): first DNA sequence
        - sequence2 (str): second DNA sequence
        - substitution_matrix (dict): dictionary that includes match, mismatch, and gap penalties
        - GP (int): gap penalty
        - global_alignment (bool): flag to indicate if it is a global alignment

    Returns:
        - scoring_matrix (np.ndarray): matrix containing scores for each alignment
        - direction_matrix (np.ndarray): directions for traceback ('d' for diagonal, 'u' for up, and 'l' for left)
    """
    x = len(sequence1) + 1
    y = len(sequence2) + 1
    scoring_matrix = np.zeros((x, y), dtype=int)
    direction_matrix = np.empty((x, y), dtype=object)

    for i in range(1, x):
        scoring_matrix[i][0] = scoring_matrix[i - 1][0] + GP if global_alignment else 0
        direction_matrix[i][0] = ['u'] if global_alignment else []

    for j in range(1, y):
        scoring_matrix[0][j] = scoring_matrix[0][j - 1] + GP if global_alignment else 0
        direction_matrix[0][j] = ['l'] if global_alignment else []

    for i in range(1, x):
        for j in range(1, y):
            diag = scoring_matrix[i - 1][j - 1] + substitution_matrix[sequence1[i - 1]][sequence2[j - 1]]
            up = scoring_matrix[i - 1][j] + GP
            left = scoring_matrix[i][j - 1] + GP

            max_score = max(diag, up, left)
            if not global_alignment:
                max_score = max(0, max_score)

            scoring_matrix[i][j] = max_score
            directions = []
            if diag == max_score:
                directions.append("d")
            if up == max_score:
                directions.append("u")
            if left == max_score:
                directions.append("l")
            direction_matrix[i][j] = directions

    return scoring_matrix, direction_matrix


def traceback(sequence1, sequence2, direction_matrix, scoring_matrix, i, j, alignment1, alignment2, alignments, n,
              global_alignment=True, start_position=None):
    """
    Recursively traces back to get optimal alignments.

    Parameters:
        - sequence1 (str): first DNA sequence
        - sequence2 (str): second DNA sequence
        - direction_matrix (np.ndarray): directions for traceback ('d' for diagonal, 'u' for up, and 'l' for left)
        - scoring_matrix (np.ndarray): matrix containing scores for each alignment
        - i (int): current row index
        - j (int): current column index
        - alignment1 (str): current alignment string for sequence1
        - alignment2 (str): current alignment string for sequence2
        - alignments (list): found alignments
        - n (int): maximum number of alignments
        - global_alignment (bool): flag to indicate if it is a global alignment
        - start_position (tuple): start position for local alignment
    """
    if not global_alignment and scoring_matrix[i][j] == 0:
        alignment = (alignment1, alignment2, scoring_matrix[start_position[0], start_position[1]])
        if alignment not in alignments:
            alignments.append(alignment)
        return
    elif global_alignment and i == 0 and j == 0:
        alignment = (alignment1, alignment2, scoring_matrix[len(sequence1), len(sequence2)])
        if alignment not in alignments:
            alignments.append(alignment)
        return

    if len(alignments) >= n:
        return

    for direction in direction_matrix[i][j]:
        if direction == "d" and i > 0 and j > 0:
            traceback(sequence1, sequence2, direction_matrix, scoring_matrix, i - 1, j - 1,
                      sequence1[i - 1] + alignment1, sequence2[j - 1] + alignment2, alignments, n, global_alignment,
                      start_position)
        elif direction == "u" and i > 0:
            traceback(sequence1, sequence2, direction_matrix, scoring_matrix, i - 1, j, sequence1[i - 1] + alignment1,
                      "-" + alignment2, alignments, n, global_alignment, start_position)
        elif direction == "l" and j > 0:
            traceback(sequence1, sequence2, direction_matrix, scoring_matrix, i, j - 1, "-" + alignment1,
                      sequence2[j - 1] + alignment2, alignments, n, global_alignment, start_position)
