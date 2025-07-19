import argparse
import sys
import time
import numpy as np
import pandas as pd
import multiprocessing
from numba import njit, prange


def check_blank(data, file_name):
    if data.iloc[:, 0].isnull().any():
        raise ValueError(f"'{file_name}' has blank row in the first column.")


def check_duplicate(data, file_name):
    if data.iloc[:, 0].duplicated().any():
        raise ValueError(f"'{file_name}' has duplicate object in the first column.")


def check_row_number(data1, data2):
    if len(data1) != len(data2):
        raise ValueError("Two files do not have the same number of rows.")


def check_order(data1, data2):
    list1 = data1.iloc[:, 0]
    list2 = data2.iloc[:, 0]
    if not list1.equals(list2):
        raise ValueError("Two files do not have the same order of objects in the first column.")


def comparison_filter(data_array, filter_val):

    row_sums = data_array.sum(axis=1)
    total_sum = row_sums.sum()
    threshold_sum = total_sum * filter_val

    sorted_indices = np.argsort(-row_sums)
    cumsum_vals = np.cumsum(row_sums[sorted_indices])

    keep_until = np.searchsorted(cumsum_vals, threshold_sum, side='right')

    data_filtered = data_array.copy()
    if keep_until < len(data_filtered):
        drop_indices = sorted_indices[keep_until:]
        data_filtered[drop_indices, :] = 0
    return data_filtered


def min_max_scale(matrix):

    row_min = matrix.min(axis=1, keepdims=True)
    row_max = matrix.max(axis=1, keepdims=True)
    diff = row_max - row_min

    diff[diff == 0] = 1

    scaled = (matrix - row_min) / diff
    return scaled

@njit
def slope_distance_np(row1, row2):
    if np.sum(row1) == 0 and np.sum(row2) == 0:
        return 0.0

    n = len(row1)
    distance = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            diff_A = row1[i] - row1[j]
            diff_B = row2[i] - row2[j]
            distance += 2 * abs(diff_A - diff_B)
    return distance


@njit(parallel=True)
def build_distance_matrix(data_matrix):

    N = data_matrix.shape[0]

    dist_matrix = np.empty((N, N), dtype=np.float64)
    for i in prange(N):
        for j in range(N):
            dist_matrix[i, j] = 0.0

    for i in prange(N):
        for j in range(i + 1, N):
            dist_ij = slope_distance_np(data_matrix[i], data_matrix[j])
            dist_matrix[i, j] = dist_ij
            dist_matrix[j, i] = dist_ij

    return dist_matrix

@njit(parallel=True)
def compute_degrees(dist_matrix1, dist_matrix2, pos_cor1, pos_cor2):

    N = dist_matrix1.shape[0]
    # strange bug when using np.zeros, sad.
    degrees = np.empty(N, dtype = np.float64)
    for i in range(N):
        degrees[i] = 0.0

    for i in prange(N):
        for j in range(N):
            if i == j:
                continue
            diff_condition1 = (dist_matrix1[i, j] < pos_cor1) and (dist_matrix2[i, j] >= pos_cor2)
            diff_condition2 = (dist_matrix1[i, j] >= pos_cor1) and (dist_matrix2[i, j] < pos_cor2)
            if diff_condition1 or diff_condition2:
                degrees[i] += 1
    return degrees


def process_filter_once(numeric_data1, numeric_data2, filter_val, pos_cor1, pos_cor2):


    filtered1 = comparison_filter(numeric_data1, filter_val)
    filtered2 = comparison_filter(numeric_data2, filter_val)

    scaled1 = min_max_scale(filtered1)
    scaled2 = min_max_scale(filtered2)

    dist_matrix1 = build_distance_matrix(scaled1)
    dist_matrix2 = build_distance_matrix(scaled2)

    degrees_arr = compute_degrees(dist_matrix1, dist_matrix2, pos_cor1, pos_cor2)
    edges = np.sum(degrees_arr) // 2

    return degrees_arr, edges


def main():
    start = time.perf_counter()

    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument('-in1', required=True, help='First input CSV file path')
    parser.add_argument('-in2', required=True, help='Second input CSV file path')
    parser.add_argument('-o', '--output', required=True, help='Output CSV file path')
    parser.add_argument('-pos1', type=float, default=None, help='Positive correlation threshold 1')
    parser.add_argument('-pos2', type=float, default=None, help='Positive correlation threshold 2')
    parser.add_argument('-start', type=float, default=0.95, help='Start filter value')
    parser.add_argument('-jump', type=float, default=0.005, help='Filter increment value')

    args = parser.parse_args()

    input_file1 = args.in1
    input_file2 = args.in2
    output_file = args.output
    start_filter = args.start
    jump_filter = args.jump

    try:
        df1 = pd.read_csv(input_file1, header=None)
        df2 = pd.read_csv(input_file2, header=None)
    except Exception as e:
        print(f"Error reading inputs: {e}")
        sys.exit(1)

    try:
        check_blank(df1, input_file1)
        check_blank(df2, input_file2)
        check_duplicate(df1, input_file1)
        check_duplicate(df2, input_file2)
        check_row_number(df1, df2)
        check_order(df1, df2)
    except ValueError as e:
        print(f"Data Check Error: {e}")
        sys.exit(1)


    col_num1 = len(df1.columns)
    col_num2 = len(df2.columns)

    if args.pos1 is None:
        pos_cor1 = col_num1 - 2
    else:
        pos_cor1 = args.pos1

    if args.pos2 is None:
        pos_cor2 = col_num2 - 2
    else:
        pos_cor2 = args.pos2

    filter_set = []
    current_val = start_filter
    while current_val <= 1.0:
        filter_set.append(round(current_val, 5))
        current_val += jump_filter

    object_names = df1.iloc[:, 0].values
    numeric_data1 = df1.iloc[:, 1:].to_numpy(dtype=np.float64)
    numeric_data2 = df2.iloc[:, 1:].to_numpy(dtype=np.float64)

    results_df = pd.DataFrame()

    for idx, fval in enumerate(filter_set):
        print(f"Processing filter={fval} ({idx + 1}/{len(filter_set)}) ...")
        degrees_arr, edges = process_filter_once(numeric_data1, numeric_data2, fval, pos_cor1, pos_cor2)

        header_row = [(f"Comparison Filter {fval}", "")]
        data_rows = [("Edges", edges)]
        data_edges = list(zip(object_names, degrees_arr))
        data_edges.sort(key=lambda x: x[1], reverse=True)
        data_rows.extend(data_edges)

        temp_df = pd.DataFrame(header_row + data_rows, columns=["Object", f"Score_{idx + 1}"])
        temp_df["Blank"] = ""

        start_col = idx * 3
        temp_df.columns = [
            f"Column_{start_col}",
            f"Column_{start_col + 1}",
            f"Column_{start_col + 2}"
        ]
        results_df = pd.concat([results_df, temp_df], axis=1)

    try:
        results_df.to_csv(output_file, index=False, header=False)
        print(f"Results successfully saved to {output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)

    print(f"total time needed: {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
