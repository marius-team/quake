# File: util_laet.py
import numpy as np
import csv
import os
import sys

def mmap_fvecs(fname):
    """ Memory-maps an fvecs file. """
    x = np.memmap(fname, dtype='int32', mode='r')
    d = x[0]
    return x.view('float32').reshape(-1, d + 1)[:, 1:]

def mmap_bvecs(fname):
    """ Memory-maps a bvecs file. """
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]

def read_tsv(filename, delimiter='\t', doInt=False, multi=1, rnd=-1):
    """
    Reads a TSV file into a list of lists.
    Handles potential errors during conversion and file access.
    """
    ret = []
    if not os.path.exists(filename):
        print(f"Warning: File not found {filename}", file=sys.stderr)
        return ret
    try:
        with open(filename, 'r', newline='') as tsvfile: # Use newline='' for universal newlines
            datareader = csv.reader(tsvfile, delimiter=delimiter)
            for row_num, row in enumerate(datareader):
                try:
                    l = [float(x) for x in row]
                    if multi != 1:
                        l = [multi * x for x in l]
                    if rnd != -1:
                        l = [round(x, rnd) for x in l]
                    if doInt:
                        l = [int(x) for x in l]
                    ret.append(l)
                except ValueError as e:
                    print(f"Skipping malformed row #{row_num+1} in {filename}: {row}. Error: {e}", file=sys.stderr)
                    continue
    except Exception as e:
        print(f"Error opening or reading file {filename}: {e}", file=sys.stderr)
    return ret

def write_tsv(data, filename, delimiter='\t', append=False, doInt=False, multi=1, rnd=-1):
    """
    Writes data to a TSV file.
    """
    if not append:
        try:
            os.remove(filename)
        except OSError:
            pass # File doesn't exist, or other error (permissions etc.)

    processed_data = []
    if doInt or multi != 1 or rnd != -1:
        for row_data in data:
            new_row = list(row_data) # Make a copy to avoid modifying original list objects
            if multi != 1:
                new_row = [multi * x for x in new_row]
            if rnd != -1:
                new_row = [round(x, rnd) for x in new_row]
            if doInt:
                new_row = [int(x) for x in new_row]
            processed_data.append(new_row)
    else:
        processed_data = data # Use data directly if no processing needed

    try:
        with open(filename, 'a', newline='') as tsvfile:
            writer = csv.writer(tsvfile, delimiter=delimiter)
            writer.writerows(processed_data)
    except Exception as e:
        print(f"Error writing to file {filename}: {e}", file=sys.stderr)