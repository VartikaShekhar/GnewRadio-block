def read_bin_file(exp_file_path : str, dtype : np.dtype = np.float32, sample_rate : int = 522000, clip_length : int | None = None) -> Tuple[np.ndarray, float]:
    '''
    Read binary IQ data from a file and return the data as a complex64 numpy array
    Args:
        exp_file_path: Full File Path to the binary file containing IQ data
        dtype: Data type of the binary file
        sample_rate: Sampling frequency in Hz, default is 600000.
        clip_length: Length of the clip in seconds, if None, the whole file is read
    Returns:
        data_iq: Numpy array of complex64 data
        trace_duration: Duration of the trace in seconds
    '''

    ### POTENTIAL IMPROVEMENT : USE MEMMAP FOR LARGE FILES -> DOES NOT DUMP ALL TO RAM?
    data_raw = np.fromfile(exp_file_path, dtype=dtype)

    if clip_length is not None:
        num_complex_samples = int(round(clip_length * sample_rate))
        num_float32_samples = num_complex_samples * 2  # 2 float32s = 1 complex sample
        data_raw = data_raw[:num_float32_samples]

    data_iq = data_raw.view(np.complex64)
    trace_duration = len(data_iq) / sample_rate

    if clip_length is not None:
        print(f"[INFO] Requested: {clip_length}s → Got: {trace_duration:.3f}s")

    return data_iq, trace_duration
